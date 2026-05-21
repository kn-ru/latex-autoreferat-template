#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
ПРОГРАММА ДЛЯ РЕИДЕНТИФИКАЦИИ ЛИЧНОСТИ НА ОСНОВЕ БАЙЕСОВСКОГО ОБЪЕДИНЕНИЯ
ЭМБЕДДИНГОВ ЛИЦА И ТЕЛА

Свидетельство о государственной регистрации программы для ЭВМ
================================================================================


Описание:
    Программа реализует комплексную методику реидентификации людей в
    информационно-поисковых системах на основе байесовского объединения
    признаковых представлений (эмбеддингов) лица и силуэта.

    Основные компоненты:
    1. Модуль загрузки и подготовки данных — организация парных наборов
       изображений лица и силуэта, фильтрация общих классов, разбиение
       на обучающую и тестовую выборки (support/query).
    2. Модуль извлечения признаков — получение L2-нормализованных
       эмбеддингов лица (128-мерных) и силуэта (256-мерных) с помощью
       нейросетевых кодировщиков, обученных с функцией потерь ArcFace.
    3. Модуль оценки статистик классов — вычисление среднего вектора и
       ковариационной матрицы для каждого класса (идентичности) по каждой
       модальности на основе обучающей выборки (support set).
    4. Модуль байесовской классификации — реализация MAP-критерия
       принятия решения на основе многомерного нормального распределения:
           y* = argmax_y [ log P(f|y) + log P(s|y) ]
       где P(f|y) = N(f; μ_face_y, Σ_face_y),
           P(s|y) = N(s; μ_body_y, Σ_body_y).
    5. Модуль вычисления метрик качества — Accuracy, Top-k Accuracy,
       Precision, Recall, F1-мера, ROC AUC, FAR/FRR, EER, mAP.
    6. Модуль визуализации — ROC-кривые, PR-кривые, t-SNE визуализация
       эмбеддингов, матрицы ошибок, распределения вероятностей.
    7. Модуль сравнительного анализа — сопоставление четырёх стратегий:
       (а) только лицо, (б) только силуэт, (в) линейное объединение,
       (г) предложенное байесовское объединение.

Входные данные:
    - Датасет изображений лиц: директория, содержащая поддиректории
      с именами классов (идентичностей), в каждой — изображения лиц.
    - Датасет изображений силуэтов (reID): директория аналогичной
      структуры с изображениями полных фигур людей.
    - Предобученные модели кодировщиков лица и силуэта (.pt файлы).

Выходные данные:
    - CSV-файлы с метриками качества для каждого метода.
    - Графики ROC-кривых, PR-кривых, FAR/FRR-кривых.
    - Визуализации t-SNE эмбеддингов.
    - Матрицы ошибок (confusion matrices).
    - Сводная сравнительная таблица результатов.

Технические требования:
    - Python >= 3.8
    - PyTorch >= 1.10
    - NumPy, SciPy, scikit-learn, matplotlib, seaborn, Pillow, tqdm, pandas

Использование:
    python bayesian_reid_system.py \\
        --face_model <путь_к_модели_лица> \\
        --body_model <путь_к_модели_тела> \\
        --face_dir <путь_к_датасету_лиц> \\
        --body_dir <путь_к_датасету_силуэтов> \\
        --output_dir <путь_к_директории_результатов> \\
        [--face_emb_size 128] \\
        [--body_emb_size 256] \\
        [--input_size 112] \\
        [--batch_size 32] \\
        [--n_support 2] \\
        [--cov_reg 1e-5] \\
        [--seed 42]

================================================================================
"""

# =============================================================================
# ИМПОРТ БИБЛИОТЕК
# =============================================================================

import os
import sys
import time
import copy
import random
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from scipy.stats import multivariate_normal
from scipy import stats as scipy_stats

from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')  # Неинтерактивный backend для серверного окружения
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

# Подавление несущественных предупреждений
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# =============================================================================

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Настраивает систему логирования программы.

    Параметры:
        log_level : str
            Уровень логирования ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file : str, optional
            Путь к файлу для записи логов. Если None — вывод только в консоль.

    Возвращает:
        logging.Logger — настроенный объект логгера.
    """
    logger = logging.getLogger('BayesianReID')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Очищаем существующие обработчики, чтобы избежать дублирования
    logger.handlers.clear()
    logger.propagate = False

    # Формат сообщений
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Обработчик вывода в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Обработчик записи в файл (опционально)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Глобальный логгер
logger = setup_logging()


# =============================================================================
# КОНФИГУРАЦИЯ ПАРАМЕТРОВ
# =============================================================================

@dataclass
class ReIDConfig:
    """
    Конфигурация параметров системы реидентификации.

    Содержит все настраиваемые параметры эксперимента, включая
    пути к моделям и данным, размеры эмбеддингов, гиперпараметры
    байесовского классификатора и параметры визуализации.

    Атрибуты:
        face_model_path : str
            Путь к предобученной модели кодировщика лица (.pt).
        body_model_path : str
            Путь к предобученной модели кодировщика силуэта (.pt).
        face_dir : str
            Путь к директории с изображениями лиц.
        body_dir : str
            Путь к директории с изображениями силуэтов.
        output_dir : str
            Путь к директории для сохранения результатов.
        face_emb_size : int
            Размерность вектора эмбеддинга лица (по умолчанию 128).
        body_emb_size : int
            Размерность вектора эмбеддинга силуэта (по умолчанию 256).
        input_size : int
            Размер входного изображения для нейросети (по умолчанию 112).
        batch_size : int
            Размер мини-батча для извлечения признаков.
        num_workers : int
            Количество рабочих потоков загрузчика данных.
        n_support : int
            Количество образцов в support-множестве для каждого класса.
        cov_reg : float
            Коэффициент регуляризации ковариационной матрицы (λ·I).
        random_seed : int
            Фиксированное зерно для воспроизводимости результатов.
        tsne_perplexity : int
            Параметр perplexity для t-SNE визуализации.
        tsne_n_iter : int
            Количество итераций t-SNE.
        top_k_values : list
            Значения k для вычисления Top-k Accuracy.
        confidence_level : float
            Уровень доверительного интервала (по умолчанию 0.95).
        top_n_classes_tsne : int
            Количество классов для t-SNE визуализации.
    """
    # --- Пути ---
    face_model_path: str = "/home/knru/Enface/insightface/recognition/arcface_torch/work_dirs/dissertation_face_mbf/model_face.pt"
    body_model_path: str = "/home/knru/Enface/insightface/recognition/arcface_torch/work_dirs/dissertation_reid_mbf/model_body.pt"
    face_dir: str = "/media/knru/DataLake/80lab/dataset_person/CUHK03/archive/DATASET/val/Face/"
    body_dir: str = "/media/knru/DataLake/80lab/dataset_person/CUHK03/archive/DATASET/val/reID/"
    output_dir: str = "reid_results"

    # --- Параметры модели ---
    face_emb_size: int = 128
    body_emb_size: int = 256
    input_size: int = 112
    batch_size: int = 32
    num_workers: int = 4

    # --- Параметры байесовского классификатора ---
    n_support: int = 2
    cov_reg: float = 1e-5

    # --- Воспроизводимость ---
    random_seed: int = 42

    # --- Визуализация ---
    tsne_perplexity: int = 30
    tsne_n_iter: int = 2000
    top_k_values: list = field(default_factory=lambda: [1, 3, 5, 10])
    confidence_level: float = 0.95
    top_n_classes_tsne: int = 10


def set_global_seed(seed: int) -> None:
    """
    Устанавливает фиксированное зерно генератора случайных чисел
    для всех используемых библиотек, обеспечивая воспроизводимость
    результатов экспериментов.

    Параметры:
        seed : int
            Значение зерна генератора случайных чисел.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Зерно генератора случайных чисел установлено: {seed}")


# =============================================================================
# МОДУЛЬ ЗАГРУЗКИ И ПОДГОТОВКИ ДАННЫХ
# =============================================================================

class ReIDDataset(Dataset):
    """
    Датасет для задачи реидентификации личности.

    Загружает изображения из директории, организованной по классам
    (идентичностям). Каждая поддиректория соответствует одному классу
    и содержит изображения данной личности.

    Структура директории:
        root_dir/
        ├── person_001/
        │   ├── img_001.jpg
        │   ├── img_002.jpg
        │   └── ...
        ├── person_002/
        │   ├── img_001.jpg
        │   └── ...
        └── ...

    Параметры:
        root_dir : str
            Корневая директория с данными.
        input_size : int
            Размер выходного изображения (input_size × input_size).
        augment : bool
            Применять ли аугментацию данных (по умолчанию False).

    Атрибуты:
        classes : list
            Отсортированный список имён классов.
        class_to_idx : dict
            Отображение имени класса в числовой индекс.
        samples : list
            Список кортежей (путь_к_файлу, индекс_класса, имя_класса).
    """

    # Допустимые расширения файлов изображений
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    def __init__(self, root_dir: str, input_size: int = 112, augment: bool = False):
        self.root_dir = root_dir
        self.input_size = input_size

        # Определяем преобразования изображений
        transform_list = [
            transforms.Resize((input_size, input_size)),
        ]

        # Аугментация данных (опционально)
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])

        transform_list.extend([
            transforms.ToTensor(),
            # Нормализация в диапазон [-1, 1] для совместимости с ArcFace
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.transform = transforms.Compose(transform_list)

        # Сканирование директории: извлечение классов и образцов
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Формирование списка всех образцов
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.lower().endswith(self.SUPPORTED_EXTENSIONS):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((
                        img_path,
                        self.class_to_idx[class_name],
                        class_name
                    ))

        logger.info(
            f"Датасет загружен: {len(self.samples)} изображений, "
            f"{len(self.classes)} классов из {root_dir}"
        )

    def __len__(self) -> int:
        """Возвращает общее количество образцов в датасете."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Возвращает образец по индексу.

        Параметры:
            idx : int — индекс образца.

        Возвращает:
            Кортеж (тензор_изображения, метка_класса, имя_класса).
        """
        img_path, label, class_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label, class_name

    def get_class_names(self) -> List[str]:
        """Возвращает отсортированный список имён классов."""
        return self.classes.copy()

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Возвращает распределение количества образцов по классам.

        Возвращает:
            dict — словарь {имя_класса: количество_образцов}.
        """
        distribution = defaultdict(int)
        for _, _, class_name in self.samples:
            distribution[class_name] += 1
        return dict(distribution)


# =============================================================================
# МОДУЛЬ ЗАГРУЗКИ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ
# =============================================================================

def load_embedding_model(
    model_path: str,
    embedding_size: int,
    input_size: int = 112,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Загружает предобученную нейросетевую модель-кодировщик из файла
    и подготавливает её к режиму извлечения признаков (inference).

    Модель загружается в режиме eval() с замороженными параметрами,
    что исключает обновление весов и снижает потребление памяти.

    Параметры:
        model_path : str
            Путь к файлу весов модели (.pt).
        embedding_size : int
            Размерность выходного вектора эмбеддинга.
        input_size : int
            Размер входного изображения (input_size × input_size).
        device : torch.device, optional
            Вычислительное устройство. Если None — автоматический выбор.

    Возвращает:
        nn.Module — загруженная модель в режиме inference.

    Исключения:
        FileNotFoundError — если файл модели не найден.
        RuntimeError — если произошла ошибка загрузки весов.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Добавляем путь к проекту для импорта архитектуры модели
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from backbones.mobilefacenet import MobileFaceNet
    except ImportError:
        raise ImportError(
            "Не удалось импортировать MobileFaceNet. "
            "Убедитесь, что модуль backbones.mobilefacenet доступен."
        )

    # Инициализация модели
    model = MobileFaceNet(
        fp16=False,
        num_features=embedding_size,
        scale=1,
        input_size=input_size
    )

    # Загрузка весов
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Перевод в режим inference
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)

    # Подсчёт параметров модели
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Модель загружена: {model_path} "
        f"({total_params:,} параметров, эмбеддинг {embedding_size}-d, "
        f"устройство: {device})"
    )

    return model


# =============================================================================
# МОДУЛЬ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ (ЭМБЕДДИНГОВ)
# =============================================================================

@dataclass
class EmbeddingResult:
    """
    Результат извлечения эмбеддингов из датасета.

    Атрибуты:
        features : np.ndarray
            Матрица эмбеддингов размера (N, D), где N — число образцов,
            D — размерность эмбеддинга.
        labels : np.ndarray
            Вектор числовых меток классов размера (N,).
        class_names : np.ndarray
            Вектор строковых имён классов размера (N,).
        unique_class_names : list
            Упорядоченный список уникальных имён классов.
    """
    features: np.ndarray
    labels: np.ndarray
    class_names: np.ndarray
    unique_class_names: List[str]


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    description: str = "Извлечение эмбеддингов"
) -> EmbeddingResult:
    """
    Извлекает L2-нормализованные эмбеддинги из всех изображений
    в датасете с помощью указанной нейросетевой модели.

    L2-нормализация обеспечивает, что все эмбеддинги лежат на
    единичной гиперсфере, что необходимо для корректного
    вычисления косинусного сходства и байесовской классификации.

    Параметры:
        model : nn.Module
            Нейросетевая модель-кодировщик.
        dataloader : DataLoader
            Загрузчик данных.
        description : str
            Описание для индикатора прогресса.

    Возвращает:
        EmbeddingResult — структура с эмбеддингами, метками и именами.
    """
    device = next(model.parameters()).device
    all_features = []
    all_labels = []
    all_class_names = []

    # Словарь для восстановления соответствия метка → имя класса
    label_to_name = {}

    with torch.no_grad():
        for images, labels, class_names in tqdm(dataloader, desc=description):
            images = images.to(device)

            # Прямой проход через модель → получение эмбеддингов
            embeddings = model(images)

            # L2-нормализация: ||e||_2 = 1 для каждого эмбеддинга
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Перенос на CPU и преобразование в NumPy
            embeddings_np = embeddings.cpu().numpy()
            labels_np = labels.numpy()

            # Обновление словаря меток
            for i, lbl in enumerate(labels_np):
                label_to_name[int(lbl)] = class_names[i]

            all_features.append(embeddings_np)
            all_labels.append(labels_np)
            all_class_names.extend(class_names)

    # Объединение батчей
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    class_names_arr = np.array(all_class_names)

    # Формирование списка уникальных имён классов
    unique_labels = np.unique(labels)
    unique_class_names = [label_to_name[int(lbl)] for lbl in unique_labels]

    logger.info(
        f"{description}: {features.shape[0]} образцов, "
        f"{len(unique_class_names)} классов, "
        f"размерность эмбеддинга: {features.shape[1]}"
    )

    return EmbeddingResult(
        features=features,
        labels=labels,
        class_names=class_names_arr,
        unique_class_names=unique_class_names
    )


# =============================================================================
# МОДУЛЬ ФИЛЬТРАЦИИ И РАЗБИЕНИЯ ДАННЫХ
# =============================================================================

def find_common_classes(
    face_class_names: np.ndarray,
    body_class_names: np.ndarray
) -> List[str]:
    """
    Определяет пересечение множеств классов двух модальностей.

    В задаче реидентификации необходимо, чтобы для каждой
    идентичности были доступны оба типа признаков: лицевые
    и силуэтные. Данная функция находит общие классы.

    Параметры:
        face_class_names : np.ndarray
            Массив имён классов из датасета лиц.
        body_class_names : np.ndarray
            Массив имён классов из датасета силуэтов.

    Возвращает:
        list — отсортированный список общих классов.
    """
    face_unique = set(np.unique(face_class_names))
    body_unique = set(np.unique(body_class_names))
    common = sorted(face_unique & body_unique)

    logger.info(
        f"Пересечение классов: {len(common)} общих из "
        f"{len(face_unique)} (лица) и {len(body_unique)} (силуэты)"
    )

    return common


def filter_by_common_classes(
    emb_result: EmbeddingResult,
    common_classes: List[str],
    class_name_to_label: Dict[str, int]
) -> EmbeddingResult:
    """
    Фильтрует эмбеддинги, оставляя только образцы общих классов,
    и переназначает метки в соответствии с общим порядком классов.

    Параметры:
        emb_result : EmbeddingResult
            Исходные эмбеддинги.
        common_classes : list
            Список общих классов.
        class_name_to_label : dict
            Отображение имени класса в новый числовой индекс.

    Возвращает:
        EmbeddingResult — отфильтрованные эмбеддинги.
    """
    common_set = set(common_classes)

    # Маска для фильтрации
    mask = np.array([name in common_set for name in emb_result.class_names])

    filtered_features = emb_result.features[mask]
    filtered_class_names = emb_result.class_names[mask]
    filtered_labels = np.array([
        class_name_to_label[name] for name in filtered_class_names
    ])

    logger.info(
        f"Фильтрация: {filtered_features.shape[0]} образцов "
        f"из {emb_result.features.shape[0]} (оставлено {len(common_classes)} классов)"
    )

    return EmbeddingResult(
        features=filtered_features,
        labels=filtered_labels,
        class_names=filtered_class_names,
        unique_class_names=common_classes
    )


@dataclass
class SupportQuerySplit:
    """
    Результат разбиения данных на support и query множества.

    Support — «база» (известные образцы каждой идентичности),
    по которым строятся статистики классов.
    Query — «запросы» (новые наблюдения), для которых определяется
    идентичность.

    Атрибуты:
        support_face_features : np.ndarray — эмбеддинги лиц (support).
        support_face_labels : np.ndarray — метки классов лиц (support).
        support_body_features : np.ndarray — эмбеддинги силуэтов (support).
        support_body_labels : np.ndarray — метки классов силуэтов (support).
        query_face_features : np.ndarray — эмбеддинги лиц (query).
        query_face_labels : np.ndarray — метки классов лиц (query).
        query_body_features : np.ndarray — эмбеддинги силуэтов (query).
        query_body_labels : np.ndarray — метки классов силуэтов (query).
        valid_classes : list — список классов, прошедших фильтрацию.
        class_name_to_label : dict — отображение имени в числовой индекс.
    """
    support_face_features: np.ndarray
    support_face_labels: np.ndarray
    support_body_features: np.ndarray
    support_body_labels: np.ndarray
    query_face_features: np.ndarray
    query_face_labels: np.ndarray
    query_body_features: np.ndarray
    query_body_labels: np.ndarray
    valid_classes: List[str]
    class_name_to_label: Dict[str, int]


def split_support_query(
    face_emb: EmbeddingResult,
    body_emb: EmbeddingResult,
    n_support: int = 2,
    random_seed: int = 42
) -> SupportQuerySplit:
    """
    Разбивает данные на support (обучающие) и query (тестовые) множества
    с синхронизацией по обеим модальностям.

    Для каждого общего класса:
    - Проверяется наличие минимум n_support образцов в каждой модальности.
    - Случайно выбираются n_support образцов для support.
    - Оставшиеся образцы формируют query (с балансировкой по минимуму).

    Параметры:
        face_emb : EmbeddingResult — эмбеддинги лиц.
        body_emb : EmbeddingResult — эмбеддинги силуэтов.
        n_support : int — количество образцов в support для каждого класса.
        random_seed : int — зерно для воспроизводимости разбиения.

    Возвращает:
        SupportQuerySplit — структура с разбитыми данными.
    """
    np.random.seed(random_seed)

    # Пересечение классов
    common_classes = find_common_classes(face_emb.class_names, body_emb.class_names)

    valid_classes = []
    face_support_idx, face_query_idx = [], []
    body_support_idx, body_query_idx = [], []

    for class_name in common_classes:
        # Индексы образцов данного класса в каждой модальности
        face_idxs = np.where(face_emb.class_names == class_name)[0]
        body_idxs = np.where(body_emb.class_names == class_name)[0]

        # Проверка достаточности образцов для support
        if len(face_idxs) < n_support or len(body_idxs) < n_support:
            continue

        # Случайный выбор support-индексов
        f_support = np.random.choice(face_idxs, n_support, replace=False)
        b_support = np.random.choice(body_idxs, n_support, replace=False)

        # Query — оставшиеся образцы
        f_query = [i for i in face_idxs if i not in f_support]
        b_query = [i for i in body_idxs if i not in b_support]

        # Синхронизация длины query по минимуму двух модальностей
        n_query = min(len(f_query), len(b_query))
        if n_query > 0:
            face_support_idx.extend(f_support)
            body_support_idx.extend(b_support)
            face_query_idx.extend(f_query[:n_query])
            body_query_idx.extend(b_query[:n_query])
            valid_classes.append(class_name)

    # Переиндексация меток: общий порядок для валидных классов
    class_name_to_label = {name: idx for idx, name in enumerate(valid_classes)}

    # Формирование массивов
    face_support_idx = np.array(face_support_idx)
    face_query_idx = np.array(face_query_idx)
    body_support_idx = np.array(body_support_idx)
    body_query_idx = np.array(body_query_idx)

    result = SupportQuerySplit(
        support_face_features=face_emb.features[face_support_idx],
        support_face_labels=np.array([
            class_name_to_label[face_emb.class_names[i]] for i in face_support_idx
        ]),
        support_body_features=body_emb.features[body_support_idx],
        support_body_labels=np.array([
            class_name_to_label[body_emb.class_names[i]] for i in body_support_idx
        ]),
        query_face_features=face_emb.features[face_query_idx],
        query_face_labels=np.array([
            class_name_to_label[face_emb.class_names[i]] for i in face_query_idx
        ]),
        query_body_features=body_emb.features[body_query_idx],
        query_body_labels=np.array([
            class_name_to_label[body_emb.class_names[i]] for i in body_query_idx
        ]),
        valid_classes=valid_classes,
        class_name_to_label=class_name_to_label,
    )

    logger.info(
        f"Разбиение support/query: {len(valid_classes)} валидных классов, "
        f"support: {len(face_support_idx)} (лица) / {len(body_support_idx)} (тела), "
        f"query: {len(face_query_idx)} (лица) / {len(body_query_idx)} (тела)"
    )

    return result


# =============================================================================
# МОДУЛЬ ОЦЕНКИ СТАТИСТИК КЛАССОВ
# =============================================================================

@dataclass
class ClassStatistics:
    """
    Статистики одного класса для одной модальности.

    Атрибуты:
        mean : np.ndarray — средний вектор (центроид) класса.
        cov : np.ndarray — ковариационная матрица класса.
        std : np.ndarray — вектор стандартных отклонений.
        n_samples : int — количество образцов класса.
        ci_lower : np.ndarray — нижняя граница доверительного интервала.
        ci_upper : np.ndarray — верхняя граница доверительного интервала.
    """
    mean: np.ndarray
    cov: np.ndarray
    std: np.ndarray
    n_samples: int
    ci_lower: np.ndarray
    ci_upper: np.ndarray


def compute_class_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    cov_reg: float = 1e-5,
    confidence_level: float = 0.95
) -> Dict[int, ClassStatistics]:
    """
    Вычисляет статистические параметры распределения эмбеддингов
    для каждого класса: средний вектор (центроид), ковариационную
    матрицу и доверительный интервал.

    Ковариационная матрица регуляризуется добавлением λ·I для
    обеспечения положительной определённости (предотвращение
    вырожденности при малом числе образцов).

    Математическая модель (формула 23 диссертации):
        P(x|y) = N(x; μ_y, Σ_y)
        Σ_y = Cov(X_y) + λ · I

    Параметры:
        features : np.ndarray — матрица эмбеддингов (N, D).
        labels : np.ndarray — вектор меток (N,).
        cov_reg : float — коэффициент регуляризации λ.
        confidence_level : float — уровень доверительного интервала.

    Возвращает:
        dict — словарь {метка_класса: ClassStatistics}.
    """
    class_stats = {}
    unique_labels = np.unique(labels)
    dim = features.shape[1]

    for label in unique_labels:
        # Выборка эмбеддингов текущего класса
        class_features = features[labels == label]
        n = len(class_features)

        # Средний вектор (центроид)
        mean = np.mean(class_features, axis=0)

        # Ковариационная матрица с регуляризацией
        if n > 1:
            cov = np.cov(class_features, rowvar=False)
            cov += cov_reg * np.eye(dim)
        else:
            # При единственном образце — изотропная ковариация
            cov = cov_reg * np.eye(dim)

        # Стандартное отклонение
        std = np.std(class_features, axis=0)

        # Доверительный интервал для среднего
        if n > 1:
            sem = std / np.sqrt(n)  # Стандартная ошибка среднего
            t_crit = scipy_stats.t.ppf((1 + confidence_level) / 2, n - 1)
            margin = sem * t_crit
        else:
            margin = np.zeros(dim)

        class_stats[int(label)] = ClassStatistics(
            mean=mean,
            cov=cov,
            std=std,
            n_samples=n,
            ci_lower=mean - margin,
            ci_upper=mean + margin,
        )

    logger.info(
        f"Статистики классов: {len(class_stats)} классов, "
        f"размерность: {dim}, регуляризация: {cov_reg}"
    )

    return class_stats


# =============================================================================
# МОДУЛЬ БАЙЕСОВСКОЙ КЛАССИФИКАЦИИ
# =============================================================================

def bayesian_map_classify(
    face_features: np.ndarray,
    body_features: np.ndarray,
    face_stats: Dict[int, ClassStatistics],
    body_stats: Dict[int, ClassStatistics],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Реализация байесовского MAP-классификатора для реидентификации
    личности по объединению двух модальностей (лицо и силуэт).

    Решающее правило (формула 22 диссертации):
        y* = argmax_y [ log P(f|y) + log P(s|y) ]

    где правдоподобие каждой модальности моделируется многомерным
    нормальным распределением (формула 23):
        P(f|y) = N(f; μ^face_y, Σ^face_y)
        P(s|y) = N(s; μ^body_y, Σ^body_y)

    Логарифм правдоподобия:
        log P(x|y) = -0.5 · [(x-μ)^T Σ^{-1} (x-μ) + log|Σ| + d·log(2π)]

    Суммирование логарифмов (вместо умножения вероятностей) обеспечивает:
    - Численную устойчивость при малых вероятностях.
    - Автоматическое взвешивание модальностей: модальность с меньшей
      дисперсией (более информативная) вносит больший вклад.

    Параметры:
        face_features : np.ndarray — эмбеддинги лиц (N, D_face).
        body_features : np.ndarray — эмбеддинги силуэтов (N, D_body).
        face_stats : dict — статистики классов для лиц.
        body_stats : dict — статистики классов для силуэтов.

    Возвращает:
        predictions : np.ndarray — предсказанные метки классов (N,).
        log_probs : np.ndarray — матрица логарифмов правдоподобий (N, K),
            где K — число классов.
    """
    n_samples = face_features.shape[0]
    classes = sorted(face_stats.keys())
    n_classes = len(classes)

    # Матрица логарифмов совместного правдоподобия
    log_probs = np.zeros((n_samples, n_classes))

    for col_idx, cls in enumerate(classes):
        # Правдоподобие по лицевой модальности: log P(f | y=cls)
        face_mvn = multivariate_normal(
            mean=face_stats[cls].mean,
            cov=face_stats[cls].cov,
            allow_singular=True
        )
        log_p_face = face_mvn.logpdf(face_features)

        # Правдоподобие по силуэтной модальности: log P(s | y=cls)
        body_mvn = multivariate_normal(
            mean=body_stats[cls].mean,
            cov=body_stats[cls].cov,
            allow_singular=True
        )
        log_p_body = body_mvn.logpdf(body_features)

        # Сумма логарифмов = логарифм произведения правдоподобий
        # log P(f,s|y) = log P(f|y) + log P(s|y)  (при условии независимости)
        log_probs[:, col_idx] = log_p_face + log_p_body

    # MAP-решение: argmax по классам
    predictions = np.argmax(log_probs, axis=1)

    return predictions, log_probs


def nearest_neighbor_classify(
    query_features: np.ndarray,
    support_features: np.ndarray,
    support_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Классификация методом ближайшего соседа по косинусному сходству.

    Для каждого query-эмбеддинга вычисляется косинусное сходство
    со всеми support-эмбеддингами; предсказанный класс определяется
    как класс ближайшего (наиболее похожего) support-образца.

    Параметры:
        query_features : np.ndarray — эмбеддинги запросов (N_q, D).
        support_features : np.ndarray — эмбеддинги базы (N_s, D).
        support_labels : np.ndarray — метки классов базы (N_s,).

    Возвращает:
        predictions : np.ndarray — предсказанные метки (N_q,).
        similarity_matrix : np.ndarray — матрица сходств (N_q, N_s).
    """
    # L2-нормализация для вычисления косинусного сходства
    query_norm = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    support_norm = support_features / np.linalg.norm(support_features, axis=1, keepdims=True)

    # Матрица косинусных сходств
    similarity_matrix = np.dot(query_norm, support_norm.T)

    # Ближайший сосед
    nn_indices = np.argmax(similarity_matrix, axis=1)
    predictions = support_labels[nn_indices]

    return predictions, similarity_matrix


# =============================================================================
# МОДУЛЬ ВЫЧИСЛЕНИЯ МЕТРИК КАЧЕСТВА
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Вычисляет функцию softmax для преобразования логитов в вероятности.

    Реализация с вычитанием максимума для численной устойчивости:
        softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))

    Параметры:
        x : np.ndarray — входной массив логитов.
        axis : int — ось, по которой применяется softmax.

    Возвращает:
        np.ndarray — массив вероятностей (сумма по axis равна 1).
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


@dataclass
class ClassificationMetrics:
    """
    Набор метрик качества классификации.

    Атрибуты:
        accuracy : float — доля правильных ответов (Top-1).
        top_k_accuracy : dict — словарь {k: Top-k Accuracy}.
        precision_macro : float — точность (макро-усреднение).
        recall_macro : float — полнота (макро-усреднение).
        f1_macro : float — F1-мера (макро-усреднение).
        precision_weighted : float — точность (взвешенная).
        recall_weighted : float — полнота (взвешенная).
        f1_weighted : float — F1-мера (взвешенная).
        roc_auc_micro : float — площадь под ROC-кривой (микро).
        roc_auc_macro : float — площадь под ROC-кривой (макро).
        method_name : str — название метода.
    """
    accuracy: float = 0.0
    top_k_accuracy: Dict[int, float] = field(default_factory=dict)
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_weighted: float = 0.0
    roc_auc_micro: float = 0.0
    roc_auc_macro: float = 0.0
    method_name: str = ""


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_probs: Optional[np.ndarray] = None,
    method_name: str = "",
    top_k_values: List[int] = None,
) -> ClassificationMetrics:
    """
    Вычисляет полный набор метрик качества многоклассовой классификации.

    Метрики включают:
    - Top-1 и Top-k Accuracy
    - Precision, Recall, F1-мера (macro и weighted усреднения)
    - ROC AUC (micro и macro для one-vs-rest)

    Параметры:
        y_true : np.ndarray — истинные метки классов.
        y_pred : np.ndarray — предсказанные метки классов.
        log_probs : np.ndarray, optional — матрица логарифмов вероятностей.
        method_name : str — название метода для логирования.
        top_k_values : list — значения k для Top-k Accuracy.

    Возвращает:
        ClassificationMetrics — структура со всеми метриками.
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    metrics = ClassificationMetrics(method_name=method_name)

    # === Top-1 Accuracy ===
    metrics.accuracy = accuracy_score(y_true, y_pred)

    # === Top-k Accuracy ===
    if log_probs is not None:
        # Преобразование log-вероятностей в вероятности (softmax)
        probs = softmax(log_probs, axis=1)

        # Обработка некорректных значений
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums

        n_classes = log_probs.shape[1]
        for k in top_k_values:
            if k <= n_classes:
                try:
                    metrics.top_k_accuracy[k] = top_k_accuracy_score(
                        y_true, probs, k=k,
                        labels=list(range(n_classes))
                    )
                except Exception:
                    metrics.top_k_accuracy[k] = 0.0

    # === Precision, Recall, F1 ===
    metrics.precision_macro = precision_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics.recall_macro = recall_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics.f1_macro = f1_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics.precision_weighted = precision_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics.recall_weighted = recall_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics.f1_weighted = f1_score(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # === ROC AUC ===
    if log_probs is not None:
        try:
            n_classes = log_probs.shape[1]
            y_onehot = np.zeros((len(y_true), n_classes))
            for i, lbl in enumerate(y_true):
                if lbl < n_classes:
                    y_onehot[i, int(lbl)] = 1

            metrics.roc_auc_micro = roc_auc_score(
                y_onehot, probs, average='micro'
            )
            metrics.roc_auc_macro = roc_auc_score(
                y_onehot, probs, average='macro'
            )
        except Exception as e:
            logger.warning(f"Ошибка ROC AUC ({method_name}): {e}")
            metrics.roc_auc_micro = 0.0
            metrics.roc_auc_macro = 0.0

    # Логирование
    logger.info(
        f"[{method_name}] Accuracy={metrics.accuracy:.4f}, "
        f"Top-5={metrics.top_k_accuracy.get(5, 0.0):.4f}, "
        f"Precision={metrics.precision_macro:.4f}, "
        f"Recall={metrics.recall_macro:.4f}, "
        f"F1={metrics.f1_macro:.4f}, "
        f"ROC-AUC(micro)={metrics.roc_auc_micro:.4f}"
    )

    return metrics


# =============================================================================
# МОДУЛЬ ВИЗУАЛИЗАЦИИ РЕЗУЛЬТАТОВ
# =============================================================================

# Настройка стиля графиков для публикации
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
sns.set_style("whitegrid")


def plot_roc_curves(
    probs_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: str,
    n_classes: Optional[int] = None
) -> None:
    """
    Строит ROC-кривые (micro-average) для нескольких методов
    на одном графике для сравнительного анализа.

    Параметры:
        probs_dict : dict — {название_метода: матрица_вероятностей (N, K)}.
        y_true : np.ndarray — истинные метки классов (N,).
        output_path : str — путь для сохранения графика.
        n_classes : int, optional — число классов.
    """
    if n_classes is None:
        n_classes = next(iter(probs_dict.values())).shape[1]

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Словарь для русских названий методов
    rus_names = {
        'Байес (MAP)': 'Байес (MAP)',
        'Линейное объединение': 'Линейное объединение',
        'Только лицо': 'Только лицо',
        'Только силуэт': 'Только силуэт',
    }

    plt.figure(figsize=(10, 8))

    for name, probs in probs_dict.items():
        # Micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        display_name = rus_names.get(name, name)
        plt.plot(
            fpr_micro, tpr_micro, lw=2,
            label=f"{display_name} (AUC = {roc_auc_micro:.3f})"
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Случайный классификатор')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Доля ложных срабатываний (FPR)')
    plt.ylabel('Доля истинных срабатываний (TPR)')
    plt.title('ROC-кривые (micro-average)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"ROC-кривые сохранены: {output_path}")


def plot_precision_recall_curves(
    probs_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: str,
    n_classes: Optional[int] = None
) -> None:
    """
    Строит кривые Precision-Recall (micro-average) для нескольких
    методов на одном графике.

    Параметры:
        probs_dict : dict — {название_метода: матрица_вероятностей (N, K)}.
        y_true : np.ndarray — истинные метки классов (N,).
        output_path : str — путь для сохранения графика.
        n_classes : int, optional — число классов.
    """
    if n_classes is None:
        n_classes = next(iter(probs_dict.values())).shape[1]

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(10, 8))

    for name, probs in probs_dict.items():
        precision_micro, recall_micro, _ = precision_recall_curve(
            y_bin.ravel(), probs.ravel()
        )
        ap_micro = average_precision_score(y_bin, probs, average='micro')
        plt.plot(
            recall_micro, precision_micro, lw=2,
            label=f"{name} (AP = {ap_micro:.3f})"
        )

    plt.xlabel('Полнота (Recall)')
    plt.ylabel('Точность (Precision)')
    plt.title('Кривые Precision-Recall (micro-average)')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"PR-кривые сохранены: {output_path}")


def plot_tsne_embeddings(
    embeddings_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    class_names: List[str],
    output_dir: str,
    top_n: int = 10,
    perplexity: int = 30,
    n_iter: int = 2000,
    random_seed: int = 42
) -> None:
    """
    Визуализирует эмбеддинги методом t-SNE для top-N наиболее
    частых классов.

    t-SNE (t-distributed Stochastic Neighbor Embedding) проецирует
    высокоразмерные эмбеддинги на двумерную плоскость с сохранением
    локальной структуры данных, что позволяет оценить качество
    кластеризации и межклассовое разделение.

    Параметры:
        embeddings_dict : dict — {название: эмбеддинги (N, D)}.
        y_true : np.ndarray — метки классов (N,).
        class_names : list — список имён классов.
        output_dir : str — директория для сохранения.
        top_n : int — количество классов для визуализации.
        perplexity : int — параметр perplexity t-SNE.
        n_iter : int — число итераций t-SNE.
        random_seed : int — зерно генератора.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Выбор top-N классов по частоте
    unique, counts = np.unique(y_true, return_counts=True)
    top_classes = unique[np.argsort(-counts)][:top_n]
    top_class_set = set(top_classes)

    # Фильтрация по top-N
    mask = np.array([lbl in top_class_set for lbl in y_true])

    for name, embeddings in embeddings_dict.items():
        emb_sel = embeddings[mask]
        labels_sel = y_true[mask]

        if len(emb_sel) < 2:
            logger.warning(f"Недостаточно данных для t-SNE ({name})")
            continue

        # Адаптация perplexity к количеству образцов
        effective_perplexity = min(perplexity, len(emb_sel) - 1)

        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            max_iter=n_iter,
            random_state=random_seed
        )
        emb_2d = tsne.fit_transform(emb_sel)

        plt.figure(figsize=(12, 10))
        for cls in top_classes:
            cls_mask = labels_sel == cls
            if not np.any(cls_mask):
                continue
            cls_name = class_names[cls] if cls < len(class_names) else str(cls)
            plt.scatter(
                emb_2d[cls_mask, 0], emb_2d[cls_mask, 1],
                label=cls_name, s=60, alpha=0.7
            )

        plt.title(f't-SNE визуализация: {name} (топ-{top_n} классов)')
        plt.xlabel('t-SNE компонента 1')
        plt.ylabel('t-SNE компонента 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Класс')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(output_dir, f'tsne_{safe_name}.png'))
        plt.close()

    logger.info(f"t-SNE визуализации сохранены в {output_dir}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Матрица ошибок",
    max_display_classes: int = 25
) -> np.ndarray:
    """
    Строит и сохраняет нормализованную матрицу ошибок (confusion matrix).

    Параметры:
        y_true : np.ndarray — истинные метки.
        y_pred : np.ndarray — предсказанные метки.
        class_names : list — имена классов.
        output_path : str — путь для сохранения.
        title : str — заголовок графика.
        max_display_classes : int — максимум классов для отображения.

    Возвращает:
        np.ndarray — матрица ошибок.
    """
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred)

    # Нормализация по строкам
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    # Ограничение количества классов для визуализации
    if len(cm) > max_display_classes:
        class_counts = np.sum(cm, axis=1)
        top_idx = np.argsort(class_counts)[-max_display_classes:]
        cm_norm = cm_norm[top_idx][:, top_idx]
        display_names = [class_names[i] if i < len(class_names) else str(i)
                         for i in top_idx]
        title += f" (топ-{max_display_classes} классов)"
    else:
        display_names = [class_names[i] if i < len(class_names) else str(i)
                         for i in range(len(cm))]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_norm, annot=(len(cm_norm) <= 20), fmt='.2f',
        cmap='Blues', xticklabels=display_names, yticklabels=display_names
    )
    plt.title(title)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Матрица ошибок сохранена: {output_path}")
    return cm


def plot_probability_distributions(
    probs_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_dir: str
) -> None:
    """
    Строит распределения предсказанных вероятностей для истинного
    и ошибочных классов для каждого метода.

    Позволяет оценить уверенность классификатора: хороший классификатор
    должен давать высокие вероятности для истинного класса и низкие
    для ошибочных.

    Параметры:
        probs_dict : dict — {название_метода: матрица_вероятностей}.
        y_true : np.ndarray — истинные метки.
        output_dir : str — директория для сохранения.
    """
    os.makedirs(output_dir, exist_ok=True)

    # === Распределение для истинного класса ===
    plt.figure(figsize=(10, 6))
    for name, probs in probs_dict.items():
        true_probs = probs[np.arange(len(y_true)), y_true]
        # Замена нулевых вероятностей для логарифмической шкалы
        true_probs = np.clip(true_probs, 1e-6, 1.0)
        try:
            sns.kdeplot(true_probs, label=name, fill=False, linewidth=2)
        except Exception:
            pass

    plt.xscale('log')
    plt.xlabel('Вероятность истинного класса (лог. шкала)')
    plt.ylabel('Плотность')
    plt.title('Распределение вероятностей для истинного класса')
    plt.legend(title='Метод')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'распределение_истинный_класс.png'))
    plt.close()

    # === Распределение для ошибочных классов ===
    plt.figure(figsize=(10, 6))
    for name, probs in probs_dict.items():
        mask = np.ones_like(probs, dtype=bool)
        mask[np.arange(len(y_true)), y_true] = False
        wrong_probs = probs[mask]
        wrong_probs = np.clip(wrong_probs, 1e-6, 1.0)
        try:
            sns.kdeplot(wrong_probs, label=name, fill=False, linewidth=2)
        except Exception:
            pass

    plt.xscale('log')
    plt.xlabel('Вероятность ошибочных классов (лог. шкала)')
    plt.ylabel('Плотность')
    plt.title('Распределение вероятностей для ошибочных классов')
    plt.legend(title='Метод')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'распределение_ошибочный_класс.png'))
    plt.close()

    logger.info(f"Распределения вероятностей сохранены в {output_dir}")


# =============================================================================
# МОДУЛЬ СРАВНИТЕЛЬНОГО АНАЛИЗА
# =============================================================================

def generate_comparison_table(
    metrics_list: List[ClassificationMetrics],
    output_path: str
) -> pd.DataFrame:
    """
    Формирует сводную сравнительную таблицу метрик качества
    для всех оцениваемых методов и сохраняет её в CSV.

    Параметры:
        metrics_list : list — список ClassificationMetrics для каждого метода.
        output_path : str — путь к выходному CSV-файлу.

    Возвращает:
        pd.DataFrame — таблица сравнения.
    """
    rows = []
    for m in metrics_list:
        row = OrderedDict()
        row['Метод'] = m.method_name
        row['Accuracy (Top-1)'] = f"{m.accuracy:.4f}"
        for k, v in sorted(m.top_k_accuracy.items()):
            row[f'Top-{k} Accuracy'] = f"{v:.4f}"
        row['Precision (macro)'] = f"{m.precision_macro:.4f}"
        row['Recall (macro)'] = f"{m.recall_macro:.4f}"
        row['F1-мера (macro)'] = f"{m.f1_macro:.4f}"
        row['Precision (weighted)'] = f"{m.precision_weighted:.4f}"
        row['Recall (weighted)'] = f"{m.recall_weighted:.4f}"
        row['F1-мера (weighted)'] = f"{m.f1_weighted:.4f}"
        row['ROC AUC (micro)'] = f"{m.roc_auc_micro:.4f}"
        row['ROC AUC (macro)'] = f"{m.roc_auc_macro:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    logger.info(f"Сводная таблица сохранена: {output_path}")

    # Вывод в консоль
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    return df


# =============================================================================
# ГЛАВНЫЙ МОДУЛЬ — КОНВЕЙЕР РЕИДЕНТИФИКАЦИИ
# =============================================================================

def run_reid_pipeline(config: ReIDConfig) -> Dict[str, Any]:
    """
    Выполняет полный конвейер байесовской реидентификации:
    1. Загрузка моделей-кодировщиков.
    2. Загрузка и подготовка данных.
    3. Извлечение эмбеддингов.
    4. Разбиение на support/query.
    5. Вычисление статистик классов.
    6. Байесовская классификация (MAP).
    7. Baseline: NN (только лицо, только тело, конкатенация).
    8. Вычисление метрик для всех методов.
    9. Визуализация и сохранение результатов.

    Параметры:
        config : ReIDConfig — конфигурация эксперимента.

    Возвращает:
        dict — словарь с метриками и путями к результатам.
    """
    start_time = time.time()

    # Установка зерна генератора
    set_global_seed(config.random_seed)

    # Создание директории результатов
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Вычислительное устройство: {device}")

    # =========================================================================
    # ШАГ 1: Загрузка моделей
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ШАГ 1: Загрузка нейросетевых моделей-кодировщиков")
    logger.info("=" * 60)

    face_model = load_embedding_model(
        config.face_model_path, config.face_emb_size,
        config.input_size, device
    )
    body_model = load_embedding_model(
        config.body_model_path, config.body_emb_size,
        config.input_size, device
    )

    # =========================================================================
    # ШАГ 2: Загрузка данных
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ШАГ 2: Загрузка и подготовка датасетов")
    logger.info("=" * 60)

    face_dataset = ReIDDataset(config.face_dir, config.input_size)
    body_dataset = ReIDDataset(config.body_dir, config.input_size)

    face_loader = DataLoader(
        face_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    body_loader = DataLoader(
        body_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # =========================================================================
    # ШАГ 3: Извлечение эмбеддингов
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ШАГ 3: Извлечение L2-нормализованных эмбеддингов")
    logger.info("=" * 60)

    face_emb = extract_embeddings(face_model, face_loader, "Эмбеддинги лиц")
    body_emb = extract_embeddings(body_model, body_loader, "Эмбеддинги силуэтов")

    # =========================================================================
    # ШАГ 4: Разбиение support/query
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ШАГ 4: Разбиение данных на support и query")
    logger.info("=" * 60)

    split = split_support_query(
        face_emb, body_emb,
        n_support=config.n_support,
        random_seed=config.random_seed
    )

    n_classes = len(split.valid_classes)
    logger.info(f"Количество валидных классов: {n_classes}")

    # =========================================================================
    # ШАГ 5: Вычисление статистик классов (на support set)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ШАГ 5: Вычисление статистик классов (μ, Σ)")
    logger.info("=" * 60)

    face_stats = compute_class_statistics(
        split.support_face_features, split.support_face_labels,
        cov_reg=config.cov_reg, confidence_level=config.confidence_level
    )
    body_stats = compute_class_statistics(
        split.support_body_features, split.support_body_labels,
        cov_reg=config.cov_reg, confidence_level=config.confidence_level
    )

    # =========================================================================
    # ШАГ 6: Классификация четырьмя методами
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ШАГ 6: Классификация (4 метода)")
    logger.info("=" * 60)

    y_true = split.query_face_labels

    # --- 6.1: Байесовский MAP-классификатор (предложенный метод) ---
    logger.info("6.1: Байесовский MAP-классификатор")
    bayes_preds, bayes_log_probs = bayesian_map_classify(
        split.query_face_features, split.query_body_features,
        face_stats, body_stats
    )
    bayes_probs = softmax(bayes_log_probs, axis=1)

    # --- 6.2: Линейное объединение (конкатенация + NN) ---
    logger.info("6.2: Линейное объединение (конкатенация + NN)")
    query_fused = np.concatenate(
        [split.query_face_features, split.query_body_features], axis=1
    )
    support_fused = np.concatenate(
        [split.support_face_features, split.support_body_features], axis=1
    )
    fusion_preds, fusion_sims = nearest_neighbor_classify(
        query_fused, support_fused, split.support_face_labels
    )
    # Преобразование сходств в вероятности по классам
    fusion_probs = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        for j, lbl in enumerate(split.support_face_labels):
            fusion_probs[i, lbl] = max(
                fusion_probs[i, lbl], fusion_sims[i, j]
            )
    fusion_probs = softmax(fusion_probs, axis=1)

    # --- 6.3: Только лицо (NN) ---
    logger.info("6.3: Только лицо (NN)")
    face_preds, face_sims = nearest_neighbor_classify(
        split.query_face_features, split.support_face_features,
        split.support_face_labels
    )
    face_probs = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        for j, lbl in enumerate(split.support_face_labels):
            face_probs[i, lbl] = max(face_probs[i, lbl], face_sims[i, j])
    face_probs = softmax(face_probs, axis=1)

    # --- 6.4: Только силуэт (NN) ---
    logger.info("6.4: Только силуэт (NN)")
    body_preds, body_sims = nearest_neighbor_classify(
        split.query_body_features, split.support_body_features,
        split.support_body_labels
    )
    body_probs = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        for j, lbl in enumerate(split.support_body_labels):
            body_probs[i, lbl] = max(body_probs[i, lbl], body_sims[i, j])
    body_probs = softmax(body_probs, axis=1)

    # =========================================================================
    # ШАГ 7: Вычисление метрик
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ШАГ 7: Вычисление метрик качества")
    logger.info("=" * 60)

    metrics_bayes = compute_classification_metrics(
        y_true, bayes_preds, bayes_log_probs,
        "Байес (MAP)", config.top_k_values
    )
    metrics_fusion = compute_classification_metrics(
        y_true, fusion_preds, np.log(fusion_probs + 1e-10),
        "Линейное объединение", config.top_k_values
    )
    metrics_face = compute_classification_metrics(
        y_true, face_preds, np.log(face_probs + 1e-10),
        "Только лицо", config.top_k_values
    )
    metrics_body = compute_classification_metrics(
        y_true, body_preds, np.log(body_probs + 1e-10),
        "Только силуэт", config.top_k_values
    )

    # =========================================================================
    # ШАГ 8: Визуализация
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ШАГ 8: Визуализация результатов")
    logger.info("=" * 60)

    probs_dict = {
        'Байес (MAP)': bayes_probs,
        'Линейное объединение': fusion_probs,
        'Только лицо': face_probs,
        'Только силуэт': body_probs,
    }

    # ROC-кривые
    plot_roc_curves(
        probs_dict, y_true,
        os.path.join(config.output_dir, 'roc_кривые.png'),
        n_classes=n_classes
    )

    # PR-кривые
    plot_precision_recall_curves(
        probs_dict, y_true,
        os.path.join(config.output_dir, 'pr_кривые.png'),
        n_classes=n_classes
    )

    # Распределения вероятностей
    plot_probability_distributions(probs_dict, y_true, config.output_dir)

    # t-SNE визуализация
    embeddings_dict = {
        'Лицо': split.query_face_features,
        'Силуэт': split.query_body_features,
        'Конкатенация': query_fused,
    }
    plot_tsne_embeddings(
        embeddings_dict, y_true, split.valid_classes,
        config.output_dir, top_n=config.top_n_classes_tsne,
        perplexity=config.tsne_perplexity, n_iter=config.tsne_n_iter,
        random_seed=config.random_seed
    )

    # Матрица ошибок для байесовского метода
    cm = plot_confusion_matrix(
        y_true, bayes_preds, split.valid_classes,
        os.path.join(config.output_dir, 'матрица_ошибок_байес.png'),
        title='Матрица ошибок — Байесовский MAP-классификатор'
    )

    # =========================================================================
    # ШАГ 9: Сохранение результатов
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ШАГ 9: Сохранение результатов")
    logger.info("=" * 60)

    all_metrics = [metrics_bayes, metrics_fusion, metrics_face, metrics_body]

    # Сводная таблица
    comparison_df = generate_comparison_table(
        all_metrics,
        os.path.join(config.output_dir, 'сравнение_методов.csv')
    )

    # Отчёт о классификации (байесовский метод)
    report = classification_report(
        y_true, bayes_preds,
        labels=range(n_classes),
        target_names=[str(c) for c in split.valid_classes],
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        os.path.join(config.output_dir, 'отчёт_классификации_байес.csv'),
        encoding='utf-8-sig'
    )

    # Сохранение матриц ошибок
    np.save(os.path.join(config.output_dir, 'confusion_matrix_bayes.npy'), cm)

    # Сохранение эмбеддингов для дальнейшего анализа
    np.savez(
        os.path.join(config.output_dir, 'embeddings.npz'),
        query_face=split.query_face_features,
        query_body=split.query_body_features,
        query_labels=y_true,
        support_face=split.support_face_features,
        support_body=split.support_body_features,
        support_labels=split.support_face_labels,
    )

    # Сохранение имён классов
    pd.DataFrame({
        'class_index': range(n_classes),
        'class_name': split.valid_classes
    }).to_csv(
        os.path.join(config.output_dir, 'классы.csv'),
        index=False, encoding='utf-8-sig'
    )

    # Сохранение конфигурации эксперимента
    config_dict = {
        'face_emb_size': config.face_emb_size,
        'body_emb_size': config.body_emb_size,
        'input_size': config.input_size,
        'n_support': config.n_support,
        'cov_reg': config.cov_reg,
        'random_seed': config.random_seed,
        'n_classes': n_classes,
        'n_query_samples': len(y_true),
        'n_support_face': len(split.support_face_labels),
        'n_support_body': len(split.support_body_labels),
    }
    pd.DataFrame([config_dict]).to_csv(
        os.path.join(config.output_dir, 'конфигурация.csv'),
        index=False, encoding='utf-8-sig'
    )

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    logger.info(f"Время выполнения: {elapsed:.1f} секунд")
    logger.info(f"Результаты сохранены: {config.output_dir}")
    logger.info("=" * 60)

    return {
        'metrics': {m.method_name: m for m in all_metrics},
        'comparison_table': comparison_df,
        'output_dir': config.output_dir,
        'elapsed_time': elapsed,
    }


# =============================================================================
# ИНТЕРФЕЙС КОМАНДНОЙ СТРОКИ
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Разбор аргументов командной строки.

    Возвращает:
        argparse.Namespace — разобранные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Программа для реидентификации личности на основе "
            "байесовского объединения эмбеддингов лица и тела"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Полный запуск с указанием всех путей
  python bayesian_reid_system.py \\
      --face_model /path/to/face_model.pt \\
      --body_model /path/to/body_model.pt \\
      --face_dir /path/to/Face/ \\
      --body_dir /path/to/reID/ \\
      --output_dir ./results

  # С нестандартными параметрами
  python bayesian_reid_system.py \\
      --face_model model_face.pt \\
      --body_model model_body.pt \\
      --face_dir ./Face/ \\
      --body_dir ./reID/ \\
      --n_support 3 \\
      --cov_reg 1e-4 \\
      --batch_size 64

Автор: Русаков К.Д., ИПУ РАН
        """
    )

    # Обязательные аргументы — пути
    paths = parser.add_argument_group('Пути к данным и моделям')
    paths.add_argument(
        '--face_model', type=str, required=True,
        help='Путь к предобученной модели кодировщика лица (.pt)'
    )
    paths.add_argument(
        '--body_model', type=str, required=True,
        help='Путь к предобученной модели кодировщика силуэта (.pt)'
    )
    paths.add_argument(
        '--face_dir', type=str, required=True,
        help='Путь к директории с изображениями лиц'
    )
    paths.add_argument(
        '--body_dir', type=str, required=True,
        help='Путь к директории с изображениями силуэтов (reID)'
    )
    paths.add_argument(
        '--output_dir', type=str, default='reid_results',
        help='Путь к директории для сохранения результатов (по умолчанию: reid_results)'
    )

    # Параметры модели
    model_params = parser.add_argument_group('Параметры модели')
    model_params.add_argument(
        '--face_emb_size', type=int, default=128,
        help='Размерность эмбеддинга лица (по умолчанию: 128)'
    )
    model_params.add_argument(
        '--body_emb_size', type=int, default=256,
        help='Размерность эмбеддинга силуэта (по умолчанию: 256)'
    )
    model_params.add_argument(
        '--input_size', type=int, default=112,
        help='Размер входного изображения (по умолчанию: 112)'
    )
    model_params.add_argument(
        '--batch_size', type=int, default=32,
        help='Размер мини-батча (по умолчанию: 32)'
    )
    model_params.add_argument(
        '--num_workers', type=int, default=4,
        help='Количество рабочих потоков загрузчика (по умолчанию: 4)'
    )

    # Параметры классификатора
    classifier_params = parser.add_argument_group('Параметры байесовского классификатора')
    classifier_params.add_argument(
        '--n_support', type=int, default=2,
        help='Количество support-образцов на класс (по умолчанию: 2)'
    )
    classifier_params.add_argument(
        '--cov_reg', type=float, default=1e-5,
        help='Регуляризация ковариационной матрицы λ (по умолчанию: 1e-5)'
    )

    # Прочие параметры
    misc = parser.add_argument_group('Прочие параметры')
    misc.add_argument(
        '--seed', type=int, default=42,
        help='Зерно генератора случайных чисел (по умолчанию: 42)'
    )
    misc.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Уровень логирования (по умолчанию: INFO)'
    )

    return parser.parse_args()


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

def main():
    """
    Главная функция программы.

    Выполняет разбор аргументов командной строки, формирует
    конфигурацию и запускает конвейер реидентификации.
    """
    args = parse_arguments()

    # Настройка логирования
    global logger
    log_file = os.path.join(args.output_dir, 'experiment.log')
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.log_level, log_file)

    # Формирование конфигурации из аргументов
    config = ReIDConfig(
        face_model_path=args.face_model,
        body_model_path=args.body_model,
        face_dir=args.face_dir,
        body_dir=args.body_dir,
        output_dir=args.output_dir,
        face_emb_size=args.face_emb_size,
        body_emb_size=args.body_emb_size,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_support=args.n_support,
        cov_reg=args.cov_reg,
        random_seed=args.seed,
    )

    # Вывод информации о конфигурации
    logger.info("=" * 60)
    logger.info("ПРОГРАММА РЕИДЕНТИФИКАЦИИ ЛИЧНОСТИ")
    logger.info("Байесовское объединение эмбеддингов лица и тела")
    logger.info("Автор: Русаков К.Д., ИПУ РАН")
    logger.info("=" * 60)
    logger.info(f"Модель лица:    {config.face_model_path}")
    logger.info(f"Модель силуэта: {config.body_model_path}")
    logger.info(f"Данные лиц:     {config.face_dir}")
    logger.info(f"Данные силуэтов: {config.body_dir}")
    logger.info(f"Результаты:     {config.output_dir}")
    logger.info(f"Эмбеддинг лица: {config.face_emb_size}-d")
    logger.info(f"Эмбеддинг тела: {config.body_emb_size}-d")
    logger.info(f"Support:        {config.n_support} образцов/класс")
    logger.info(f"Регуляризация:  {config.cov_reg}")
    logger.info(f"Random seed:    {config.random_seed}")
    logger.info(f"CUDA доступна:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU:            {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    # Запуск конвейера
    results = run_reid_pipeline(config)

    return results


if __name__ == "__main__":
    main()
