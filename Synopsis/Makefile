# Makefile для компиляции автореферата

# Имя главного tex-файла (без расширения)
TARGET = autoreferat

# Компилятор (можно изменить на xelatex или lualatex)
LATEX = pdflatex

# Флаги компилятора
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error -file-line-error

# Основная цель - собрать PDF
all: $(TARGET).pdf

# Правило сборки PDF
$(TARGET).pdf: $(TARGET).tex *.tex
	$(LATEX) $(LATEX_FLAGS) $(TARGET).tex
	$(LATEX) $(LATEX_FLAGS) $(TARGET).tex

# Быстрая сборка (один проход)
quick:
	$(LATEX) $(LATEX_FLAGS) $(TARGET).tex

# Полная сборка с библиографией (если будет использоваться)
full: $(TARGET).tex *.tex
	$(LATEX) $(LATEX_FLAGS) $(TARGET).tex
	biber $(TARGET)
	$(LATEX) $(LATEX_FLAGS) $(TARGET).tex
	$(LATEX) $(LATEX_FLAGS) $(TARGET).tex

# Очистка временных файлов
clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.bcf *.run.xml
	rm -f *.fls *.fdb_latexmk *.synctex.gz

# Полная очистка (включая PDF)
distclean: clean
	rm -f $(TARGET).pdf

# Открыть PDF после сборки (Linux)
view: all
	xdg-open $(TARGET).pdf &

# Справка
help:
	@echo "Доступные команды:"
	@echo "  make          - Собрать автореферат (2 прохода)"
	@echo "  make quick    - Быстрая сборка (1 проход)"
	@echo "  make full     - Полная сборка с библиографией"
	@echo "  make clean    - Удалить временные файлы"
	@echo "  make distclean- Удалить все сгенерированные файлы"
	@echo "  make view     - Собрать и открыть PDF"
	@echo "  make help     - Показать эту справку"

.PHONY: all quick full clean distclean view help
