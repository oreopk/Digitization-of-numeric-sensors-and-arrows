Что нужно установить

py -m pip install --user pipx

pipx ensurepath или py -m pipx ensurepath

pipx install poetry

poetry install 

Если не работает Tesseract-OCR через зависимости, важно установить exe файл самому и указать путь в одном из файлов
Пример: pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


ЗАПУСК ПРОГРАММЫ:  poetry run digitize или python -m app.cli





Автоформатер запуск: poetry run black
Линтер запуск: poetry run ruff check .
Статический анализ типов: poetry run mypy src









ЧТОБЫ ЗАВИСИМОСТИ БЬЛИ В ПАПКЕ ПРОЕКТА
py -3.12 -m venv .venv
poetry env use .\.venv\Scripts\python.exe
poetry install
poetry env info --path


В теории можно установить всё без сборщика руками, но не рекомендуется.

python -m pip install Pillow
pip install opencv-python
pip install opencv-python-headless
pip install easyocr
pip install pytesseract
pip install matplotlib
pip install pandas
pip install psutil

install opencv-contrib-python --upgrade
pip install pandas openpyxl

сборка в EXE файл

python -m PyInstaller 0214_norm.py

python -m PyInstaller --onefile 0214_norm.py
