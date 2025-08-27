Что нужно установить

py -m pip install --user pipx

pipx ensurepath ИЛИ py -m pipx ensurepath

pipx install poetry

poetry install 


ЗАПУСК ПРОГРАММЫ:  poetry run digitize





Автоформатер запуск: poetry run black
Линтер запуск: poetry run ruff check .
Статический анализ типов: poetry run mypy src









ЧТОБЫ ЗАВИСИМОСТИ БЬЛИ В ПАПКЕ ПРОЕКТА
py -3.12 -m venv .venv
poetry env use .\.venv\Scripts\python.exe
poetry install
poetry env info --path