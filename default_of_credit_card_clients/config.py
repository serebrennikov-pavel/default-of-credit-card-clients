"""
Центральный модуль конфигурации: все пути и переменные окружения проекта.
Импортируйте константы отсюда вместо хардкода путей в коде.
"""
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Загружаем переменные окружения из .env (если файл существует)
load_dotenv()

# Корень проекта — два уровня вверх от этого файла
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"Корень проекта: {PROJ_ROOT}")

# ── Директории данных ─────────────────────────────────────────────────────────
DATA_DIR          = PROJ_ROOT / "data"
RAW_DATA_DIR      = DATA_DIR / "raw"        # исходные данные (неизменяемые)
INTERIM_DATA_DIR  = DATA_DIR / "interim"    # промежуточные преобразования
PROCESSED_DATA_DIR = DATA_DIR / "processed" # финальные данные для моделирования
EXTERNAL_DATA_DIR = DATA_DIR / "external"   # данные из внешних источников

# ── Директории артефактов ─────────────────────────────────────────────────────
MODELS_DIR  = PROJ_ROOT / "models"          # обученные модели (.joblib, .pkl)
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"       # графики и визуализации

# ── Совместимость loguru с tqdm (прогресс-бары) ───────────────────────────────
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass