"""
Скачивает датасет с Kaggle и сохраняет в data/raw/.

Требования:
    - Установленный пакет kaggle: pip install kaggle
    - Файл с учётными данными: ~/.kaggle/kaggle.json
      (создаётся в личном кабинете Kaggle: Account → API → Create New Token)

Использование:
    python scripts/download_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from default_of_credit_card_clients.config import RAW_DATA_DIR

# Идентификатор датасета на Kaggle в формате <владелец>/<название>
KAGGLE_DATASET = "uciml/default-of-credit-card-clients-dataset"


def download_dataset() -> None:
    # Проверяем, есть ли уже файл датасета в папке (XLS, XLSX или CSV)
    existing_files = (
        list(RAW_DATA_DIR.glob("*.xls"))
        + list(RAW_DATA_DIR.glob("*.xlsx"))
        + list(RAW_DATA_DIR.glob("*.csv"))
    )
    if existing_files:
        logger.info(f"Датасет уже существует: {existing_files[0]}, пропускаем загрузку.")
        return

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Скачиваем датасет с Kaggle: {KAGGLE_DATASET} ...")

    try:
        import kaggle
        kaggle.api.authenticate()
    except OSError:
        logger.error(
            "Файл учётных данных ~/.kaggle/kaggle.json не найден. "
            "Создайте его в личном кабинете Kaggle: Account → API → Create New Token."
        )
        raise

    # Скачиваем и сразу распаковываем архив в папку data/raw/
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(RAW_DATA_DIR),
        unzip=True,
    )

    logger.success(f"Датасет готов: {RAW_DATA_DIR}")


if __name__ == "__main__":
    download_dataset()