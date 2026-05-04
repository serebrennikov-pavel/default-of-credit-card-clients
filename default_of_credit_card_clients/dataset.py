"""
Загрузка, очистка и сохранение датасета Default of Credit Card Clients.

Входные данные:  data/raw/*.xls / *.xlsx / *.csv
Выходные данные: data/processed/dataset.csv — признаки + целевая переменная без столбца ID.

Запуск:
    python -m default_of_credit_card_clients.dataset
"""
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from default_of_credit_card_clients.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# Список признаков согласно документации датасета UCI #350
FEATURE_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET_COL = "default.payment.next.month"


def _find_raw_file(raw_dir: Path) -> Path:
    """Ищет файл датасета в указанной папке (XLS, XLSX или CSV)."""
    for pattern in ["*.xls", "*.xlsx", "*.csv"]:
        candidates = list(raw_dir.glob(pattern))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(
        f"Файл датасета не найден в {raw_dir}. "
        "Запустите: python scripts/download_data.py"
    )


def load_raw(raw_path: Path) -> pd.DataFrame:
    """Загружает сырые данные из XLS/XLSX или CSV файла."""
    suffix = raw_path.suffix.lower()
    if suffix in (".xls", ".xlsx"):
        # В оригинальном XLS-файле от UCI двойной заголовок:
        # строка 0 содержит 'X1', 'X2', ..., настоящие имена столбцов — в строке 1
        try:
            df = pd.read_excel(raw_path, header=1, engine="xlrd" if suffix == ".xls" else "openpyxl")
            df.columns = df.columns.str.strip()
        except Exception:
            df = pd.read_excel(raw_path, header=0, engine="xlrd" if suffix == ".xls" else "openpyxl")
            df.columns = df.columns.str.strip()
    else:
        # CSV-файл имеет обычный одинарный заголовок
        df = pd.read_csv(raw_path)
        df.columns = df.columns.str.strip()

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит датафрейм к стандартному виду: переименовывает столбцы, удаляет ID."""
    rename_map = {}
    for col in df.columns:
        # Нормализуем имя для поиска: верхний регистр, пробелы и точки → подчёркивание
        normalised = col.upper().replace(" ", "_").replace(".", "_")
        if "DEFAULT" in normalised and "PAYMENT" in normalised:
            rename_map[col] = TARGET_COL
        elif col.upper() == "PAY_1":
            # В некоторых версиях файла столбец называется PAY_1, а не PAY_0
            rename_map[col] = "PAY_0"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Удаляем столбец ID — он не нужен для обучения модели
    id_cols = [c for c in df.columns if c.upper() in ("ID", "ROW_ID")]
    df = df.drop(columns=id_cols, errors="ignore")

    # Оставляем только нужные столбцы
    keep = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Столбцы не найдены в датасете: {missing}\nДоступные: {list(df.columns)}")

    df = df[keep].copy()

    # Приводим все значения к числовому типу, строки с NaN удаляем
    df = df.apply(pd.to_numeric, errors="coerce")
    n_before = len(df)
    df = df.dropna()
    if len(df) < n_before:
        logger.warning(f"Удалено {n_before - len(df)} строк с пропущенными значениями.")

    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df.reset_index(drop=True)


@app.command()
def main(
    input_path: Path = typer.Option(None, help="Путь к исходному файлу (определяется автоматически)"),
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
) -> None:
    if input_path is None:
        input_path = _find_raw_file(RAW_DATA_DIR)

    logger.info(f"Загрузка данных из {input_path} ...")
    df_raw = load_raw(input_path)
    logger.info(f"Размер исходного датасета: {df_raw.shape}")

    logger.info("Очистка данных ...")
    steps = ["load", "clean", "validate", "save"]
    for step in tqdm(steps, desc="Обработка"):
        if step == "clean":
            df = clean(df_raw)
        elif step == "validate":
            assert TARGET_COL in df.columns
            assert df[TARGET_COL].isin([0, 1]).all(), "Целевая переменная должна быть бинарной (0/1)"
        elif step == "save":
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

    logger.success(f"Обработанный датасет сохранён ({len(df)} строк): {output_path}")
    logger.info(f"Распределение целевой переменной:\n{df[TARGET_COL].value_counts(normalize=True).round(3)}")


if __name__ == "__main__":
    app()