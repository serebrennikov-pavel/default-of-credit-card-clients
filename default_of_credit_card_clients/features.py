"""
Генерация новых признаков на основе исходного датасета.

Читает:  data/processed/dataset.csv
Записывает:
  - data/processed/X_train.csv  (обучающая выборка, признаки)
  - data/processed/X_test.csv   (тестовая выборка, признаки)
  - data/processed/y_train.csv  (обучающая выборка, целевая переменная)
  - data/processed/y_test.csv   (тестовая выборка, целевая переменная)

Запуск:
    python -m default_of_credit_card_clients.features
"""
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import typer

from default_of_credit_card_clients.config import PROCESSED_DATA_DIR, MODELS_DIR
from default_of_credit_card_clients.dataset import FEATURE_COLS, TARGET_COL

app = typer.Typer()

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет новые признаки, вычисленные из исходных столбцов."""
    df = df.copy()

    # Средняя загруженность кредитного лимита за 6 месяцев
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    df["avg_utilisation"] = df[bill_cols].mean(axis=1) / df["LIMIT_BAL"].clip(lower=1)

    # Суммарная сумма платежей за последние 6 месяцев
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    df["total_paid"] = df[pay_amt_cols].sum(axis=1)

    # Максимальный статус просрочки (чем больше, тем хуже)
    pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["max_delay"] = df[pay_status_cols].max(axis=1)

    # Количество месяцев с просроченным платежом (PAY_x > 0)
    df["months_delayed"] = (df[pay_status_cols] > 0).sum(axis=1)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Возвращает список столбцов-признаков (все столбцы кроме целевой переменной)."""
    return [c for c in df.columns if c != TARGET_COL]


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    scaler_path: Path = MODELS_DIR / "scaler.joblib",
) -> None:
    logger.info(f"Загрузка обработанных данных из {input_path} ...")
    df = pd.read_csv(input_path)

    logger.info("Генерация признаков ...")
    df = build_features(df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Обучаем масштабировщик только на обучающей выборке, чтобы избежать утечки данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Сохраняем масштабировщик на диск
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Масштабировщик сохранён: {scaler_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in [
        ("X_train", X_train_scaled),
        ("X_test", X_test_scaled),
        ("y_train", y_train),
        ("y_test", y_test),
    ]:
        path = output_dir / f"{name}.csv"
        pd.DataFrame(arr).to_csv(path, index=False)

    # Сохраняем список признаков — он понадобится API при загрузке модели
    pd.Series(feature_cols).to_csv(output_dir / "feature_columns.csv", index=False)

    logger.success(
        f"Признаки сохранены. Обучение: {X_train_scaled.shape}, тест: {X_test_scaled.shape}"
    )


if __name__ == "__main__":
    app()