"""
Обучение двух версий модели и сохранение в папку models/.

v1 — LogisticRegression          (простая линейная модель, легко интерпретировать)
v2 — GradientBoostingClassifier  (более сложная, как правило точнее)

Каждая модель сохраняется как Pipeline: StandardScaler + классификатор,
поэтому при инференсе не нужно отдельно нормализовывать входные данные.

Запуск:
    python -m default_of_credit_card_clients.modeling.train
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from default_of_credit_card_clients.config import MODELS_DIR, PROCESSED_DATA_DIR
from default_of_credit_card_clients.dataset import FEATURE_COLS, TARGET_COL
from default_of_credit_card_clients.features import build_features

app = typer.Typer()

RANDOM_STATE = 42
TEST_SIZE = 0.20


def _load_data(data_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Загружает обработанный датасет и создаёт новые признаки."""
    logger.info(f"Загрузка данных из {data_path}")
    df = pd.read_csv(data_path)
    df = build_features(df)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    return X, y, feature_cols


def _evaluate(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    """Вычисляет метрики качества модели на тестовой выборке."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    logger.info(f"\nМетрики модели {name}:\n{classification_report(y_test, y_pred)}")
    return metrics


def build_v1() -> Pipeline:
    """Модель v1: логистическая регрессия — простая и понятная базовая модель."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",  # учитываем дисбаланс классов
            C=0.1,
        )),
    ])


def build_v2() -> Pipeline:
    """Модель v2: градиентный бустинг — сложнее, но даёт более высокий F1."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )),
    ])


@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    models_dir: Path = MODELS_DIR,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_cols = _load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Обучение: {X_train.shape}, тест: {X_test.shape}")
    logger.info(f"Доля дефолтов — обучение: {y_train.mean():.3f}, тест: {y_test.mean():.3f}")

    # Сохраняем список признаков — API читает его при запуске для валидации входных данных
    pd.Series(feature_cols).to_csv(models_dir / "feature_columns.csv", index=False, header=False)

    all_metrics = []
    for version, pipeline in [("v1", build_v1()), ("v2", build_v2())]:
        logger.info(f"Обучение модели {version} ...")
        pipeline.fit(X_train, y_train)

        metrics = _evaluate(pipeline, X_test, y_test, version)
        all_metrics.append(metrics)

        save_path = models_dir / f"model_{version}.joblib"
        joblib.dump(pipeline, save_path)
        logger.success(f"Модель {version} сохранена: {save_path}")

    # Итоговая таблица сравнения моделей
    summary = pd.DataFrame(all_metrics).set_index("model")
    logger.info(f"\nСравнение моделей:\n{summary.to_string()}")


if __name__ == "__main__":
    app()