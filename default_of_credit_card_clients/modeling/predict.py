"""
Вспомогательные функции для загрузки модели и выполнения предсказаний.
"""
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline
import typer

from default_of_credit_card_clients.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_model(model_path: Union[str, Path]) -> Pipeline:
    """Загружает обученный Pipeline из файла .joblib."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {path}")
    pipeline: Pipeline = joblib.load(path)
    logger.info(f"Модель загружена из {path}")
    return pipeline


def load_feature_columns(models_dir: Path = MODELS_DIR) -> list[str]:
    """Загружает упорядоченный список признаков, использованных при обучении."""
    path = models_dir / "feature_columns.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Файл feature_columns.csv не найден в {models_dir}. "
            "Запустите пайплайн обучения."
        )
    return pd.read_csv(path, header=None)[0].tolist()


def predict_single(
    pipeline: Pipeline,
    features: dict,
    feature_columns: list[str],
) -> dict:
    """
    Предсказание для одного клиента.

    Аргументы:
        pipeline:        обученный Pipeline (StandardScaler + классификатор).
        features:        словарь {имя_признака: значение}.
        feature_columns: упорядоченный список признаков (тот же порядок, что при обучении).

    Возвращает:
        словарь с ключами prediction (int) и probability (float).
    """
    # Собираем вектор признаков в правильном порядке
    X = np.array([[features[col] for col in feature_columns]])
    prediction = int(pipeline.predict(X)[0])
    probability = float(pipeline.predict_proba(X)[0][1])
    return {"prediction": prediction, "probability": probability}


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model_v1.joblib",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
) -> None:
    pipeline = load_model(model_path)
    feature_cols = load_feature_columns()

    df = pd.read_csv(features_path)
    X = df[feature_cols].values

    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    results = pd.DataFrame({"prediction": y_pred, "probability": y_prob})
    results.to_csv(predictions_path, index=False)
    logger.success(f"Сохранено {len(results)} предсказаний: {predictions_path}")


if __name__ == "__main__":
    app()