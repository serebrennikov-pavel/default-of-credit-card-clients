"""
Flask веб-сервис для прогнозирования дефолта по кредитным картам.

Эндпоинты:
  POST /predict  — принимает JSON с признаками клиента, возвращает прогноз
  GET  /health   — проверка работоспособности сервиса

Поддерживает две версии модели (v1 / v2) для A/B-тестирования.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from loguru import logger

from default_of_credit_card_clients.features import build_features

# ── Настройка логирования: вывод в JSON-формат ──────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    serialize=False,   # serialize=True → чистый JSON в stdout
    level="INFO",
)

# ── Инициализация приложения ────────────────────────────────────────────────
app = Flask(__name__)

# Директория с моделями берётся из переменной окружения (удобно для Docker)
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))

# Версия модели по умолчанию (переопределяется через MODEL_VERSION в .env / docker)
DEFAULT_VERSION = os.getenv("MODEL_VERSION", "v1")

# Реестр загруженных моделей: {"v1": pipeline, "v2": pipeline}
MODELS: dict = {}

# Упорядоченный список признаков — загружается вместе с моделями
FEATURE_COLUMNS: list[str] = []


def _load_all_models() -> None:
    """Загружает все доступные версии моделей при старте сервиса."""
    import joblib

    # Загружаем список признаков, сохранённый при обучении
    feat_path = MODEL_DIR / "feature_columns.csv"
    if feat_path.exists():
        import pandas as pd
        FEATURE_COLUMNS.extend(pd.read_csv(feat_path, header=None)[0].tolist())
        logger.info(f"Загружено {len(FEATURE_COLUMNS)} признаков из {feat_path}")
    else:
        logger.warning(f"Файл признаков не найден: {feat_path}. Используется список по умолчанию.")
        # Резервный список признаков (порядок важен!)
        FEATURE_COLUMNS.extend([
            "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
            "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
            "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
            "avg_utilisation", "total_paid", "max_delay", "months_delayed",
        ])

    # Ищем файлы model_v1.joblib, model_v2.joblib, ...
    for model_file in sorted(MODEL_DIR.glob("model_v*.joblib")):
        version = model_file.stem.replace("model_", "")   # "v1", "v2", ...
        MODELS[version] = joblib.load(model_file)
        logger.info(f"Модель {version} загружена из {model_file}")

    if not MODELS:
        logger.error(
            f"Не найдено ни одной модели в {MODEL_DIR}. "
            "Запустите: python -m default_of_credit_card_clients.modeling.train"
        )


# Загружаем модели сразу при импорте модуля
_load_all_models()


# ── Эндпоинт /health ────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """
    Проверка работоспособности сервиса.

    Ответ:
        200 OK  — сервис работает, модели загружены
        503     — модели не загружены
    """
    if not MODELS:
        return jsonify({"status": "unavailable", "reason": "модели не загружены"}), 503

    return jsonify({
        "status": "healthy",
        "loaded_models": sorted(MODELS.keys()),
        "default_version": DEFAULT_VERSION,
        "timestamp": time.time(),
    }), 200


# ── Эндпоинт /predict ───────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Прогноз дефолта по кредитной карте.

    Тело запроса (JSON):
        Обязательные поля: все признаки из FEATURE_COLUMNS
        Опциональные поля:
            version  — версия модели ("v1", "v2", "ab").
                       "ab" — случайный выбор 50/50 для A/B-теста.
                       По умолчанию берётся DEFAULT_VERSION.

    Пример запроса:
        {
            "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 10000, "BILL_AMT2": 9000, "BILL_AMT3": 8000,
            "BILL_AMT4": 7000,  "BILL_AMT5": 6000, "BILL_AMT6": 5000,
            "PAY_AMT1": 1000, "PAY_AMT2": 1000, "PAY_AMT3": 1000,
            "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000,
            "version": "v1"
        }

    Ответ (JSON):
        {
            "prediction":     1,           // 1 — дефолт, 0 — нет дефолта
            "probability":    0.7823,      // вероятность дефолта
            "label":          "default",   // текстовая метка
            "model_version":  "v1",
            "duration_ms":    12.5
        }
    """
    t_start = time.time()

    # ── Парсим тело запроса ──────────────────────────────────────────────────
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Тело запроса должно быть в формате JSON"}), 400

    # ── Определяем версию модели ─────────────────────────────────────────────
    version = data.pop("version", DEFAULT_VERSION)
    if version == "ab":
        # A/B-тест: случайное равномерное разделение трафика 50/50
        version = random.choice(sorted(MODELS.keys()))

    if version not in MODELS:
        return jsonify({
            "error": f"Версия модели '{version}' недоступна. "
                     f"Доступные: {sorted(MODELS.keys())}"
        }), 400

    # ── Собираем вектор признаков ─────────────────────────────────────────────
    # Сырые признаки (без инженерных) — те, что не вычисляются автоматически
    engineered = {"avg_utilisation", "total_paid", "max_delay", "months_delayed"}
    raw_cols = [col for col in FEATURE_COLUMNS if col not in engineered]

    missing = [col for col in raw_cols if col not in data]
    if missing:
        return jsonify({"error": f"Отсутствуют обязательные признаки: {missing}"}), 422

    try:
        row = pd.DataFrame([{col: float(data[col]) for col in raw_cols}])
        row = build_features(row)
        X = np.array([[float(row[col].iloc[0]) for col in FEATURE_COLUMNS]])
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Ошибка преобразования признаков: {exc}"}), 422

    # ── Инференс ──────────────────────────────────────────────────────────────
    pipeline = MODELS[version]
    prediction = int(pipeline.predict(X)[0])
    probability = float(pipeline.predict_proba(X)[0][1])
    duration_ms = round((time.time() - t_start) * 1000, 2)

    result = {
        "prediction":    prediction,
        "probability":   round(probability, 4),
        "label":         "default" if prediction == 1 else "no_default",
        "model_version": version,
        "duration_ms":   duration_ms,
    }

    # ── Структурированное логирование запроса ─────────────────────────────────
    logger.info(json.dumps({
        "event":       "prediction",
        "version":     version,
        "prediction":  prediction,
        "probability": round(probability, 4),
        "duration_ms": duration_ms,
    }, ensure_ascii=False))

    return jsonify(result), 200


# ── Запуск напрямую (для разработки) ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
