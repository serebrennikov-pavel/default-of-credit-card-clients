"""
Тесты для Flask API сервиса прогнозирования дефолта.

Запуск:
    pytest tests/test_api.py -v
"""
import json
import pytest

# Минимальный набор признаков для тестовых запросов (базовые + engineered)
SAMPLE_FEATURES = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 10000,
    "BILL_AMT2": 9000,
    "BILL_AMT3": 8000,
    "BILL_AMT4": 7000,
    "BILL_AMT5": 6000,
    "BILL_AMT6": 5000,
    "PAY_AMT1": 1000,
    "PAY_AMT2": 1000,
    "PAY_AMT3": 1000,
    "PAY_AMT4": 1000,
    "PAY_AMT5": 1000,
    "PAY_AMT6": 1000,
    "avg_utilisation": 0.32,
    "total_paid": 6000,
    "max_delay": 0,
    "months_delayed": 0,
}


@pytest.fixture
def client():
    """Создаём тестовый клиент Flask без поднятия настоящего сервера."""
    from app.api import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── Тест /health ─────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    """Эндпоинт /health должен возвращать 200 и поле status."""
    response = client.get("/health")
    assert response.status_code == 200
    body = json.loads(response.data)
    assert body["status"] == "healthy"
    assert "loaded_models" in body


# ── Тесты /predict ────────────────────────────────────────────────────────────

def test_predict_v1_returns_prediction(client):
    """Запрос с version=v1 возвращает корректный ответ."""
    payload = {**SAMPLE_FEATURES, "version": "v1"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    body = json.loads(response.data)
    assert body["model_version"] == "v1"
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("default", "no_default")


def test_predict_v2_returns_prediction(client):
    """Запрос с version=v2 возвращает корректный ответ."""
    payload = {**SAMPLE_FEATURES, "version": "v2"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    body = json.loads(response.data)
    assert body["model_version"] == "v2"
    assert body["prediction"] in (0, 1)


def test_predict_ab_mode(client):
    """Режим 'ab' должен случайно выбирать версию v1 или v2."""
    payload = {**SAMPLE_FEATURES, "version": "ab"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    body = json.loads(response.data)
    assert body["model_version"] in ("v1", "v2")


def test_predict_missing_feature_returns_422(client):
    """Отсутствие обязательного признака должно возвращать 422."""
    # Убираем один признак из запроса
    payload = {k: v for k, v in SAMPLE_FEATURES.items() if k != "LIMIT_BAL"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 422
    body = json.loads(response.data)
    assert "error" in body


def test_predict_invalid_json_returns_400(client):
    """Запрос без JSON-тела должен возвращать 400."""
    response = client.post("/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_predict_unknown_version_returns_400(client):
    """Неизвестная версия модели должна возвращать 400."""
    payload = {**SAMPLE_FEATURES, "version": "v99"}
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    body = json.loads(response.data)
    assert "error" in body