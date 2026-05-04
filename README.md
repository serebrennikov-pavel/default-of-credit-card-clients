    # Credit Card Default Prediction Service

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)

Сервис машинного обучения для прогнозирования дефолта по кредитным картам.
Реализует полный production-like цикл: обучение → сохранение → REST API → контейнеризация → A/B-тест.

**Датасет:** [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (UCI ML Repository)  
**Задача:** бинарная классификация — предсказать дефолт клиента в следующем месяце.

---

## Структура репозитория

```
.
├── app/
│   ├── __init__.py
│   └── api.py                  # Flask-приложение (эндпоинты /predict, /health)
├── default_of_credit_card_clients/
│   ├── config.py               # Константы путей
│   ├── dataset.py              # Загрузка и очистка датасета
│   ├── features.py             # Инжиниринг признаков
│   └── modeling/
│       ├── train.py            # Обучение моделей v1 и v2
│       └── predict.py          # Инференс-хелперы
├── scripts/
│   └── download_data.py        # Скачивание датасета с UCI
├── models/                     # Сохранённые модели (.joblib) — создаётся при обучении
├── data/
│   ├── raw/                    # Исходный XLS-файл датасета
│   └── processed/              # Очищенные данные (CSV)
├── tests/
│   ├── test_data.py
│   └── test_api.py             # Тесты Flask API
├── Dockerfile
├── docker-compose.yml          # Оркестрация v1 + v2 + nginx (A/B-тест)
├── nginx.conf                  # Балансировщик нагрузки 50/50
├── ARCHITECTURE.md             # Архитектурные решения, MLOps, бизнес-метрики
├── AB_TEST_PLAN.md             # План A/B-тестирования
├── requirements.txt
└── pyproject.toml
```

---

## Быстрый старт (локальный запуск)

### 1. Установка зависимостей

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Скачать датасет

```bash
python scripts/download_data.py
```

Файл сохраняется в `data/raw/`. Если скачать не получается — поместите XLS-файл
датасета вручную в `data/raw/`.

### 3. Подготовить данные

```bash
python -m default_of_credit_card_clients.dataset
```

Результат: `data/processed/dataset.csv`

### 4. Обучить модели

```bash
python -m default_of_credit_card_clients.modeling.train
```

Результат: `models/model_v1.joblib`, `models/model_v2.joblib`, `models/feature_columns.csv`

### 5. Запустить сервис

```bash
python app/api.py
```

Сервис доступен на `http://localhost:5000`.

---

## Запуск тестов

```bash
pytest tests/ -v
```

---

## API: формат запросов и ответов

### GET /health

Проверка работоспособности сервиса.

```bash
curl http://localhost:5000/health
```

Ответ:
```json
{
  "status": "healthy",
  "loaded_models": ["v1", "v2"],
  "default_version": "v1",
  "timestamp": 1735700000.0
}
```

---

### POST /predict

Прогноз дефолта для одного клиента.

**Обязательные поля признаков:**

| Признак | Описание | Тип |
|---|---|---|
| LIMIT_BAL | Кредитный лимит (NT$) | float |
| SEX | Пол (1=мужчина, 2=женщина) | int |
| EDUCATION | Образование (1=аспирантура, 2=университет, 3=школа, 4=прочее) | int |
| MARRIAGE | Семейное положение (1=женат/замужем, 2=холост/незамужем, 3=прочее) | int |
| AGE | Возраст (лет) | int |
| PAY_0 | Статус оплаты за сентябрь (-1=вовремя, 1=задержка 1 мес, ...) | int |
| PAY_2 … PAY_6 | Статус оплаты август–апрель | int |
| BILL_AMT1 … BILL_AMT6 | Сумма счёта сентябрь–апрель (NT$) | float |
| PAY_AMT1 … PAY_AMT6 | Сумма платежа сентябрь–апрель (NT$) | float |
| avg_utilisation | Средняя утилизация кредита | float |
| total_paid | Итого выплачено за 6 месяцев | float |
| max_delay | Максимальная задержка в месяцах | int |
| months_delayed | Количество месяцев с задержкой | int |

**Опциональные поля:**

| Поле | Описание | По умолчанию |
|---|---|---|
| version | Версия модели: `"v1"`, `"v2"`, `"ab"` (случайный A/B) | `"v1"` |

**Пример запроса:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000,
    "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 10000, "BILL_AMT2": 9000, "BILL_AMT3": 8000,
    "BILL_AMT4": 7000,  "BILL_AMT5": 6000, "BILL_AMT6": 5000,
    "PAY_AMT1": 1000, "PAY_AMT2": 1000, "PAY_AMT3": 1000,
    "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000,
    "avg_utilisation": 0.32, "total_paid": 6000,
    "max_delay": 0, "months_delayed": 0,
    "version": "v1"
  }'
```

**Ответ:**

```json
{
  "prediction":    0,
  "probability":   0.1243,
  "label":         "no_default",
  "model_version": "v1",
  "duration_ms":   14.2
}
```

| Поле | Описание |
|---|---|
| prediction | 1 = дефолт, 0 = нет дефолта |
| probability | Вероятность дефолта [0, 1] |
| label | `"default"` или `"no_default"` |
| model_version | Какая версия модели ответила |
| duration_ms | Время обработки запроса в мс |

---

## Docker

### Сборка и запуск одного образа

```bash
# Сборка
docker build -t credit-default-service .

# Запуск (модели должны быть обучены локально, монтируем папку)
# Linux
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_VERSION=v1 \
  credit-default-service
  
 # Windows
 docker run -p 5000:5000 ^
  -v %cd%/models:/app/models:ro ^
  -e MODEL_VERSION=v1 ^
  credit-default-service
```

### Docker Hub

Страница образа на Docker Hub: https://hub.docker.com/repository/docker/serebrennikovps/credit-default-service/
```
docker pull serebrennikovps/credit-default-service:latest
```

### Docker Compose (v1 + v2 + Nginx для A/B-теста)

```bash
# Предварительно обучить модели (шаги 2-4 из «Быстрого старта»)
docker compose up --build

# Проверка
curl http://localhost:5001/health   # v1
curl http://localhost:5002/health   # v2
curl http://localhost/health        # через nginx (A/B)
```

| Адрес | Описание |
|---|---|
| `http://localhost:5001` | Сервис v1 (LogisticRegression) |
| `http://localhost:5002` | Сервис v2 (GradientBoosting) |
| `http://localhost:80` | Nginx балансировщик (A/B-тест 50/50) |

---

## Модели

| Версия | Алгоритм | Описание |
|---|---|---|
| v1 | LogisticRegression | Базовая интерпретируемая модель, быстрый инференс |
| v2 | GradientBoostingClassifier | Более высокий F1, используется как challenger в A/B-тесте |

Оба артефакта — `sklearn.pipeline.Pipeline` (StandardScaler → Classifier),
сериализованные через `joblib`. Самодостаточны для инференса без дополнительной предобработки.

---

## Архитектура и A/B-тест

- Подробное обоснование архитектурных решений: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- План A/B-тестирования: [`AB_TEST_PLAN.md`](AB_TEST_PLAN.md)