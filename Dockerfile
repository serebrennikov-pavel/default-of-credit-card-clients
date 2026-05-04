# ────────────────────────────────────────────────────────────────────────────
# Многоэтапная сборка Docker-образа для ML-сервиса прогнозирования дефолта
# ────────────────────────────────────────────────────────────────────────────

# ── Этап 1: установка зависимостей ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Копируем только файлы зависимостей — кэш слоя не инвалидируется при изменении кода
COPY requirements.txt pyproject.toml README.md ./

RUN pip install --upgrade pip && \
    grep -v "^-e" requirements.txt > /tmp/reqs_no_editable.txt && \
    pip install --no-cache-dir -r /tmp/reqs_no_editable.txt

# ── Этап 2: финальный образ ──────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="SerebrennikovPS" \
      description="Credit card default prediction service" \
      version="1.0"

WORKDIR /app

# Копируем установленные пакеты из builder-этапа
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем исходный код проекта
COPY default_of_credit_card_clients/ ./default_of_credit_card_clients/
COPY app/                            ./app/
COPY pyproject.toml README.md ./

# Устанавливаем пакет в режиме editable (нужен для импорта модуля)
RUN pip install --no-cache-dir -e .

# Папка для моделей; при запуске контейнера монтируется как volume
RUN mkdir -p /app/models

# ── Переменные окружения ──────────────────────────────────────────────────────
# MODEL_DIR    — путь к директории с обученными моделями
# MODEL_VERSION — версия модели по умолчанию (v1 | v2 | ab)
# PORT          — порт, на котором слушает gunicorn
ENV MODEL_DIR=/app/models \
    MODEL_VERSION=v1 \
    PORT=5000 \
    PYTHONUNBUFFERED=1

EXPOSE 5000

# healthcheck — Docker будет опрашивать /health каждые 30 секунд
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Запускаем через gunicorn с 2 воркерами (достаточно для учебного проекта)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "app.api:app"]