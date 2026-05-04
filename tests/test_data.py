"""
Тесты корректности данных и пайплайна предобработки.

Запуск:
    pytest tests/test_data.py -v
"""
from pathlib import Path

import pandas as pd
import pytest

from default_of_credit_card_clients.config import PROCESSED_DATA_DIR
from default_of_credit_card_clients.dataset import FEATURE_COLS, TARGET_COL, clean


# ── Тесты функции clean() ─────────────────────────────────────────────────────

def test_clean_drops_id_column():
    """Функция clean() должна удалять столбец ID."""
    import numpy as np
    # Создаём минимальный DataFrame с нужными столбцами + ID
    data = {col: np.zeros(5) for col in FEATURE_COLS + [TARGET_COL]}
    data["ID"] = range(5)
    df = pd.DataFrame(data)

    df_clean = clean(df)
    assert "ID" not in df_clean.columns


def test_clean_target_is_binary():
    """После clean() целевая переменная содержит только 0 и 1."""
    import numpy as np
    data = {col: np.zeros(10) for col in FEATURE_COLS}
    data[TARGET_COL] = [0, 1] * 5
    df_clean = clean(pd.DataFrame(data))
    assert set(df_clean[TARGET_COL].unique()).issubset({0, 1})


def test_clean_no_nan_after_clean():
    """clean() не должен оставлять NaN в данных."""
    import numpy as np
    data = {col: np.ones(10) for col in FEATURE_COLS + [TARGET_COL]}
    df = pd.DataFrame(data)
    df.iloc[0, 0] = np.nan  # добавляем одну строку с NaN
    df_clean = clean(df)
    assert df_clean.isna().sum().sum() == 0


# ── Тест обработанного файла (если он существует) ─────────────────────────────

@pytest.mark.skipif(
    not (PROCESSED_DATA_DIR / "dataset.csv").exists(),
    reason="Обработанный датасет не найден. Запустите: python -m default_of_credit_card_clients.dataset",
)
def test_processed_dataset_shape():
    """Обработанный датасет должен иметь все ожидаемые столбцы."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "dataset.csv")
    assert TARGET_COL in df.columns
    for col in FEATURE_COLS:
        assert col in df.columns, f"Отсутствует столбец: {col}"
    assert len(df) > 1000, "Датасет должен содержать более 1000 строк"


@pytest.mark.skipif(
    not (PROCESSED_DATA_DIR / "dataset.csv").exists(),
    reason="Обработанный датасет не найден.",
)
def test_processed_dataset_target_distribution():
    """Доля дефолтов должна быть в разумных пределах (10–40%)."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "dataset.csv")
    default_rate = df[TARGET_COL].mean()
    assert 0.10 <= default_rate <= 0.40, (
        f"Неожиданная доля дефолтов: {default_rate:.3f}"
    )