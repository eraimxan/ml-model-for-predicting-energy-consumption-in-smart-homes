"""Phase 2 tests — Feature Engineering.

Verifies outputs/data/features.parquet against structure, count, leakage,
size, and value-range assertions defined in phase_02_features.md.
"""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "outputs" / "data" / "features.parquet"

TARGET = "Global_active_power"
COVARIATES = [
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
LAG_VARS = [TARGET] + COVARIATES  # 7
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]  # 8
ROLLING_WINDOWS = [3, 6, 24]

EXPECTED_LAG_FEATURES = [f"{v}_lag_{h}h" for v in LAG_VARS for h in LAGS]
EXPECTED_ROLLING_FEATURES = (
    [f"{TARGET}_rmean_{w}h" for w in ROLLING_WINDOWS]
    + [f"{TARGET}_rstd_{w}h" for w in ROLLING_WINDOWS]
)
EXPECTED_TIME_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "mon_sin", "mon_cos", "is_weekend",
]
EXPECTED_FEATURES = (
    EXPECTED_LAG_FEATURES + EXPECTED_ROLLING_FEATURES + EXPECTED_TIME_FEATURES
)


@pytest.fixture(scope="module")
def features():
    assert FEATURES_PATH.exists(), f"Phase 2 output missing: {FEATURES_PATH}"
    return pd.read_parquet(FEATURES_PATH)


# --- structure ---

def test_output_file_exists():
    assert FEATURES_PATH.exists(), f"missing: {FEATURES_PATH}"


def test_dataframe_has_datetime_index(features):
    assert isinstance(features.index, pd.DatetimeIndex)


def test_seventy_columns(features):
    assert features.shape[1] == 70, f"got {features.shape[1]} columns"


def test_target_column_present(features):
    assert TARGET in features.columns


def test_all_69_feature_names_present(features):
    missing = [f for f in EXPECTED_FEATURES if f not in features.columns]
    assert not missing, f"missing features: {missing[:10]}"


def test_no_extra_columns(features):
    extra = set(features.columns) - set(EXPECTED_FEATURES) - {TARGET}
    assert not extra, f"unexpected extra columns: {extra}"


def test_no_nan_values(features):
    total = int(features.isna().sum().sum())
    assert total == 0, f"{total} NaN values"


# --- feature counts ---

def test_56_lag_features(features):
    cols = [c for c in features.columns if "_lag_" in c]
    assert len(cols) == 56, f"got {len(cols)} lag features"


def test_6_rolling_features(features):
    cols = [c for c in features.columns if ("_rmean_" in c or "_rstd_" in c)]
    assert len(cols) == 6, f"got {len(cols)} rolling features"


def test_7_time_features(features):
    cols = [c for c in features.columns if c in EXPECTED_TIME_FEATURES]
    assert len(cols) == 7


# --- leakage ---

def test_lag_1h_equals_previous_row_target(features):
    rng = np.random.default_rng(SEED)
    target = features[TARGET].to_numpy()
    lag1 = features[f"{TARGET}_lag_1h"].to_numpy()
    n = len(features)
    for i in rng.integers(low=1, high=n, size=100):
        assert np.isclose(lag1[i], target[i - 1], atol=1e-9, rtol=0), (
            f"row {i}: lag_1h={lag1[i]} != target[i-1]={target[i - 1]}"
        )


def test_lag_168h_equals_168_rows_earlier(features):
    rng = np.random.default_rng(SEED)
    target = features[TARGET].to_numpy()
    lag168 = features[f"{TARGET}_lag_168h"].to_numpy()
    n = len(features)
    for i in rng.integers(low=168, high=n, size=100):
        assert np.isclose(lag168[i], target[i - 168], atol=1e-9, rtol=0)


def test_rolling_mean_3h_is_past_only(features):
    """rmean_3h at row t must equal mean of lag_1h, lag_2h, lag_3h at row t."""
    rng = np.random.default_rng(SEED + 1)
    n = len(features)
    rmean = features[f"{TARGET}_rmean_3h"].to_numpy()
    l1 = features[f"{TARGET}_lag_1h"].to_numpy()
    l2 = features[f"{TARGET}_lag_2h"].to_numpy()
    l3 = features[f"{TARGET}_lag_3h"].to_numpy()
    for i in rng.integers(low=3, high=n, size=100):
        expected = (l1[i] + l2[i] + l3[i]) / 3.0
        assert np.isclose(rmean[i], expected, rtol=1e-7, atol=1e-9), (
            f"row {i}: rmean_3h={rmean[i]} vs mean(lags)={expected}"
        )


def test_target_not_in_feature_list(features):
    feature_cols = [c for c in features.columns if c != TARGET]
    assert TARGET not in feature_cols
    assert len(feature_cols) == 69


# --- size ---

def test_row_count_in_expected_range(features):
    n = len(features)
    assert 33_000 <= n <= 35_000, f"row count {n} outside [33000, 35000]"


# --- value ranges ---

def test_target_lag_features_non_negative(features):
    for h in LAGS:
        col = f"{TARGET}_lag_{h}h"
        bad = (features[col] < 0).sum()
        assert bad == 0, f"{col} has {bad} negative values"


def test_cyclical_features_bounded(features):
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "mon_sin", "mon_cos"]:
        assert features[col].between(-1.0, 1.0).all(), f"{col} outside [-1, 1]"


def test_is_weekend_binary(features):
    vals = set(np.unique(features["is_weekend"].to_numpy()))
    assert vals.issubset({0, 1}), f"is_weekend has values {vals}"
