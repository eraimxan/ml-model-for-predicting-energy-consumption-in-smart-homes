from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from api.features import FEATURE_NAMES, TARGET, WINDOWS, build_features
from api.schemas import HourlyReading

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent
HOURLY_PATH = DATA_ROOT / "outputs" / "data" / "hourly_clean.parquet"
TEST_PATH = DATA_ROOT / "outputs" / "splits" / "test.parquet"
SCALER_PATH = DATA_ROOT / "outputs" / "models" / "scaler.joblib"


def _readings_from_hourly(n: int = 192) -> list[HourlyReading]:
    df = pd.read_parquet(HOURLY_PATH).iloc[-n:]
    out: list[HourlyReading] = []
    for ts, row in df.iterrows():
        out.append(
            HourlyReading(
                timestamp=ts,
                Global_active_power=float(row["Global_active_power"]),
                Global_reactive_power=float(row["Global_reactive_power"]),
                Voltage=float(row["Voltage"]),
                Global_intensity=float(row["Global_intensity"]),
                Sub_metering_1=float(row["Sub_metering_1"]),
                Sub_metering_2=float(row["Sub_metering_2"]),
                Sub_metering_3=float(row["Sub_metering_3"]),
            )
        )
    return out


def test_feature_count_is_69() -> None:
    assert len(FEATURE_NAMES) == 69


def test_feature_names_match_scaler() -> None:
    scaler = joblib.load(SCALER_PATH)
    assert int(scaler.n_features_in_) == 69
    if hasattr(scaler, "feature_names_in_"):
        assert list(scaler.feature_names_in_) == FEATURE_NAMES
    else:
        cols = list(pd.read_parquet(TEST_PATH).columns)
        feature_cols = [c for c in cols if c != "Global_active_power"]
        assert feature_cols == FEATURE_NAMES


def test_no_nan_in_output() -> None:
    X = build_features(_readings_from_hourly(192))
    assert not np.isnan(X).any()


def test_output_shape_is_1_by_69() -> None:
    X = build_features(_readings_from_hourly(192))
    assert X.shape == (1, 69)


def test_rolling_features_use_shift() -> None:
    readings = _readings_from_hourly(192)
    X = build_features(readings)
    gap_series = np.array([r.Global_active_power for r in readings], dtype=np.float64)
    for w in WINDOWS:
        rmean_idx = FEATURE_NAMES.index(f"{TARGET}_rmean_{w}h")
        rstd_idx = FEATURE_NAMES.index(f"{TARGET}_rstd_{w}h")
        causal_window = gap_series[-w:]
        expected_mean = float(causal_window.mean())
        expected_std = float(np.std(causal_window, ddof=1))
        assert X[0, rmean_idx] == pytest.approx(expected_mean, abs=1e-9), (
            f"rmean_{w}h leaks the predicted row"
        )
        assert X[0, rstd_idx] == pytest.approx(expected_std, abs=1e-9), (
            f"rstd_{w}h leaks the predicted row"
        )
    lag1_idx = FEATURE_NAMES.index(f"{TARGET}_lag_1h")
    assert X[0, lag1_idx] == pytest.approx(float(gap_series[-1]), abs=1e-12)
    lag168_idx = FEATURE_NAMES.index(f"{TARGET}_lag_168h")
    assert X[0, lag168_idx] == pytest.approx(float(gap_series[-168]), abs=1e-12)
