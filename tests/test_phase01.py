"""Phase 1 tests — Data Loading and Cleaning.

Verifies outputs/data/hourly_clean.parquet against the structure, quality,
size, and resampling-correctness assertions defined in phase_01_data.md.
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
HOURLY_PATH = PROJECT_ROOT / "outputs" / "data" / "hourly_clean.parquet"
RAW_PATH = PROJECT_ROOT / "household_power_consumption.txt"

EXPECTED_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


@pytest.fixture(scope="module")
def hourly():
    assert HOURLY_PATH.exists(), f"Phase 1 output missing: {HOURLY_PATH}"
    return pd.read_parquet(HOURLY_PATH)


@pytest.fixture(scope="module")
def raw_minute():
    assert RAW_PATH.exists(), f"Raw file missing: {RAW_PATH}"
    df = pd.read_csv(RAW_PATH, sep=";", na_values=["?"], low_memory=False)
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.set_index("timestamp").drop(columns=["Date", "Time"]).sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --- structure ---

def test_output_file_exists():
    assert HOURLY_PATH.exists(), f"missing: {HOURLY_PATH}"


def test_dataframe_has_datetime_index(hourly):
    assert isinstance(hourly.index, pd.DatetimeIndex), (
        f"index is {type(hourly.index).__name__}, not DatetimeIndex"
    )


def test_seven_columns_match_expected_names(hourly):
    assert list(hourly.columns) == EXPECTED_COLUMNS, (
        f"columns are {list(hourly.columns)}"
    )


# --- data quality ---

def test_no_nan_values(hourly):
    total = int(hourly.isna().sum().sum())
    assert total == 0, f"{total} NaN values present"


def test_global_active_power_non_negative(hourly):
    bad = (hourly["Global_active_power"] < 0).sum()
    assert bad == 0, f"{bad} negative Global_active_power values"


def test_sub_metering_non_negative(hourly):
    for col in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        bad = (hourly[col] < 0).sum()
        assert bad == 0, f"{bad} negative values in {col}"


def test_index_is_monotonically_increasing(hourly):
    assert hourly.index.is_monotonic_increasing


def test_hourly_frequency(hourly):
    diffs = hourly.index.to_series().diff().dropna()
    one_hour = pd.Timedelta(hours=1)
    assert (diffs >= one_hour).all(), "some adjacent timestamps are < 1 hour apart"
    pct_one_hour = float((diffs == one_hour).mean())
    assert pct_one_hour > 0.95, (
        f"only {pct_one_hour:.2%} of gaps are exactly 1 hour — hourly resampling may be off"
    )


# --- size ---

def test_row_count_in_expected_range(hourly):
    n = len(hourly)
    assert 30_000 <= n <= 36_000, f"row count {n} outside [30000, 36000]"


def test_earliest_timestamp_in_december_2006(hourly):
    earliest = hourly.index.min()
    assert earliest.year == 2006 and earliest.month == 12, (
        f"earliest is {earliest}"
    )


def test_latest_timestamp_in_november_2010(hourly):
    latest = hourly.index.max()
    assert latest.year == 2010 and latest.month == 11, f"latest is {latest}"


# --- resampling correctness ---

def test_resampling_uses_mean_for_rate_and_sum_for_energy(hourly, raw_minute):
    """Pick a fully-observed hour from the raw data and verify aggregation."""
    raw_count = raw_minute["Global_active_power"].resample("1h").count()
    fully_observed = raw_count.index[raw_count == 60]
    assert len(fully_observed) > 0, "no fully-observed hours in raw data"

    target = fully_observed[len(fully_observed) // 2]
    bucket = raw_minute.loc[
        (raw_minute.index >= target)
        & (raw_minute.index < target + pd.Timedelta(hours=1))
    ]

    assert target in hourly.index, f"hour {target} missing from hourly output"

    expected_mean = bucket["Global_active_power"].mean()
    expected_sum_sm1 = bucket["Sub_metering_1"].sum()

    assert hourly.loc[target, "Global_active_power"] == pytest.approx(
        expected_mean, rel=1e-5, abs=1e-6
    ), "Global_active_power should be the mean of the minute-level values in the hour"
    assert hourly.loc[target, "Sub_metering_1"] == pytest.approx(
        expected_sum_sm1, rel=1e-5, abs=1e-6
    ), "Sub_metering_1 should be the sum of the minute-level values in the hour"
