"""Benchmark comparison tests — verifies outputs/results/benchmark_comparison.csv
contextualises the diploma's XGBoost result against four published studies.

Comparison is indirect (each cited study uses a different dataset), so every
published row must carry a note explaining which dataset was used and why direct
numerical comparison is dataset-dependent.
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
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
BENCH_CSV = RESULTS_DIR / "benchmark_comparison.csv"

EXPECTED_COLUMNS = ["study", "model", "dataset", "RMSE", "MAE", "R2", "note"]
OUR_RMSE = 0.4402
OUR_RMSE_TOL = 0.005


@pytest.fixture(scope="module")
def df():
    assert BENCH_CSV.exists(), f"Benchmark output missing: {BENCH_CSV}"
    return pd.read_csv(BENCH_CSV)


def test_benchmark_csv_exists():
    assert BENCH_CSV.exists(), f"Benchmark output missing: {BENCH_CSV}"


def test_benchmark_csv_has_all_columns(df):
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Expected columns {EXPECTED_COLUMNS}, got {list(df.columns)}"
    )


def test_benchmark_contains_this_work(df):
    matches = df["study"].astype(str).str.contains("This work", case=False, na=False)
    assert matches.any(), "no row whose 'study' field contains 'This work'"


def test_benchmark_contains_salman(df):
    matches = df.apply(lambda r: "salman" in " ".join(map(str, r.values)).lower(), axis=1)
    assert matches.any(), "no row mentioning Salman"


def test_benchmark_contains_han(df):
    matches = df.apply(lambda r: "han" in " ".join(map(str, r.values)).lower(), axis=1)
    assert matches.any(), "no row mentioning Han"


def test_benchmark_contains_teslyuk(df):
    matches = df.apply(lambda r: "teslyuk" in " ".join(map(str, r.values)).lower(), axis=1)
    assert matches.any(), "no row mentioning Teslyuk"


def test_benchmark_our_rmse_value(df):
    our_rows = df[df["study"].astype(str).str.contains("This work", case=False, na=False)]
    assert len(our_rows) >= 1, "no 'This work' row to read RMSE from"
    rmse = float(our_rows.iloc[0]["RMSE"])
    assert abs(rmse - OUR_RMSE) <= OUR_RMSE_TOL, (
        f"This-work RMSE {rmse:.4f} differs from {OUR_RMSE} by more than {OUR_RMSE_TOL}"
    )


def test_benchmark_at_least_four_rows(df):
    assert len(df) >= 4, f"expected at least 4 rows, got {len(df)}"


def test_benchmark_published_rows_have_notes(df):
    published = df[~df["study"].astype(str).str.contains("This work", case=False, na=False)]
    assert len(published) >= 1, "no published rows to validate notes for"
    for _, row in published.iterrows():
        note = str(row["note"]) if not pd.isna(row["note"]) else ""
        assert len(note.strip()) >= 20, (
            f"published row {row['study']!r} has note shorter than 20 chars: {note!r}"
        )
