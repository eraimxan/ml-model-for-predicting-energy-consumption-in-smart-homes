"""Phase 3 tests — Chronological Split and Scaler.

Verifies the three split parquets and the joblib scaler against the
existence, no-overlap, completeness, proportion, ordering, scaler-fit,
and column-presence assertions defined in phase_03_split.md.
"""
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "outputs" / "splits"
TRAIN_PATH = SPLITS_DIR / "train.parquet"
VAL_PATH = SPLITS_DIR / "val.parquet"
TEST_PATH = SPLITS_DIR / "test.parquet"
SCALER_PATH = PROJECT_ROOT / "outputs" / "models" / "scaler.joblib"
FEATURES_PATH = PROJECT_ROOT / "outputs" / "data" / "features.parquet"

TARGET = "Global_active_power"


@pytest.fixture(scope="module")
def splits():
    for p in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        assert p.exists(), f"Phase 3 output missing: {p}"
    return {
        "train": pd.read_parquet(TRAIN_PATH),
        "val": pd.read_parquet(VAL_PATH),
        "test": pd.read_parquet(TEST_PATH),
    }


@pytest.fixture(scope="module")
def scaler():
    assert SCALER_PATH.exists(), f"Scaler missing: {SCALER_PATH}"
    return joblib.load(SCALER_PATH)


@pytest.fixture(scope="module")
def features():
    assert FEATURES_PATH.exists()
    return pd.read_parquet(FEATURES_PATH)


# --- file existence ---

def test_train_file_exists():
    assert TRAIN_PATH.exists()


def test_val_file_exists():
    assert VAL_PATH.exists()


def test_test_file_exists():
    assert TEST_PATH.exists()


def test_scaler_file_exists():
    assert SCALER_PATH.exists()


# --- no overlap (most critical) ---

def test_train_strictly_before_val(splits):
    assert splits["train"].index.max() < splits["val"].index.min(), (
        f"train.max={splits['train'].index.max()} not < val.min={splits['val'].index.min()}"
    )


def test_val_strictly_before_test(splits):
    assert splits["val"].index.max() < splits["test"].index.min()


def test_no_timestamp_in_two_splits(splits):
    train_ts = set(splits["train"].index)
    val_ts = set(splits["val"].index)
    test_ts = set(splits["test"].index)
    assert not (train_ts & val_ts), "train and val share timestamps"
    assert not (train_ts & test_ts), "train and test share timestamps"
    assert not (val_ts & test_ts), "val and test share timestamps"


# --- completeness ---

def test_split_lengths_sum_to_features(splits, features):
    total = sum(len(splits[k]) for k in ("train", "val", "test"))
    assert total == len(features), (
        f"sum of splits = {total}, features = {len(features)}"
    )


# --- proportions ---

def test_train_proportion(splits):
    total = sum(len(splits[k]) for k in ("train", "val", "test"))
    p = len(splits["train"]) / total
    assert 0.68 <= p <= 0.72, f"train proportion {p:.4f} outside [0.68, 0.72]"


def test_val_proportion(splits):
    total = sum(len(splits[k]) for k in ("train", "val", "test"))
    p = len(splits["val"]) / total
    assert 0.13 <= p <= 0.17, f"val proportion {p:.4f} outside [0.13, 0.17]"


def test_test_proportion(splits):
    total = sum(len(splits[k]) for k in ("train", "val", "test"))
    p = len(splits["test"]) / total
    assert 0.13 <= p <= 0.17, f"test proportion {p:.4f} outside [0.13, 0.17]"


# --- chronological order ---

def test_train_index_monotonic(splits):
    assert splits["train"].index.is_monotonic_increasing


def test_val_index_monotonic(splits):
    assert splits["val"].index.is_monotonic_increasing


def test_test_index_monotonic(splits):
    assert splits["test"].index.is_monotonic_increasing


# --- scaler ---

def test_scaler_n_features(scaler):
    assert scaler.n_features_in_ == 69, (
        f"scaler.n_features_in_ = {scaler.n_features_in_}"
    )


def test_scaler_zeroes_mean_and_unit_std_on_train(splits, scaler):
    feature_cols = [c for c in splits["train"].columns if c != TARGET]
    assert len(feature_cols) == 69
    X_train = splits["train"][feature_cols].to_numpy()
    Z = scaler.transform(X_train)
    means = Z.mean(axis=0)
    stds = Z.std(axis=0, ddof=0)
    assert np.all(np.abs(means) < 1e-2), (
        f"max |mean| over scaled train features = {np.max(np.abs(means)):.4g}"
    )
    assert np.all(np.abs(stds - 1.0) < 1e-2), (
        f"max |std-1| over scaled train features = {np.max(np.abs(stds - 1.0)):.4g}"
    )


def test_scaler_was_fit_on_train_only(splits, scaler):
    """scaler.mean_ must equal mean of X_train (not all features). This
    confirms it was not refit on, e.g., the full feature matrix."""
    feature_cols = [c for c in splits["train"].columns if c != TARGET]
    X_train = splits["train"][feature_cols].to_numpy()
    np.testing.assert_allclose(
        scaler.mean_, X_train.mean(axis=0), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        scaler.scale_, X_train.std(axis=0, ddof=0), rtol=1e-10, atol=1e-12
    )


def test_scaler_does_not_zero_mean_val(splits, scaler):
    """The val split was unseen during fitting, so its scaled mean should
    deviate from zero on at least some features. This is the empirical
    confirmation that the scaler was train-only."""
    feature_cols = [c for c in splits["val"].columns if c != TARGET]
    X_val = splits["val"][feature_cols].to_numpy()
    Z_val = scaler.transform(X_val)
    val_means = np.abs(Z_val.mean(axis=0))
    assert np.any(val_means > 0.01), (
        "scaled val features all have |mean| <= 0.01 — scaler may have seen val data"
    )


# --- columns ---

def test_target_column_in_all_splits(splits):
    for name, df in splits.items():
        assert TARGET in df.columns, f"{name} missing TARGET"


def test_all_69_feature_columns_in_all_splits(splits, features):
    expected_features = [c for c in features.columns if c != TARGET]
    assert len(expected_features) == 69
    for name, df in splits.items():
        missing = [c for c in expected_features if c not in df.columns]
        assert not missing, f"{name} missing features: {missing[:5]}"
        assert df.shape[1] == 70, f"{name} has {df.shape[1]} cols (expected 70)"
