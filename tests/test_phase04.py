"""Phase 4 tests — Tabular Models (Lag-1 baseline, Random Forest, XGBoost).

Verifies the three saved prediction arrays plus RF and XGBoost model files
against the existence, shape, range, baseline-correctness, metric-sanity,
and model-loading-consistency assertions from phase_04_tabular.md.
"""
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT_ROOT / "outputs" / "splits" / "test.parquet"
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

PRED_BASELINE = PRED_DIR / "pred_baseline.npy"
PRED_RF = PRED_DIR / "pred_rf.npy"
PRED_XGB = PRED_DIR / "pred_xgb.npy"
RF_PATH = MODEL_DIR / "rf_model.joblib"
XGB_PATH = MODEL_DIR / "xgb_model.json"

TARGET = "Global_active_power"


def _clip(arr):
    return np.clip(arr, 0.0, 20.0)


def _rmse(y_true, y_pred):
    return float(np.sqrt(((np.asarray(y_true) - np.asarray(y_pred)) ** 2).mean()))


@pytest.fixture(scope="module")
def test_df():
    assert TEST_PATH.exists(), f"missing {TEST_PATH} (Phase 3 not done?)"
    return pd.read_parquet(TEST_PATH)


@pytest.fixture(scope="module")
def y_test(test_df):
    return test_df[TARGET].to_numpy()


@pytest.fixture(scope="module")
def X_test(test_df):
    feature_cols = [c for c in test_df.columns if c != TARGET]
    return test_df[feature_cols].to_numpy()


@pytest.fixture(scope="module")
def preds():
    out = {}
    for name, path in [("baseline", PRED_BASELINE), ("rf", PRED_RF), ("xgb", PRED_XGB)]:
        assert path.exists(), f"Phase 4 prediction missing: {path}"
        out[name] = np.load(path)
    return out


# --- existence / loading ---

def test_pred_baseline_exists_and_loads():
    assert PRED_BASELINE.exists()
    arr = np.load(PRED_BASELINE)
    assert isinstance(arr, np.ndarray)


def test_pred_rf_exists_and_loads():
    assert PRED_RF.exists()
    arr = np.load(PRED_RF)
    assert isinstance(arr, np.ndarray)


def test_pred_xgb_exists_and_loads():
    assert PRED_XGB.exists()
    arr = np.load(PRED_XGB)
    assert isinstance(arr, np.ndarray)


def test_rf_model_file_exists_and_loads():
    assert RF_PATH.exists()
    model = joblib.load(RF_PATH)
    assert hasattr(model, "predict")


def test_xgb_model_file_exists_and_loads():
    assert XGB_PATH.exists()
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(XGB_PATH))
    assert hasattr(model, "predict")


# --- shape / no NaN / no inf ---

@pytest.mark.parametrize("name", ["baseline", "rf", "xgb"])
def test_prediction_length_matches_test(name, preds, test_df):
    assert len(preds[name]) == len(test_df), (
        f"{name}: len={len(preds[name])} vs test={len(test_df)}"
    )


@pytest.mark.parametrize("name", ["baseline", "rf", "xgb"])
def test_prediction_no_nan_no_inf(name, preds):
    arr = preds[name]
    assert not np.isnan(arr).any(), f"{name} contains NaN"
    assert np.isfinite(arr).all(), f"{name} contains inf"


# --- range ---

@pytest.mark.parametrize("name", ["baseline", "rf", "xgb"])
def test_prediction_in_physical_range(name, preds):
    arr = preds[name]
    assert (arr >= 0).all(), f"{name}: {(arr < 0).sum()} negative values"
    assert (arr <= 20).all(), f"{name}: {(arr > 20).sum()} values exceed 20 kW"


# --- baseline correctness ---

def test_baseline_equals_lag_1h_column(preds, test_df):
    expected = test_df[f"{TARGET}_lag_1h"].to_numpy()
    np.testing.assert_allclose(preds["baseline"], expected, atol=1e-9, rtol=0)


# --- metric sanity ---

def test_xgb_beats_baseline_on_rmse(preds, y_test):
    rmse_baseline = _rmse(y_test, preds["baseline"])
    rmse_xgb = _rmse(y_test, preds["xgb"])
    assert rmse_xgb < rmse_baseline, (
        f"XGBoost RMSE {rmse_xgb:.4f} not less than baseline RMSE {rmse_baseline:.4f}"
    )


def test_xgb_beats_rf_on_rmse(preds, y_test):
    rmse_rf = _rmse(y_test, preds["rf"])
    rmse_xgb = _rmse(y_test, preds["xgb"])
    assert rmse_xgb < rmse_rf, (
        f"XGBoost RMSE {rmse_xgb:.4f} not less than RF RMSE {rmse_rf:.4f}"
    )


def test_xgb_r2_above_0_5(preds, y_test):
    r2 = r2_score(y_test, preds["xgb"])
    assert r2 > 0.5, f"XGBoost R^2 = {r2:.4f}"


def test_rf_r2_above_0_4(preds, y_test):
    r2 = r2_score(y_test, preds["rf"])
    assert r2 > 0.4, f"Random Forest R^2 = {r2:.4f}"


@pytest.mark.parametrize("name", ["baseline", "rf", "xgb"])
def test_rmse_in_literature_sanity_range(name, preds, y_test):
    r = _rmse(y_test, preds[name])
    assert 0.3 <= r <= 0.7, f"{name} RMSE {r:.4f} outside literature sanity [0.3, 0.7]"


# --- model loading consistency (predictions are clipped to [0, 20] before save) ---

def test_rf_loaded_model_matches_saved_predictions(preds, X_test):
    model = joblib.load(RF_PATH)
    pred = _clip(model.predict(X_test))
    np.testing.assert_allclose(pred, preds["rf"], rtol=1e-6, atol=1e-7)


def test_xgb_loaded_model_matches_saved_predictions(preds, X_test):
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(XGB_PATH))
    pred = _clip(model.predict(X_test))
    np.testing.assert_allclose(pred, preds["xgb"], rtol=1e-5, atol=1e-6)
