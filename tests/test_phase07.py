"""Phase 7 tests — Deep Learning (LSTM and GRU on multivariate sequences).

Verifies the saved predictions, model files, and training-history JSONs
against the existence, shape, range, history, architecture, metric-sanity,
and fairness assertions defined in phase_07_deeplearning.md.

The fairness test (`test_test_seq_third_dim_is_69_features`) is the key
correction from v1, which fed LSTM only the raw target series (n, 24, 1).
"""
import json
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
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

TRAIN_PATH = SPLITS_DIR / "train.parquet"
VAL_PATH = SPLITS_DIR / "val.parquet"
TEST_PATH = SPLITS_DIR / "test.parquet"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

PRED_LSTM = PRED_DIR / "pred_lstm.npy"
PRED_GRU = PRED_DIR / "pred_gru.npy"
LSTM_MODEL = MODELS_DIR / "lstm_model.keras"
GRU_MODEL = MODELS_DIR / "gru_model.keras"
LSTM_HIST = RESULTS_DIR / "lstm_history.json"
GRU_HIST = RESULTS_DIR / "gru_history.json"

TARGET = "Global_active_power"
SEQ_LEN = 24
N_FEATURES = 69


def _build_sequences(X_scaled: np.ndarray, y: np.ndarray, seq_len: int):
    """Mirror of the notebook's sequence builder, used to recompute X_test_seq
    for the fairness assertion and metric checks."""
    n = len(X_scaled)
    if n <= seq_len:
        raise ValueError(f"need more than {seq_len} rows, got {n}")
    X_seq = np.stack([X_scaled[i - seq_len : i] for i in range(seq_len, n)])
    y_seq = y[seq_len:]
    return X_seq, y_seq


@pytest.fixture(scope="module")
def test_df():
    assert TEST_PATH.exists(), f"missing {TEST_PATH} (Phase 3 not done?)"
    return pd.read_parquet(TEST_PATH)


@pytest.fixture(scope="module")
def scaler():
    assert SCALER_PATH.exists(), f"missing {SCALER_PATH} (Phase 3 not done?)"
    return joblib.load(SCALER_PATH)


@pytest.fixture(scope="module")
def test_seq(test_df, scaler):
    feature_cols = [c for c in test_df.columns if c != TARGET]
    X = test_df[feature_cols].to_numpy()
    y = test_df[TARGET].to_numpy()
    X_scaled = scaler.transform(X)
    return _build_sequences(X_scaled, y, SEQ_LEN)


@pytest.fixture(scope="module")
def pred_lstm():
    assert PRED_LSTM.exists(), f"Phase 7 output missing: {PRED_LSTM}"
    return np.load(PRED_LSTM)


@pytest.fixture(scope="module")
def pred_gru():
    assert PRED_GRU.exists(), f"Phase 7 output missing: {PRED_GRU}"
    return np.load(PRED_GRU)


@pytest.fixture(scope="module")
def lstm_history():
    assert LSTM_HIST.exists(), f"Phase 7 output missing: {LSTM_HIST}"
    with open(LSTM_HIST, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def gru_history():
    assert GRU_HIST.exists(), f"Phase 7 output missing: {GRU_HIST}"
    with open(GRU_HIST, "r", encoding="utf-8") as f:
        return json.load(f)


# --- existence ---

def test_pred_lstm_exists():
    assert PRED_LSTM.exists()


def test_pred_gru_exists():
    assert PRED_GRU.exists()


def test_lstm_model_file_exists():
    assert LSTM_MODEL.exists()


def test_gru_model_file_exists():
    assert GRU_MODEL.exists()


def test_lstm_history_file_exists():
    assert LSTM_HIST.exists()


def test_gru_history_file_exists():
    assert GRU_HIST.exists()


# --- prediction shape ---

def test_pred_lstm_length_matches_test_minus_seq_len(pred_lstm, test_df):
    expected = len(test_df) - SEQ_LEN
    assert len(pred_lstm) == expected, (
        f"len(pred_lstm)={len(pred_lstm)} != len(test)-SEQ_LEN={expected}"
    )


def test_pred_gru_length_matches_pred_lstm(pred_lstm, pred_gru):
    assert len(pred_gru) == len(pred_lstm)


def test_pred_lstm_no_nan_no_inf(pred_lstm):
    assert not np.isnan(pred_lstm).any()
    assert np.isfinite(pred_lstm).all()


def test_pred_gru_no_nan_no_inf(pred_gru):
    assert not np.isnan(pred_gru).any()
    assert np.isfinite(pred_gru).all()


# --- prediction range (clipped to [0, 20] on save) ---

def test_pred_lstm_in_physical_range(pred_lstm):
    assert (pred_lstm >= 0).all(), f"{(pred_lstm < 0).sum()} negative LSTM predictions"
    assert (pred_lstm <= 20).all(), f"{(pred_lstm > 20).sum()} LSTM predictions exceed 20 kW"


def test_pred_gru_in_physical_range(pred_gru):
    assert (pred_gru >= 0).all(), f"{(pred_gru < 0).sum()} negative GRU predictions"
    assert (pred_gru <= 20).all(), f"{(pred_gru > 20).sum()} GRU predictions exceed 20 kW"


# --- training history ---

def test_lstm_history_has_loss_keys(lstm_history):
    assert "loss" in lstm_history
    assert "val_loss" in lstm_history


def test_gru_history_has_loss_keys(gru_history):
    assert "loss" in gru_history
    assert "val_loss" in gru_history


def test_lstm_epoch_count_in_range(lstm_history):
    n = len(lstm_history["loss"])
    assert 10 <= n <= 100, f"LSTM trained for {n} epochs (expected 10-100, early stopping)"


def test_gru_epoch_count_in_range(gru_history):
    n = len(gru_history["loss"])
    assert 10 <= n <= 100, f"GRU trained for {n} epochs (expected 10-100, early stopping)"


def test_lstm_val_loss_improved_from_epoch_1(lstm_history):
    val = lstm_history["val_loss"]
    best = min(val)
    first = val[0]
    assert best < first, f"LSTM best val_loss {best:.4f} not below epoch-1 val_loss {first:.4f}"


def test_gru_val_loss_improved_from_epoch_1(gru_history):
    val = gru_history["val_loss"]
    best = min(val)
    first = val[0]
    assert best < first, f"GRU best val_loss {best:.4f} not below epoch-1 val_loss {first:.4f}"


# --- model architecture ---

def test_lstm_model_loads_and_has_correct_io_shape():
    from tensorflow.keras.models import load_model
    model = load_model(LSTM_MODEL)
    in_shape = model.input_shape
    out_shape = model.output_shape
    assert in_shape == (None, SEQ_LEN, N_FEATURES), (
        f"LSTM input_shape={in_shape}, expected (None, {SEQ_LEN}, {N_FEATURES})"
    )
    assert out_shape == (None, 1), f"LSTM output_shape={out_shape}, expected (None, 1)"


def test_gru_model_loads_and_has_correct_io_shape():
    from tensorflow.keras.models import load_model
    model = load_model(GRU_MODEL)
    in_shape = model.input_shape
    out_shape = model.output_shape
    assert in_shape == (None, SEQ_LEN, N_FEATURES), (
        f"GRU input_shape={in_shape}, expected (None, {SEQ_LEN}, {N_FEATURES})"
    )
    assert out_shape == (None, 1), f"GRU output_shape={out_shape}, expected (None, 1)"


def test_lstm_has_six_layers():
    """Architecture: LSTM -> Dropout -> LSTM -> Dropout -> Dense -> Dense (= 6 layers)."""
    from tensorflow.keras.models import load_model
    model = load_model(LSTM_MODEL)
    assert len(model.layers) == 6, (
        f"LSTM has {len(model.layers)} layers, expected 6 "
        f"(LSTM, Dropout, LSTM, Dropout, Dense, Dense)"
    )


def test_gru_has_six_layers():
    from tensorflow.keras.models import load_model
    model = load_model(GRU_MODEL)
    assert len(model.layers) == 6, (
        f"GRU has {len(model.layers)} layers, expected 6 "
        f"(GRU, Dropout, GRU, Dropout, Dense, Dense)"
    )


# --- metric sanity ---

def test_lstm_rmse_in_sanity_range(pred_lstm, test_seq):
    _, y_test_seq = test_seq
    rmse = float(np.sqrt(((y_test_seq - pred_lstm) ** 2).mean()))
    assert 0.3 <= rmse <= 0.75, f"LSTM RMSE {rmse:.4f} outside sanity [0.3, 0.75]"


def test_gru_rmse_in_sanity_range(pred_gru, test_seq):
    _, y_test_seq = test_seq
    rmse = float(np.sqrt(((y_test_seq - pred_gru) ** 2).mean()))
    assert 0.3 <= rmse <= 0.75, f"GRU RMSE {rmse:.4f} outside sanity [0.3, 0.75]"


def test_lstm_and_gru_rmse_close(pred_lstm, pred_gru, test_seq):
    """Same features + same sequences => similar RMSE. A gap > 0.1 kW signals a bug."""
    _, y_test_seq = test_seq
    rmse_lstm = float(np.sqrt(((y_test_seq - pred_lstm) ** 2).mean()))
    rmse_gru = float(np.sqrt(((y_test_seq - pred_gru) ** 2).mean()))
    assert abs(rmse_lstm - rmse_gru) < 0.1, (
        f"|LSTM RMSE {rmse_lstm:.4f} - GRU RMSE {rmse_gru:.4f}| >= 0.1 kW"
    )


# --- fairness test (the v1-bug guard) ---

def test_test_seq_third_dim_is_69_features(test_seq):
    """v1 bug: LSTM received (n, 24, 1) raw target sequences while XGBoost got 69
    features. Phase 7 must build sequences from the full 69-feature scaled matrix."""
    X_test_seq, _ = test_seq
    assert X_test_seq.ndim == 3, f"X_test_seq.ndim={X_test_seq.ndim}, expected 3"
    assert X_test_seq.shape[1] == SEQ_LEN, (
        f"X_test_seq.shape[1]={X_test_seq.shape[1]}, expected {SEQ_LEN}"
    )
    assert X_test_seq.shape[2] == N_FEATURES, (
        f"X_test_seq.shape[2]={X_test_seq.shape[2]}, expected {N_FEATURES} "
        f"(if 1, this is the v1 LSTM-1D-input bug)"
    )


def test_lstm_input_shape_accepts_69_feature_sequences():
    from tensorflow.keras.models import load_model
    model = load_model(LSTM_MODEL)
    assert model.input_shape[2] == N_FEATURES, (
        f"LSTM input_shape[2]={model.input_shape[2]}, expected {N_FEATURES}"
    )


def test_gru_input_shape_accepts_69_feature_sequences():
    from tensorflow.keras.models import load_model
    model = load_model(GRU_MODEL)
    assert model.input_shape[2] == N_FEATURES, (
        f"GRU input_shape[2]={model.input_shape[2]}, expected {N_FEATURES}"
    )
