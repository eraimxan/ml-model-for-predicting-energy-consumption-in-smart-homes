"""Master integration test — verifies the entire diploma pipeline end-to-end.

This is the final quality gate before thesis writing begins. It does NOT
re-run any phase; it only reads on-disk artifacts produced by P1-P8 plus
the three appendix tasks (ablation, reproducibility note, benchmark).

If any test here fails, fix the root cause in the notebook — these tests
define correctness for the pipeline as a whole.
"""
import json
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

DATA_DIR = PROJECT_ROOT / "outputs" / "data"
SPLITS_DIR = PROJECT_ROOT / "outputs" / "splits"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PREDS_DIR = PROJECT_ROOT / "outputs" / "predictions"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
NOTEBOOK = PROJECT_ROOT / "diploma_pipeline.ipynb"

HOURLY_CLEAN = DATA_DIR / "hourly_clean.parquet"
FEATURES = DATA_DIR / "features.parquet"
TRAIN_PQ = SPLITS_DIR / "train.parquet"
VAL_PQ = SPLITS_DIR / "val.parquet"
TEST_PQ = SPLITS_DIR / "test.parquet"
SCALER = MODELS_DIR / "scaler.joblib"

RF_MODEL = MODELS_DIR / "rf_model.joblib"
XGB_MODEL = MODELS_DIR / "xgb_model.json"
LSTM_MODEL = MODELS_DIR / "lstm_model.keras"
GRU_MODEL = MODELS_DIR / "gru_model.keras"

PRED_BASELINE = PREDS_DIR / "pred_baseline.npy"
PRED_RF = PREDS_DIR / "pred_rf.npy"
PRED_XGB = PREDS_DIR / "pred_xgb.npy"
PRED_LSTM = PREDS_DIR / "pred_lstm.npy"
PRED_GRU = PREDS_DIR / "pred_gru.npy"
PRED_OPTUNA_XGB = PREDS_DIR / "pred_optuna_xgb.npy"

FINAL_CSV = RESULTS_DIR / "final_results.csv"
WF_XGB_CSV = RESULTS_DIR / "walkforward_xgb.csv"
OPTUNA_BEST = RESULTS_DIR / "optuna_best_params.json"
SHAP_RANKING = RESULTS_DIR / "shap_top_features.json"
ABLATION_CSV = RESULTS_DIR / "ablation_study.csv"
BENCHMARK_CSV = RESULTS_DIR / "benchmark_comparison.csv"

FIGURE_FILES = [
    "01_prediction_vs_actual.png",
    "02_shap_summary.png",
    "02b_shap_bar.png",
    "03_model_comparison.png",
    "04_residuals_over_time.png",
    "05_residual_distribution.png",
    "06_walkforward_rmse.png",
    "07_scatter_xgb.png",
    "08_training_history.png",
    "09_eda_patterns.png",
]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

SEQ_LEN = 24


# ===== DATA ARTEFACTS =====================================================

def test_hourly_clean_exists():
    assert HOURLY_CLEAN.exists(), f"missing {HOURLY_CLEAN}"


def test_features_exists_and_has_69_feature_columns():
    assert FEATURES.exists(), f"missing {FEATURES}"
    df = pd.read_parquet(FEATURES)
    assert df.shape[1] == 70, (
        f"features.parquet has {df.shape[1]} columns, expected 70 "
        f"(69 features + 1 target column)"
    )
    assert "Global_active_power" in df.columns, (
        "target column 'Global_active_power' missing from features.parquet"
    )
    n_features = df.shape[1] - 1
    assert n_features == 69, f"feature columns = {n_features}, expected 69"


def test_train_val_test_parquets_exist():
    assert TRAIN_PQ.exists(), f"missing {TRAIN_PQ}"
    assert VAL_PQ.exists(), f"missing {VAL_PQ}"
    assert TEST_PQ.exists(), f"missing {TEST_PQ}"


def test_train_val_chronological_no_overlap():
    train = pd.read_parquet(TRAIN_PQ)
    val = pd.read_parquet(VAL_PQ)
    assert train.index.max() < val.index.min(), (
        f"train ends at {train.index.max()}, val starts at {val.index.min()} "
        "— train must end strictly before val starts"
    )


def test_val_test_chronological_no_overlap():
    val = pd.read_parquet(VAL_PQ)
    test = pd.read_parquet(TEST_PQ)
    assert val.index.max() < test.index.min(), (
        f"val ends at {val.index.max()}, test starts at {test.index.min()} "
        "— val must end strictly before test starts"
    )


def test_scaler_exists():
    assert SCALER.exists(), f"missing {SCALER}"


# ===== MODEL ARTEFACTS ====================================================

def test_rf_model_exists():
    assert RF_MODEL.exists(), f"missing {RF_MODEL}"


def test_xgb_model_exists():
    assert XGB_MODEL.exists(), f"missing {XGB_MODEL}"


def test_lstm_model_exists():
    assert LSTM_MODEL.exists(), f"missing {LSTM_MODEL}"


def test_gru_model_exists():
    assert GRU_MODEL.exists(), f"missing {GRU_MODEL}"


# ===== PREDICTION ARTEFACTS ===============================================

def test_pred_baseline_exists():
    assert PRED_BASELINE.exists(), f"missing {PRED_BASELINE}"


def test_pred_rf_exists():
    assert PRED_RF.exists(), f"missing {PRED_RF}"


def test_pred_xgb_exists():
    assert PRED_XGB.exists(), f"missing {PRED_XGB}"


def test_pred_lstm_exists():
    assert PRED_LSTM.exists(), f"missing {PRED_LSTM}"


def test_pred_gru_exists():
    assert PRED_GRU.exists(), f"missing {PRED_GRU}"


def test_pred_optuna_xgb_exists():
    assert PRED_OPTUNA_XGB.exists(), f"missing {PRED_OPTUNA_XGB}"


# ===== RESULTS QUALITY ====================================================

@pytest.fixture(scope="module")
def final_df():
    assert FINAL_CSV.exists(), f"missing {FINAL_CSV}"
    return pd.read_csv(FINAL_CSV)


def test_final_results_exists():
    assert FINAL_CSV.exists(), f"missing {FINAL_CSV}"


def test_xgboost_has_lowest_rmse_in_final_results(final_df):
    idx = final_df["RMSE"].idxmin()
    best = final_df.loc[idx, "Model"]
    assert "XGBoost" in best, (
        f"lowest-RMSE model in final_results.csv is '{best}', expected XGBoost"
    )


def test_xgboost_rmse_in_expected_band(final_df):
    xgb_row = final_df[final_df["Model"].str.contains("XGBoost")]
    assert len(xgb_row) == 1, f"expected 1 XGBoost row, got {len(xgb_row)}"
    rmse = float(xgb_row["RMSE"].iloc[0])
    assert 0.40 <= rmse <= 0.48, (
        f"XGBoost RMSE {rmse:.4f} outside expected [0.40, 0.48] kW band"
    )


def test_walkforward_xgb_has_7_rows():
    assert WF_XGB_CSV.exists(), f"missing {WF_XGB_CSV}"
    wf = pd.read_csv(WF_XGB_CSV)
    assert len(wf) == 7, f"walkforward_xgb.csv has {len(wf)} rows, expected 7"


def test_optuna_best_params_exists():
    assert OPTUNA_BEST.exists(), f"missing {OPTUNA_BEST}"


def test_shap_top_feature_contains_lag():
    assert SHAP_RANKING.exists(), f"missing {SHAP_RANKING}"
    with open(SHAP_RANKING, "r", encoding="utf-8") as f:
        data = json.load(f)
    top = data["ranking"][0]["feature"]
    assert "lag" in top.lower(), (
        f"top SHAP feature '{top}' does not contain 'lag' in its name"
    )


def test_ablation_csv_exists_for_pipeline():
    assert ABLATION_CSV.exists(), f"missing {ABLATION_CSV}"


def test_ablation_lag_removal_degrades_rmse():
    df = pd.read_csv(ABLATION_CSV)
    full = df[df["variant"].str.contains("Full")].iloc[0]
    no_lag = df[df["variant"].str.contains("No lag")].iloc[0]
    assert float(no_lag["RMSE"]) > float(full["RMSE"]), (
        f"ablation: removing lag features (RMSE {no_lag['RMSE']:.4f}) "
        f"did not degrade vs full (RMSE {full['RMSE']:.4f})"
    )


def test_benchmark_csv_exists_for_pipeline():
    assert BENCHMARK_CSV.exists(), f"missing {BENCHMARK_CSV}"


def test_benchmark_contains_this_work_row():
    df = pd.read_csv(BENCHMARK_CSV)
    matches = df["study"].astype(str).str.contains("This work", case=False, na=False)
    assert matches.any(), "benchmark_comparison.csv has no 'This work' row"


# ===== FIGURES (parametrized) =============================================

@pytest.mark.parametrize("name", FIGURE_FILES)
def test_figure_exists_size_and_png_magic(name):
    p = FIG_DIR / name
    assert p.exists(), f"missing figure {p}"
    size = p.stat().st_size
    assert size > 50_000, f"{name}: {size} bytes (likely blank/corrupted)"
    with open(p, "rb") as f:
        magic = f.read(8)
    assert magic == PNG_MAGIC, f"{name}: bad PNG magic {magic!r}"


# ===== DL FAIRNESS ========================================================

def test_pred_lstm_length_matches_test_minus_seqlen():
    test = pd.read_parquet(TEST_PQ)
    pred = np.load(PRED_LSTM)
    expected = len(test) - SEQ_LEN
    assert len(pred) == expected, (
        f"pred_lstm length {len(pred)} != len(test) - {SEQ_LEN} = {expected}"
    )


# ===== TEMPORAL STABILITY =================================================

def test_xgb_single_split_within_3_std_of_walkforward(final_df):
    wf = pd.read_csv(WF_XGB_CSV)
    wf_mean = wf["RMSE"].mean()
    wf_std = wf["RMSE"].std(ddof=1)
    xgb_row = final_df[final_df["Model"].str.contains("XGBoost")]
    final_xgb = float(xgb_row["RMSE"].iloc[0])
    delta = abs(final_xgb - wf_mean)
    assert delta < 3 * wf_std + 1e-9, (
        f"|XGB single-split {final_xgb:.4f} - WF mean {wf_mean:.4f}| = {delta:.4f} "
        f">= 3 * WF std {wf_std:.4f} — temporal stability violated"
    )


# ===== NOTEBOOK CONTENT ===================================================

@pytest.fixture(scope="module")
def notebook_text():
    assert NOTEBOOK.exists(), f"missing notebook {NOTEBOOK}"
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    parts = []
    for cell in nb["cells"]:
        parts.append("".join(cell.get("source", [])))
    return "\n".join(parts)


def test_notebook_contains_wilcoxon(notebook_text):
    assert "wilcoxon" in notebook_text.lower(), (
        "diploma_pipeline.ipynb does not mention 'wilcoxon' anywhere"
    )


def test_notebook_contains_non_determinism(notebook_text):
    assert "non-determinism" in notebook_text.lower(), (
        "diploma_pipeline.ipynb does not mention 'non-determinism' anywhere"
    )
