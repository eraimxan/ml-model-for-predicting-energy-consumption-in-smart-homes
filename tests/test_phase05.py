"""Phase 5 tests — Optuna Bayesian Hyperparameter Optimisation.

Verifies the saved best-params JSON and study-trials CSV against the
existence, structure, study-quality, comparison, and refit-correctness
assertions defined in phase_05_optuna.md.
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
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
BEST_JSON = RESULTS_DIR / "optuna_best_params.json"
TRIALS_CSV = RESULTS_DIR / "optuna_study_trials.csv"
PRED_OPTUNA = PROJECT_ROOT / "outputs" / "predictions" / "pred_optuna_xgb.npy"
PRED_DEFAULT = PROJECT_ROOT / "outputs" / "predictions" / "pred_xgb.npy"
TEST_PARQUET = PROJECT_ROOT / "outputs" / "splits" / "test.parquet"
TRAIN_PARQUET = PROJECT_ROOT / "outputs" / "splits" / "train.parquet"
VAL_PARQUET = PROJECT_ROOT / "outputs" / "splits" / "val.parquet"

EXPECTED_PARAM_KEYS = {
    "n_estimators", "max_depth", "learning_rate",
    "subsample", "colsample_bytree", "min_child_weight",
    "reg_alpha", "reg_lambda", "gamma",
}


@pytest.fixture(scope="module")
def best():
    assert BEST_JSON.exists(), f"Phase 5 output missing: {BEST_JSON}"
    with open(BEST_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def trials():
    assert TRIALS_CSV.exists(), f"Phase 5 output missing: {TRIALS_CSV}"
    return pd.read_csv(TRIALS_CSV)


# --- file existence / loading ---

def test_best_params_json_exists_and_loads():
    assert BEST_JSON.exists()
    with open(BEST_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_trials_csv_exists():
    assert TRIALS_CSV.exists()


# --- best params structure ---

def test_all_9_param_keys_present(best):
    params = best["params"]
    missing = EXPECTED_PARAM_KEYS - set(params.keys())
    assert not missing, f"missing param keys: {missing}"


def test_n_estimators_in_range(best):
    n = best["params"]["n_estimators"]
    assert isinstance(n, int)
    assert 200 <= n <= 1200, n


def test_max_depth_in_range(best):
    d = best["params"]["max_depth"]
    assert isinstance(d, int)
    assert 3 <= d <= 10, d


def test_learning_rate_in_range(best):
    lr = best["params"]["learning_rate"]
    assert 5e-3 <= lr <= 0.3, lr


def test_subsample_in_range(best):
    s = best["params"]["subsample"]
    assert 0.5 <= s <= 1.0, s


def test_colsample_bytree_in_range(best):
    c = best["params"]["colsample_bytree"]
    assert 0.5 <= c <= 1.0, c


def test_min_child_weight_in_range(best):
    mcw = best["params"]["min_child_weight"]
    assert 1 <= mcw <= 10, mcw


def test_reg_alpha_in_range(best):
    a = best["params"]["reg_alpha"]
    assert 1e-3 <= a <= 3.0, a


def test_reg_lambda_in_range(best):
    l = best["params"]["reg_lambda"]
    assert 1e-3 <= l <= 6.0, l


def test_gamma_in_range(best):
    g = best["params"]["gamma"]
    assert 1e-6 <= g <= 1.0, g


# --- study quality ---

def test_at_least_40_trials_in_csv(trials):
    assert len(trials) >= 40, f"only {len(trials)} trials in CSV"


def test_csv_has_value_column(trials):
    assert "value" in trials.columns


def test_best_val_rmse_below_0_55(trials):
    completed = trials.dropna(subset=["value"])
    assert len(completed) > 0
    best_val = completed["value"].min()
    assert best_val < 0.55, f"best val RMSE = {best_val}"


def test_search_explored_diverse_n_estimators(trials):
    """Bayesian search should explore a wide range, not collapse to one corner."""
    col_candidates = [c for c in trials.columns if c.endswith("n_estimators")]
    assert col_candidates, f"no n_estimators column in {list(trials.columns)}"
    n_est = trials[col_candidates[0]].dropna()
    assert n_est.std() > 100, f"std of n_estimators across trials = {n_est.std()}"


# --- comparison: tuned barely beats default (the methodological finding) ---

def test_tuned_vs_default_rmse_difference_small():
    assert PRED_OPTUNA.exists(), f"missing tuned predictions: {PRED_OPTUNA}"
    assert PRED_DEFAULT.exists(), f"missing default predictions: {PRED_DEFAULT}"
    test_df = pd.read_parquet(TEST_PARQUET)
    y_test = test_df["Global_active_power"].to_numpy()
    p_tuned = np.load(PRED_OPTUNA)
    p_default = np.load(PRED_DEFAULT)
    rmse_tuned = float(np.sqrt(((y_test - p_tuned) ** 2).mean()))
    rmse_default = float(np.sqrt(((y_test - p_default) ** 2).mean()))
    delta = abs(rmse_tuned - rmse_default)
    assert delta < 0.05, (
        f"|tuned RMSE - default RMSE| = {delta:.4f} >= 0.05 "
        f"(tuned={rmse_tuned:.4f}, default={rmse_default:.4f})"
    )


# --- refit correctness: tuned model trained on train+val combined ---

def test_tuned_model_refit_on_train_plus_val(best):
    train = pd.read_parquet(TRAIN_PARQUET)
    val = pd.read_parquet(VAL_PARQUET)
    expected = len(train) + len(val)
    train_size = best["meta"]["train_size"]
    assert train_size == expected, (
        f"meta.train_size = {train_size} != len(train)+len(val) = {expected}"
    )


# --- count of completed trials ---

def test_n_trials_meta_recorded(best):
    n = best["meta"]["n_trials"]
    assert n == 40, f"meta.n_trials = {n}, expected 40"
