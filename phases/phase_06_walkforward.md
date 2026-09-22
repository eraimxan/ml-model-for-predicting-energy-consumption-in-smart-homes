# PHASE 06 — Walk-Forward Backtesting
# Input:  outputs/data/features.parquet
#         outputs/splits/ (for row count reference)
# Output: outputs/results/walkforward_xgb.csv
#         outputs/results/walkforward_rf.csv
# Notebook section: "Phase 6: Walk-Forward Backtesting"

---

## WHAT THIS PHASE DOES

Runs 7-fold expanding-window walk-forward cross-validation on XGBoost and Random
Forest. This confirms the single-split test results from Phase 4 are not an artefact
of one lucky test period. Saves per-fold results and summary statistics.

---

## DOMAIN KNOWLEDGE FOR THIS PHASE

**Why walk-forward and not k-fold cross-validation:**

Standard k-fold cross-validation randomly assigns rows to folds. For time series,
this causes leakage: a model trained on fold 5 (2010 data) predicts fold 2 (2008 data),
which it has already "seen" in its training set.

Walk-forward (also called time-series cross-validation or rolling-origin evaluation)
preserves temporal order. In each fold, the model trains on past data and tests
on future data — exactly how a real deployed model would operate.

**Expanding window vs. rolling window:**

Expanding window: fold k trains on ALL data before the test fold.
Rolling window: fold k trains on a fixed-length window before the test fold.

This study uses expanding window because:
- More training data generally improves tree ensemble performance
- The dataset is small enough that discarding older data would hurt
- Moosbrugger et al. (2025) use expanding window for their benchmark

**The 7-fold protocol:**

Split the full feature dataset (after Phase 2 dropna) into 8 contiguous chunks.
Fold 1: train on chunk 0, test on chunk 1
Fold 2: train on chunks 0+1, test on chunk 2
...
Fold 7: train on chunks 0-6, test on chunk 7

Each fold covers approximately 34,000/8 ≈ 4,250 hourly rows ≈ 177 days.

**Why train a fresh model every fold:**
Using the same model object across folds and just calling fit() again does NOT
guarantee a fresh start — some libraries accumulate state. Always instantiate
a new model object for each fold. This was done correctly in v1.

**Early stopping inside walk-forward:**

XGBoost uses early stopping to avoid overfitting. Inside the walk-forward,
the last 15% of the training fold is used as an internal validation set for
the early stopping signal. This ensures consistent training behaviour between
the single-split (Phase 4) and the walk-forward results.

This was a confirmed bug in v1: walk-forward did not use early stopping while
the single-split did. The two training regimes were inconsistent.

**Why apply walk-forward to XGBoost AND Random Forest (not LSTM/GRU):**

Deep learning models take minutes to train per fold. 7 folds × 2 DL models =
14 long training runs. For a diploma, this is computationally expensive and
the insight gained is marginal — the DL results from Phase 7 already represent
a valid single-split evaluation. Tree models train in seconds per fold.

The commission will understand this pragmatic choice.

**What the walk-forward results prove:**

If the 7-fold mean RMSE is close to (within ~1 std dev of) the single-split
RMSE from Phase 4, then the Phase 4 result is temporally stable — it is not
a lucky accident of which test period was chosen.

In v1: single-split RMSE = 0.4345 kW; walk-forward mean = 0.4532 ± 0.0186 kW.
The single-split result was inside one standard deviation of the mean → stable.

**Relationship between phases:**

The walk-forward uses the full feature dataset from Phase 2 (not the Phase 3 splits).
This is because the walk-forward defines its own train/test boundaries independently.
It does not use the Phase 3 scaler — tree models don't need scaling.

---

## WHAT TESTS MUST PASS (write these first — TDD)

Tests in `tests/test_phase06.py` AND a test cell in the notebook.

**File existence tests:**
- outputs/results/walkforward_xgb.csv exists
- outputs/results/walkforward_rf.csv exists

**Structure tests:**
- Each CSV has exactly 7 rows (one per fold)
- Each CSV has columns: fold, RMSE, MAE, R2, MAPE
- No NaN values in either CSV

**Metric sanity tests:**
- All RMSE values across all 14 rows (7 XGB + 7 RF) are between 0.3 and 0.8
- All R² values are between -0.1 and 0.9
- XGBoost mean RMSE across 7 folds is less than Random Forest mean RMSE
  (XGBoost expected to win here too)

**Consistency test — the key validation:**
- Load Phase 4 XGBoost test RMSE (from pred_xgb.npy evaluated against test.parquet)
- Load Phase 6 walk-forward mean RMSE (from walkforward_xgb.csv)
- Assert that |Phase4_RMSE - Phase6_mean_RMSE| < 3 × Phase6_std_RMSE
  (the single-split result should be within 3 standard deviations of the
  walk-forward distribution — a looser check than 1 std dev to be robust)

**Temporal stability test:**
- Assert walk-forward XGB RMSE std < 0.05 kW
  (very high variance across folds would suggest the model is unreliable)

**Fresh model test:**
- This cannot be verified directly from saved files, but the test should
  check that fold 1 and fold 7 XGBoost predictions are genuinely different
  (if they were the same, the model was not retrained)

---

## WHAT TO SAVE

- `outputs/results/walkforward_xgb.csv` — per-fold metrics for XGBoost
- `outputs/results/walkforward_rf.csv` — per-fold metrics for Random Forest

Print in notebook:
- Per-fold RMSE for both models (table)
- Summary statistics (mean ± std) for RMSE, MAE, R², MAPE
- Comparison: single-split (Phase 4) vs walk-forward mean
- Conclusion: "single-split result is [inside/outside] one std dev of walk-forward mean"

The walk-forward RMSE per fold plot is generated in Phase 8 (all plots together).
