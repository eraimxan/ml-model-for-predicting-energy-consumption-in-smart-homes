# PHASE 03 — Chronological Split and Scaler
# Input:  outputs/data/features.parquet
# Output: outputs/splits/train.parquet, val.parquet, test.parquet
#         outputs/models/scaler.joblib
# Notebook section: "Phase 3: Chronological Split"

---

## WHAT THIS PHASE DOES

Splits the feature matrix into train/validation/test sets in strict chronological
order. Fits a StandardScaler on the training set only and saves it.
No models. No predictions. No plots.

---

## DOMAIN KNOWLEDGE FOR THIS PHASE

**Why strictly chronological — no shuffling, ever:**
In time series, shuffling creates data leakage. If you train on data from 2010
and test on data from 2007, the model has implicitly "seen the future" of the
test period through the 2010 training rows. This inflates accuracy dramatically.

This is the most common methodological flaw in residential STLF papers.
Mehdipour Pirbazari et al. (2020) specifically call this out: "many studies
employ random cross-validation which introduces temporal leakage."
Moosbrugger et al. (2025) make the same point and use strictly chronological splits.

The diploma_v2.pdf got this right — preserve it.

**Why 70 / 15 / 15 and not 80 / 20:**
The 15% validation set serves two purposes:
(a) early stopping signal for XGBoost and the deep learning models
(b) Optuna objective evaluation in Phase 5
Having 15% (rather than 10%) gives more reliable early stopping signals.
Salman et al. (2026) and Han & Wang (2023) both use similar 70/15/15 or 80/10/10
splits. The exact ratio is less important than the absence of shuffling.

**Why the validation set is never mixed into test evaluation:**
The validation set is used ONLY for:
- Early stopping (XGBoost, LSTM, GRU)
- Optuna hyperparameter search objective
It is NEVER used to compute the reported test metrics. The test set is held out
completely until each model's final evaluation.

**What the scaler is and why it is fit here:**
StandardScaler transforms features to zero mean and unit variance.
It is needed for LSTM and GRU (neural networks are sensitive to feature scale).
Tree-based models (RF, XGBoost) do not need scaling — they are invariant to
monotonic feature transformations.

**Critical rule about the scaler:**
The scaler must be fit ONLY on X_train features. It must NOT see any validation
or test feature values during fitting. This prevents the subtle "test set
statistics leakage" where the scaler uses information about future data.

The scaler is saved to disk so Phase 7 (deep learning) can load it without
re-fitting, ensuring consistency.

**What splits are produced:**

train: first 70% of rows chronologically
val:   next  15% of rows chronologically
test:  final 15% of rows chronologically

Each split file contains ALL 70 columns (69 features + TARGET).
The split is on the feature DataFrame from Phase 2, not on the raw hourly data.

**Expected approximate sizes:**
train: ~24,000 rows (2006–2009 approximately)
val:   ~5,100 rows  (early 2009–mid 2009 approximately)
test:  ~5,100 rows  (mid 2009–2010 approximately)

---

## WHAT TESTS MUST PASS (write these first — TDD)

Tests in `tests/test_phase03.py` AND a test cell in the notebook.
All tests load files from disk — no in-memory variables.

**File existence tests:**
- outputs/splits/train.parquet exists
- outputs/splits/val.parquet exists
- outputs/splits/test.parquet exists
- outputs/models/scaler.joblib exists

**No-overlap tests (most critical):**
- train.index.max() is strictly less than val.index.min()
- val.index.max() is strictly less than test.index.min()
- No timestamp appears in more than one split

**Completeness test:**
- len(train) + len(val) + len(test) equals len(features) — no rows lost

**Size proportion tests:**
- len(train) / total is between 0.68 and 0.72
- len(val) / total is between 0.13 and 0.17
- len(test) / total is between 0.13 and 0.17

**Chronological order tests:**
- train.index is monotonically increasing
- val.index is monotonically increasing
- test.index is monotonically increasing

**Scaler tests:**
- Scaler loads from joblib without error
- Scaler has n_features_in_ == 69
- When applied to X_train features, the result has mean ≈ 0 and std ≈ 1
  for each feature (within tolerance of 0.01)
- When applied to X_val features (which the scaler has NOT seen during fitting),
  the result does NOT necessarily have mean=0 — assert this is NOT enforced,
  which confirms the scaler was fit on train only

**Column tests:**
- TARGET column is present in all three splits
- All 69 feature columns are present in all three splits

---

## WHAT TO SAVE

- `outputs/splits/train.parquet`
- `outputs/splits/val.parquet`
- `outputs/splits/test.parquet`
- `outputs/models/scaler.joblib`

Print in notebook:
- Date range and row count for each split
- Proportions (% of total)
- Confirmation that no overlap exists
- Scaler: feature count, mean and std of first 5 features after scaling
