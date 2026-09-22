# PHASE 04 — Tabular Models (Baseline, Random Forest, XGBoost)
# Input:  outputs/splits/train.parquet, val.parquet, test.parquet
# Output: outputs/predictions/pred_baseline.npy
#         outputs/predictions/pred_rf.npy
#         outputs/predictions/pred_xgb.npy
#         outputs/models/rf_model.joblib
#         outputs/models/xgb_model.json
# Notebook section: "Phase 4: Tabular Models"

---

## WHAT THIS PHASE DOES

Trains the three tabular models (Lag-1 baseline, Random Forest, XGBoost),
evaluates them on the test set, and saves predictions and model files.
The evaluation function is defined here and reused in all later phases.

---

## DOMAIN KNOWLEDGE FOR THIS PHASE

**Model 1 — Lag-1 Persistence Baseline:**

The prediction at time t is simply the observed value at time t-1.
This is already available in the feature matrix as the column `Global_active_power_lag_1h`.
No fitting is required.

Why this baseline matters (from literature):
Moosbrugger et al. (2025) proved that simple persistence baselines beat LSTM,
GRU, and Transformer models when training data is less than ~6 months.
In the v1 results, the lag-1 baseline (RMSE=0.5037) beat SVR (RMSE=0.5614)
and tied LSTM (RMSE=0.5009). Any model that cannot beat this baseline
has learned nothing useful about the data.

Papers that omit the persistence baseline typically overstate their gains.
Pirbazari et al. (2020): "The persistence model is essential for contextualising
claimed improvements in residential forecasting accuracy."

**Model 2 — Random Forest:**

Random Forest is an ensemble of decision trees trained on random subsets of
features and samples. It is invariant to feature scaling (tree-based split
criterion), handles non-linear interactions automatically, and provides
feature importance via mean decrease in impurity.

Why it belongs in this study:
It consistently ranks second behind XGBoost in the literature on tabular energy
forecasting data. Kamre et al. (2025) showed RF+LSTM hybrid beats standalone
models — the standalone RF result here provides the reference for that argument.
Karuna et al. (2024) achieved R²=0.95 with Gradient Boosting on IoT smart home data.
Almughram et al. (2021) identified RF as one of the two most common models in HEMS.

No feature scaling is needed for RF (tree splits are scale-invariant).
Full training set must be used — no subset. The v1 implementation did this correctly.

**Model 3 — XGBoost:**

XGBoost (Chen & Guestrin, 2016) is the dominant tabular ML model in competitive
machine learning since 2016. It uses histogram-based gradient boosting with
regularisation (L1 + L2), subsampling, and column subsampling to prevent overfitting.

Why XGBoost is expected to win on this dataset:
This dataset has ~24,000 training rows and 69 engineered temporal features.
XGBoost excels at exploiting manually engineered lag and rolling features on
tabular data of this scale. Moosbrugger et al. (2025) explain why deep learning
underperforms on single-household data at this scale — the recurrent models
need more data to learn temporal patterns that XGBoost gets "for free"
from the explicit lag features.

In the v1 results: XGBoost achieved RMSE=0.4345, R²=0.618, MAE=0.2964,
clearly outperforming all other models. This result is expected to hold.

**Early stopping for XGBoost:**
XGBoost's early stopping uses the validation set (not the test set) to decide
when to stop adding trees. This prevents overfitting to the training set.
The validation set is used ONLY for this stopping signal — not for metric reporting.
patience = 50 rounds without improvement on validation RMSE.

**The evaluation function — defined once, used everywhere:**

All metrics in this study:
- RMSE (primary): sqrt(mean_squared_error) — penalises large errors, stable on near-zero
- MAE: mean_absolute_error — interpretable in original kW units
- R²: r2_score — coefficient of determination, can be negative (valid, not clipped)
- MAPE (secondary): requires epsilon guard

**MAPE epsilon guard — critical:**
Standard sklearn MAPE = mean(|y-ŷ| / |y|). When y ≈ 0 (overnight, household sleeps
at 0.01-0.05 kW), this fraction diverges.

The epsilon guard: denominator = max(|y_true|, ε) where ε = 0.1 kW.
ε = 0.1 kW is the minimum detectable household load — below this threshold,
the household is essentially off. This threshold comes from Pirbazari et al. (2020).

MAPE is reported but RMSE is the primary ranking metric throughout.
This is consistent with the recommendations of Pirbazari et al. (2020).

**Why SVR is excluded:**
O(N²) space and time complexity. In v1, SVR was trained on an undocumented subset
of the training data to avoid memory exhaustion, making the result neither fair
nor reproducible. The literature abandoned SVR for load forecasting around 2019-2020.
Cite: Parizad & Hatziadoniu (2022) — XGBoost was compared to SVR and found superior.

---

## WHAT TESTS MUST PASS (write these first — TDD)

Tests in `tests/test_phase04.py` AND a test cell in the notebook.
All tests load from disk.

**File existence tests:**
- outputs/predictions/pred_baseline.npy exists and loads as numpy array
- outputs/predictions/pred_rf.npy exists
- outputs/predictions/pred_xgb.npy exists
- outputs/models/rf_model.joblib exists and loads
- outputs/models/xgb_model.json exists and loads

**Prediction shape tests:**
- len(pred_baseline) == len(test set)
- len(pred_rf) == len(test set)
- len(pred_xgb) == len(test set)
- No NaN values in any prediction array
- No infinite values in any prediction array

**Prediction range tests:**
- All predictions are ≥ 0 (negative power is physically impossible)
- All predictions are ≤ 20 kW (maximum plausible household consumption)

**Baseline correctness test:**
- pred_baseline must exactly equal the `Global_active_power_lag_1h` column
  of the test set (within floating-point tolerance)
- This verifies the baseline implementation is correct

**Metric sanity tests (load y_test from test.parquet, compute metrics):**
- XGBoost RMSE is less than lag-1 baseline RMSE (XGBoost must beat the baseline)
- XGBoost RMSE is less than Random Forest RMSE (XGBoost is expected to win)
- XGBoost R² is greater than 0.5 (R² < 0.5 would indicate a serious bug)
- Random Forest R² is greater than 0.4
- All RMSE values are between 0.3 and 0.7 (sanity range from literature)

**Model loading tests:**
- rf_model.predict(X_test) produces the same array as pred_rf.npy
  (verifies model was saved and loaded correctly, not just predictions)
- xgb_model.predict(X_test) produces the same array as pred_xgb.npy

---

## WHAT TO SAVE

Predictions (numpy arrays):
- `outputs/predictions/pred_baseline.npy`
- `outputs/predictions/pred_rf.npy`
- `outputs/predictions/pred_xgb.npy`

Models:
- `outputs/models/rf_model.joblib`
- `outputs/models/xgb_model.json`

Print in notebook:
- Full metrics table for all 3 models on test set
- XGBoost: best iteration (from early stopping), validation RMSE at best iteration
- Random Forest: out-of-bag score if available
- Ranking by RMSE (ascending)
