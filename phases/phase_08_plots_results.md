# PHASE 08 — SHAP Analysis, All Plots, and Final Results Table
# Input:  all prediction .npy files from phases 4 and 7
#         all split parquets from phase 3
#         walkforward CSVs from phase 6
#         training history JSONs from phase 7
#         xgb_model.json from phase 4
# Output: outputs/figures/*.png (9 figures)
#         outputs/results/final_results.csv
# Notebook section: "Phase 8: SHAP, Plots, and Final Results"

---

## WHAT THIS PHASE DOES

Produces all visualisations, the SHAP interpretability analysis, and the final
comparative results table. Nothing new is trained here — all predictions are
loaded from disk. This phase can be re-run without retraining anything.

---

## DOMAIN KNOWLEDGE FOR THIS PHASE

**SHAP — the novelty contribution:**

SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017) is a game-theoretic
framework for explaining model predictions. For any prediction, SHAP decomposes
it into additive contributions from each feature: the prediction equals the
model's expected output plus the sum of SHAP values across all features.

Why this is a novelty contribution for this diploma:
Siphocly (2026) PRISMA review of 86 residential STLF papers identified XAI/SHAP
as the single most prominent research gap — almost no papers include it.
Including SHAP analysis makes this diploma more thorough than most published papers.

**TreeExplainer (not KernelExplainer):**

For tree-based models (XGBoost, Random Forest), shap.TreeExplainer computes
exact SHAP values using the tree structure directly — no approximation needed.
This runs in seconds to minutes.

KernelExplainer is model-agnostic and works for any model, but it requires
thousands of model evaluations per sample and can take hours.
Always use TreeExplainer for XGBoost.

**What SHAP reveals for this study:**

The SHAP summary plot will show which of the 69 features most influence predictions.
Expected findings (from literature and feature engineering logic):
- Global_active_power_lag_1h (most recent observation) should be the top feature
- Global_active_power_lag_24h (yesterday same hour) should be second
- Global_active_power_lag_168h (last week same hour) should be high
- Rolling means (rmean_24h) should be significant
- Time features (hour_sin, hour_cos) should be significant
These findings directly support the feature engineering decisions in Phase 2
and give the commission something concrete to discuss.

**Sample size for SHAP:**

Computing SHAP values for all test rows is slow (~5,100 rows × 69 features).
A random sample of 500-1000 rows gives stable SHAP estimates.
Sort the sample indices to maintain chronological order.

**The evaluate function must handle alignment:**

DL predictions cover len(test) - SEQ_LEN rows (test rows SEQ_LEN onwards).
Tabular predictions cover all len(test) rows.
For the final comparison table, use the OVERLAPPING portion:
  - Tabular: test rows from index SEQ_LEN onwards
  - DL: all DL predictions
This ensures all models are compared on exactly the same rows.
Document this clearly in the notebook.

**Figure descriptions:**

Figure 1 — Prediction vs Actual (two representative weeks)
Purpose: The most intuitive result visualisation. The commission will look at this
first. It shows whether the model tracks spikes and troughs correctly.
Shows: actual (black), XGBoost (blue), Random Forest (orange), LSTM (green), GRU (purple).
Pick weeks that show interesting behaviour — one "typical" week, one "high variance" week.

Literature precedent: Han & Wang (2023), Teslyuk (2025), Salman (2026) all show
time-series plots of predictions vs actual as their primary result figure.

Figure 2 — SHAP Summary (beeswarm)
Purpose: Shows feature importance AND direction of effect.
Each dot = one sample. x-axis = SHAP value. Colour = feature value (red=high, blue=low).
This reveals: "high last-hour consumption (red dot, right side) → higher prediction"

Figure 2b — SHAP Bar (mean absolute SHAP)
Purpose: Simpler version for the thesis. Clean horizontal bar chart of top 20 features.
Colour-coded by feature type (lag=blue, rolling=orange, time=green).

Figure 3 — Model Comparison Bar Chart
Purpose: Summary of all 5 models in one figure. Sorted by RMSE ascending.
Dual axis: RMSE bars (left) + R² line (right).
This is the primary comparison figure for the thesis — equivalent to Table 2.1 but visual.

Figure 4 — Residuals Over Time (XGBoost)
Purpose: Reveals WHEN the model fails. Clustering of errors at certain times (peaks,
weekends) indicates systematic patterns the model cannot capture.
Adds a 24h rolling mean of residuals to show systematic bias.
Pirbazari et al. (2020) discuss this type of residual analysis for residential STLF.

Figure 5 — Residual Distribution (XGBoost vs LSTM)
Purpose: Compares error distributions. Ideal: symmetric, centred at zero, thin tails.
Heavy tails indicate the model struggles with extreme consumption events.
Overlays a normal distribution fit to show deviation from normality.

Figure 6 — Walk-Forward RMSE per Fold
Purpose: Shows temporal stability of XGBoost and RF results.
If RMSE is consistent across all 7 folds, the models are reliable.
If one fold is dramatically worse, that fold likely covers an unusual period.

Figure 7 — Predicted vs Actual Scatter (XGBoost)
Purpose: Shows bias and heteroscedasticity. Points above the diagonal = overprediction.
Hexbin density plot handles the 5,000+ test points without overplotting.
R² shown in legend.

Figure 8 — Training History (LSTM and GRU)
Purpose: Shows the learning dynamics. The epoch where early stopping triggered
(vertical dashed line) shows whether training converged smoothly or was cut short.
Divergence between train and val loss indicates overfitting.

Figure 9 — EDA Patterns (generated in Phase 1 but consolidated here)
Purpose: Diurnal and weekly seasonal patterns in the data. Motivates the feature
engineering choices and contextualises the difficulty of the task.

**Figure quality requirements:**
- dpi=150 minimum (readable in printed thesis)
- All axes labelled with units
- All figures have titles
- All multi-series figures have legends
- tight_layout() or constrained_layout applied
- Saved as .png to outputs/figures/

---

## WHAT TESTS MUST PASS (write these first — TDD)

Tests in `tests/test_phase08.py` AND a test cell in the notebook.

**Figure file tests:**
- All 9 figures exist in outputs/figures/
- All files are > 50 KB (too small = probably blank or corrupted)
- All files are valid PNG (check magic bytes)

**Results table tests:**
- outputs/results/final_results.csv exists
- Has exactly 5 rows (one per model)
- Has columns: Model, RMSE, MAE, R2, MAPE
- All RMSE values are positive
- All R2 values are less than 1.0
- XGBoost has the lowest RMSE (primary metric)
- The row with lowest RMSE has R2 > 0.5
- The comparison is fair: all metrics computed on the same rows of the test set

**SHAP tests:**
- SHAP values were computed (check that Figure 2 is non-blank)
- The top-ranked feature by mean absolute SHAP is a lag feature of Global_active_power
  (not a covariate or time feature — this is the expected finding)

**Ranking consistency test:**
- Load final_results.csv
- Load walkforward_xgb.csv
- Assert walk-forward mean RMSE is within 3 × std of single-split XGBoost RMSE
  (the two evaluation protocols should agree)

---

## WHAT TO SAVE

Figures:
- `outputs/figures/01_prediction_vs_actual.png`
- `outputs/figures/02_shap_summary.png`
- `outputs/figures/02b_shap_bar.png`
- `outputs/figures/03_model_comparison.png`
- `outputs/figures/04_residuals_over_time.png`
- `outputs/figures/05_residual_distribution.png`
- `outputs/figures/06_walkforward_rmse.png`
- `outputs/figures/07_scatter_xgb.png`
- `outputs/figures/08_training_history.png`
- `outputs/figures/09_eda_patterns.png`

Results:
- `outputs/results/final_results.csv`

Print in notebook — the full final summary:

```
====================================================================
FINAL MODEL COMPARISON — TEST SET
====================================================================
Model                          RMSE    MAE     R²      MAPE
--------------------------------------------------------------------
Lag-1 Persistence              X.XXXX  X.XXXX  X.XXXX  XX.XX%
Random Forest                  X.XXXX  X.XXXX  X.XXXX  XX.XX%
LSTM (multivariate, 2-layer)   X.XXXX  X.XXXX  X.XXXX  XX.XX%
GRU  (multivariate, 2-layer)   X.XXXX  X.XXXX  X.XXXX  XX.XX%
XGBoost (default)              X.XXXX  X.XXXX  X.XXXX  XX.XX%
--------------------------------------------------------------------
Primary metric: RMSE (lower is better)
MAPE computed with epsilon=0.1 kW guard (Pirbazari et al. 2020)
All models evaluated on identical test rows (aligned for DL offset)
====================================================================

Walk-Forward Backtesting (XGBoost, 7 folds):
  Mean RMSE: X.XXXX ± X.XXXX kW
  Mean R²:   X.XXXX ± X.XXXX
  Single-split RMSE: X.XXXX kW (within X std devs of walk-forward mean)

Optuna Validation (reported in Section 2.9, not a competing model):
  Default test RMSE:  X.XXXX kW
  Tuned test RMSE:    X.XXXX kW
  Δ RMSE:             X.XXXX kW (interpretation: near-optimal / meaningful)

SHAP Top-5 features (XGBoost):
  1. [feature_name] — mean|SHAP| = X.XXXX
  2. [feature_name] — mean|SHAP| = X.XXXX
  3. [feature_name] — mean|SHAP| = X.XXXX
  4. [feature_name] — mean|SHAP| = X.XXXX
  5. [feature_name] — mean|SHAP| = X.XXXX
====================================================================
```
