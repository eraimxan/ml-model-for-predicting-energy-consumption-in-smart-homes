# PHASE 00 — Project Setup & TDD Harness
# Diploma: ML Model for Predicting Energy Consumption in Smart Homes
# AITU, June 2026 | Students: Raimkhan, Mussepov, Gaisiev

---

## WHAT THIS PHASE PRODUCES

- `diploma_pipeline.ipynb` — the single notebook that all phases write into
- `tests/` — test suite that grows with each phase (TDD: tests written before cells)
- `outputs/` — directory for saved artefacts (CSVs, models, figures)
- `requirements.txt` — pinned dependencies
- `README.md` — how to run the project

The notebook is the primary deliverable. Every phase adds cells to it.
The test suite runs independently and must pass completely before the next phase begins.

---

## PROJECT IDENTITY

This is an AITU bachelor diploma for programme 6B0611 — Computer Science.
The commission grades code on: architecture description, test results, comparison to
known solutions, and visualised outputs. Everything produced must serve those criteria.

---

## ACADEMIC CONTEXT THAT SHAPES ALL DECISIONS

**The task:** 1-hour-ahead forecasting of household electricity consumption (kW)
from a single French household, using only past observations and time features.

**Why this is hard:**
A single household has high variance, near-zero overnight values, and irregular spikes
from appliances. Pirbazari et al. (2020) showed that MAPE is unstable on residential
data because near-zero ground-truth values cause the denominator to blow up.
RMSE is the primary ranking metric throughout this project for that reason.

**What the literature established by 2026:**
- Moosbrugger et al. (2025): deep learning underperforms simple persistence when
  training data is less than ~6 months. Our training set is ~24,000 rows (~2.8 years),
  so DL is on the edge — this is an empirical question we answer by experiment.
- Salman et al. (2026): a two-stage LSTM+GRU→XGBoost hybrid achieved RMSE=0.4817
  on Kaggle smart home data using weather features only.
- Han & Wang (2023): LSTM with dynamic mirror descent achieved RMSE=0.448 on
  Irish CER data with online adaptation.
- Teslyuk et al. (2025): single-layer LSTM with RMSprop achieved MAE=0.072 (97.8%
  accuracy) on a GitHub household dataset.
- Devanathan et al. (2026): no single model wins across all time horizons — algorithm
  selection must match resolution. Hourly = tree ensembles competitive.
- Siphocly PRISMA review (2026): XAI/SHAP is a major identified gap in residential
  STLF. Almost no papers do it. It is a novelty contribution if included.

**What the previous version (diploma_v2.pdf) got wrong that must not recur:**
- LSTM received a 1D raw sequence; XGBoost received 48 engineered features → unfair
- Feature count stated as "48" but actual count was 69 → inconsistency
- SVR trained on undocumented subset → not reproducible → removed entirely
- Walk-forward had no early stopping → inconsistent training regime
- MAPE used sklearn's version without epsilon guard → inflated values on near-zero
- Optuna objective passed eval_set but no early_stopping_rounds → wasteful
- No SHAP, no prediction vs actual plot, no GRU

---

## FIVE MODELS — FINAL DECISION (do not change this)

```
Layer 1 — Naive reference
  Model 1: Lag-1 Persistence Baseline
  Justification: Moosbrugger (2025) — must report it; many papers omit it and overstate gains.

Layer 2 — Classical ensemble ML (tabular, 69 features)
  Model 2: Random Forest
  Model 3: XGBoost + Optuna validation subsection
  Justification: Parizad & Hatziadoniu (2022) — XGBoost 40× faster than GB, superior accuracy.
                 Devanathan et al. (2026) — Optuna+GBM standard on residential STLF.
                 Salman et al. (2026) — XGBoost as stacking layer in best hybrid.

Layer 3 — Deep learning (multivariate sequences, same 69 features)
  Model 4: LSTM (2-layer stacked)
  Model 5: GRU  (2-layer stacked, identical setup to LSTM)
  Justification: Han & Wang (2023), Teslyuk (2025) — LSTM state of art on residential.
                 Salman (2026) — GRU competitive with LSTM; almost always paired.
                 Commission will ask "why not GRU?" — including it prevents the question.

Bonus — Interpretability
  SHAP analysis on XGBoost
  Justification: Siphocly (2026) identifies XAI as the #1 research gap in residential STLF.
                 TreeExplainer makes this cheap (seconds, not hours).
```

**SVR excluded:** O(N²) complexity forced an undocumented subset in v1; abandoned in
literature ~2019. Cite Parizad & Hatziadoniu (2022) in the thesis.

**SARIMA excluded:** Rolling one-step-ahead SARIMA with seasonal order 24 at hourly
resolution is computationally prohibitive. Cite Hyndman & Athanasopoulos (2021) in
thesis. Lag-1 captures the AR(1) component that makes SARIMA competitive.

**Optuna not a separate model row:** The finding (Δ RMSE ≈ 0.0003 kW ≈ 0.07%) is
a methodological result — "default XGBoost is near-optimal for this feature space."
It is presented as Section 2.9 of the thesis, not as a competing model.

---

## DATA THAT ALL PHASES USE

**File:** `household_power_consumption.txt`
**Source:** UCI Individual Household Electric Power Consumption (Hébrail & Berard, 2012)
**Separator:** semicolon
**Rows:** 2,075,259 at 1-minute resolution
**Period:** 16 December 2006 – 26 November 2010
**Location:** Single household near Paris, France

**Columns:**
- Date (dd/mm/yyyy) + Time (hh:mm:ss) → combine into DatetimeIndex
- Global_active_power [kW] ← TARGET variable for all models
- Global_reactive_power [kVAR]
- Voltage [V]
- Global_intensity [A]
- Sub_metering_1 [Wh] — kitchen appliances
- Sub_metering_2 [Wh] — laundry room
- Sub_metering_3 [Wh] — climate control (water heater + AC)

**Missing values:** literal string "?" — approximately 1.25% of rows, in short clusters.

---

## TDD GROUND RULES FOR ALL PHASES

1. Write the test cell first. Run it. Watch it fail. Then write the implementation cell.
2. Every test uses `assert` statements with descriptive failure messages.
3. Tests must be self-contained — they load their inputs from saved files, not from
   in-memory variables, so they can be re-run independently at any time.
4. A phase is complete only when ALL its tests pass with zero errors.
5. Do not proceed to the next phase if any test in the current phase fails.
6. Tests are written as a separate `tests/test_phaseNN.py` file AND as notebook cells.

---

## NOTEBOOK STRUCTURE

The notebook `diploma_pipeline.ipynb` is organised into sections matching the phases.
Each phase appends cells to the same notebook. By the end of Phase 8 the complete
notebook is the diploma appendix code listing.

Section headers in the notebook use Markdown cells:
```
# Phase 1: Data Loading and Cleaning
# Phase 2: Feature Engineering
# Phase 3: Chronological Split
# Phase 4: Tabular Models
# Phase 5: Optuna Hyperparameter Optimisation
# Phase 6: Walk-Forward Backtesting
# Phase 7: Deep Learning (LSTM & GRU)
# Phase 8: SHAP, Plots, and Final Results
```

---

## REPRODUCIBILITY REQUIREMENTS

Random seed 42 must be set for: Python's `random`, `numpy`, `tensorflow`,
`optuna` (via sampler seed), `sklearn` (random_state parameter).
Set these at the very first code cell of the notebook and in every test file.

Library versions must be printed in the first notebook cell.
Required libraries: pandas, numpy, scikit-learn, xgboost, tensorflow,
optuna, shap, matplotlib, seaborn, scipy.

---

## OUTPUTS DIRECTORY STRUCTURE

```
diploma_pipeline/
├── diploma_pipeline.ipynb        ← primary deliverable
├── tests/
│   ├── test_phase01.py
│   ├── test_phase02.py
│   ├── test_phase03.py
│   ├── test_phase04.py
│   ├── test_phase05.py
│   ├── test_phase06.py
│   ├── test_phase07.py
│   └── test_phase08.py
├── outputs/
│   ├── data/
│   │   ├── hourly_clean.parquet      ← Phase 1 output
│   │   └── features.parquet          ← Phase 2 output
│   ├── splits/
│   │   ├── train.parquet             ← Phase 3 output
│   │   ├── val.parquet
│   │   └── test.parquet
│   ├── predictions/
│   │   ├── pred_baseline.npy         ← Phase 4 output
│   │   ├── pred_rf.npy
│   │   ├── pred_xgb.npy
│   │   ├── pred_lstm.npy             ← Phase 7 output
│   │   └── pred_gru.npy
│   ├── models/
│   │   ├── rf_model.joblib           ← Phase 4 output
│   │   ├── xgb_model.json
│   │   ├── scaler.joblib             ← Phase 3 output
│   │   ├── lstm_model.keras          ← Phase 7 output
│   │   └── gru_model.keras
│   ├── results/
│   │   ├── walkforward_xgb.csv       ← Phase 6 output
│   │   ├── walkforward_rf.csv
│   │   ├── optuna_best_params.json   ← Phase 5 output
│   │   └── final_results.csv         ← Phase 8 output
│   └── figures/
│       ├── 01_prediction_vs_actual.png
│       ├── 02_shap_summary.png
│       ├── 02b_shap_bar.png
│       ├── 03_model_comparison.png
│       ├── 04_residuals_over_time.png
│       ├── 05_residual_distribution.png
│       ├── 06_walkforward_rmse.png
│       ├── 07_scatter_xgb.png
│       ├── 08_training_history.png
│       └── 09_eda_patterns.png
├── requirements.txt
└── README.md
```
