# Smart-Home Hourly Energy-Consumption Forecasting — Project Overview

> Diploma project, **6B0611 Computer Science**, Astana IT University.
> A machine-learning pipeline and a deployable REST API that predict
> household active power consumption one hour ahead from the seven raw
> sensor channels of the UCI *Individual Household Electric Power
> Consumption* (IHEPC) dataset.

---

## 1. Problem statement

Smart-home energy management systems require short-horizon forecasts of
electrical consumption to schedule loads (e.g. heating, dishwasher,
EV-charging) and to support time-of-use tariff optimisation. The
diploma project addresses the following supervised task:

> Given the last 168 hourly readings of seven raw electrical
> measurements taken inside a single household, predict the
> *Global active power* (kW) for the next hour.

The 168-hour history corresponds to a full weekly cycle and therefore
captures both daily and weekly seasonality.

## 2. Dataset

| Property               | Value                                                              |
| ---------------------- | ------------------------------------------------------------------ |
| Source                 | UCI ML Repository — *Individual Household Electric Power Consumption* |
| Original cadence       | 1 minute, 4 years (Dec 2006 → Nov 2010), one French household      |
| Cleaned cadence        | 1 hour (mean of minute samples; flagged measurements removed)      |
| Cleaned hourly rows    | 34,157                                                             |
| Raw channels (7)       | Global_active_power, Global_reactive_power, Voltage, Global_intensity, Sub_metering_1/2/3 |

The cleaned hourly series is the canonical input from Phase 2 onward.

## 3. Feature engineering

A 69-feature representation is derived per hour with strict
no-leakage guarantees (every feature uses `.shift(≥1)`):

| Group              | Count | Description                                                                                  |
| ------------------ | :---: | -------------------------------------------------------------------------------------------- |
| Lag features       |  56   | 7 variables × 8 horizons (1, 2, 3, 6, 12, 24, 48, 168 hours)                                  |
| Rolling features   |   6   | Mean and std of the target over 3-, 6-, 24-hour windows, all `.shift(1).rolling(w)`           |
| Cyclical time      |   7   | sin/cos of hour-of-day, day-of-week, month-of-year, plus a binary `is_weekend` flag           |

Causality is asserted in `tests/test_features.py::test_rolling_features_use_shift`:
the rolling mean at row *t* equals the mean of rows *[t−w, t)* and never
includes row *t* itself.

## 4. Experimental protocol

A strictly chronological 70 / 15 / 15 split with no shuffling is used to
respect the temporal nature of the data.

| Split | Rows   | Period (UTC)              | Share |
| ----- | -----: | ------------------------- | :---: |
| Train | 23,792 | 2006-12-23 → 2009-09-15   | 70.0% |
| Val   |  5,098 | 2009-09-15 → 2010-04-19   | 15.0% |
| Test  |  5,099 | 2010-04-19 → 2010-11-26   | 15.0% |

The scaler (`StandardScaler`) is fit on **train only** and then frozen;
val and test are merely transformed. Tree-based models (Random Forest,
XGBoost) are scale-invariant and operate on raw features; the sequence
models (LSTM, GRU) consume the scaled matrix.

## 5. Models compared

Five models are benchmarked on the same held-out test split.

| Model                          | RMSE (kW) | MAE (kW) | R²    | MAPE (%) |
| ------------------------------ | :-------: | :------: | :---: | :------: |
| Lag-1 Persistence (baseline)   |  0.5852   |  0.3868  | 0.321 |  46.43   |
| Random Forest                  |  0.4581   |  0.3156  | 0.584 |  42.74   |
| **XGBoost (default)**          | **0.4402**| **0.3008**| **0.6156** | **39.36** |
| LSTM (multivariate, 2-layer)   |  0.5586   |  0.4004  | 0.381 |  53.67   |
| GRU  (multivariate, 2-layer)   |  0.5627   |  0.3987  | 0.372 |  52.17   |

**XGBoost is the production model.** It beats the persistence baseline by
24.8 % on RMSE and outperforms every neural alternative — a finding that
is consistent with the wider literature on short-horizon tabular
forecasting.

## 6. Hyperparameter search and tuning

A 40-trial **Optuna TPE** search over a nine-dimensional XGBoost
configuration space was executed with early stopping on the validation
split (`early_stopping_rounds=50`). The tuned model achieved a marginal
improvement that did **not** justify the added complexity for the
diploma submission, so the simpler default-config XGBoost was kept as
the final model. The search trace (`outputs/results/optuna_study_trials.csv`)
and the best-configuration JSON are retained for reproducibility.

## 7. Robustness analyses

Three independent checks underpin the headline numbers.

1. **7-fold expanding-window walk-forward validation.** The full
   feature matrix is divided into 8 contiguous chunks. For each fold
   k ∈ {1..7} the model is refitted on chunks [0..k) and evaluated on
   chunk k. RMSE remains in a narrow [0.42, 0.49] kW band — confirming
   that the single-split RMSE = 0.4402 kW is not an artefact of one
   lucky temporal cut.

2. **Feature-group ablation study.** Removing each of the three feature
   groups (lag / rolling / time) in turn quantifies their contribution:
   lag features dominate, time features are non-negligible
   (≈ 0.02 kW RMSE), rolling features contribute the least but are
   retained because they consistently help on noisy days.

3. **Inline pipeline assertions.** The notebook embeds structural and
   leakage assertions throughout Phases 2–8 (column count = 70 after
   feature engineering, `lag_1h` ≡ previous-row target, scaler fit on
   train only, walk-forward never trains on future folds, etc.). The
   assertions are mirrored in `tests/` and re-executed by pytest in CI.

## 8. Interpretability — SHAP

A SHAP TreeExplainer is run on 500 random test rows. The top features
by mean |SHAP| are:

| Rank | Feature                          | mean \|SHAP\| |
| :--: | -------------------------------- | :-----------: |
|  1   | Global_intensity_lag_1h          |    0.348      |
|  2   | Global_active_power_lag_1h       |    0.116      |
|  3   | Sub_metering_3_lag_1h            |    0.080      |
|  4   | hour_cos                         |    0.057      |
|  5   | hour_sin                         |    0.039      |

The dominance of `Global_intensity_lag_1h` (the previous hour's amperage)
is intuitively correct — instantaneous current is the single best
predictor of next-hour power — and serves as a domain-knowledge sanity
check on the model.

## 9. Information system (REST API)

The trained model is exposed as a FastAPI service so that the
forecasting capability is testable end-to-end and demonstrably running,
as required by п.23 of ДП-AITU-19.

| Method | Path                | Purpose                                                        |
| ------ | ------------------- | -------------------------------------------------------------- |
| GET    | `/health`           | Liveness probe — load state, feature count, uptime.            |
| GET    | `/model/info`       | Model card — RMSE, MAE, R², MAPE, training/test periods.       |
| GET    | `/model/features`   | Top-20 SHAP feature importances with rank, name, group.        |
| POST   | `/predict`          | Returns predicted kW for the hour after the last reading.      |
| GET    | `/predict/example`  | Returns a 168-row request body ready to be POSTed back.        |

**Architectural properties.**

- Python 3.10, FastAPI 0.115, Pydantic v2, XGBoost 2.1, pandas 2.2.
- Inputs are validated by Pydantic schemas (168 ≤ N ≤ 336 readings,
  strictly ascending timestamps).
- The 69-feature vector is recomputed at request time using the same
  feature-engineering module that produced the training matrix, so the
  training/serving feature contract cannot drift.
- The XGBoost model and `StandardScaler` are loaded once at startup via
  the FastAPI `lifespan` hook and reused across requests by a
  module-level singleton.
- Cross-origin requests are permitted, and every request is logged with
  method, path, status, and latency.
- Interactive Swagger UI is exposed at `/docs`, ReDoc at `/redoc`, the
  raw OpenAPI document at `/openapi.json`.

**Verified parity with the training pipeline.** For the last
held-out test timestamp `2010-11-26 20:00`, the API reproduces the
notebook's `pred_xgb[-1]` to **0.000000 kW**, and the engineered
feature row matches `outputs/splits/test.parquet` to within
2.3 × 10⁻¹³ on every column — guaranteeing that no implementation
drift has crept in between research code and the deployed service.

## 10. Quality assurance

| Layer                      | Mechanism                                                                      |
| -------------------------- | ------------------------------------------------------------------------------ |
| Notebook                   | Inline assertions in every phase (shape, leakage, scaler-state, etc.)          |
| Repository                 | `tests/` directory mirroring those assertions for CI execution                 |
| API                        | 14 pytest cases (5 feature-engineering + 9 HTTP) — all green                   |
| Demo                       | `demo/run_demo.py` boots the server, sends a real request, persists the output |
| Static evidence            | `demo/demo_output.json`, `demo/sample_input.json`, `pytest_full_report.txt`    |

## 11. Mapping to п.23 ДП-AITU-19

| Requirement                            | Evidence                                                                                              |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Running information system             | `python -m uvicorn api.main:app` → http://localhost:8000/docs                                         |
| UML architectural model                | `component_diagram.png`, `deployment_diagram.png`, `UML_activity diagram.png`, `er_schema.png`         |
| Code testing and test runs             | `pytest tests/ -v` (14/14 passed), `demo/demo_output.json`                                            |
| Results match problem statement        | `GET /model/info` returns RMSE = 0.4402 kW, R² = 0.6156 — identical to `final_results.csv`             |
| Formal verification                    | Ablation study, 7-fold walk-forward validation, pipeline-wide assertions                              |

## 12. Limitations

1. **Single-household model.** The model is trained on one French
   household. Transfer to a different home or country would require
   retraining; the architecture (features, pipeline, API contract)
   transfers without change.
2. **Hourly horizon only.** Sub-hourly forecasts and multi-step-ahead
   forecasts (e.g. 24 hours ahead) are out of scope; both would require
   re-engineering the feature window and the loss function.
3. **No exogenous regressors.** Weather (outside temperature is the
   strongest known driver of household consumption), holidays, and
   tariff signals are not used. Incorporating them is the most
   promising direction for future work.
4. **Static model file.** The deployed model is a frozen artefact;
   continuous learning, drift detection, and online retraining are
   left for future engineering.

## 13. Contributions

- A reproducible, fully tested pipeline that takes the raw UCI IHEPC
  dump and produces a benchmark-quality XGBoost forecaster
  (RMSE = 0.4402 kW, 24.8 % below the persistence baseline).
- An empirical demonstration that, on a small tabular smart-home
  series, gradient boosting outperforms two-layer recurrent networks
  — corroborating recent findings in the time-series ML literature.
- A SHAP-based interpretability layer that exposes the model's top
  drivers in academically defensible form.
- A deployable REST API that constitutes the prototype information
  system required by п.23 ДП-AITU-19 and demonstrably reproduces the
  notebook predictions to machine precision.

## 14. Talking points for the defense

> *(Suggested 5-minute structure for the presentation.)*

1. **Why this matters** — smart homes need short-horizon forecasts to
   manage tariffs and storage; this work prototypes the forecasting
   component.
2. **What I built** — an end-to-end pipeline (data → features → model)
   plus a FastAPI service that exposes the model as a real, callable
   information system.
3. **How it performs** — XGBoost beats persistence by 24.8 % on RMSE
   and beats LSTM/GRU/Random Forest on the same chronological test
   split.
4. **Why I trust it** — chronological split, walk-forward validation,
   SHAP interpretability, an ablation study, and inline pipeline
   assertions.
5. **What is reproducible** — the API reproduces the notebook's
   prediction for the last test row to 0.000000 kW, and 14/14 pytest
   cases pass.
6. **What is next** — adding weather and tariff signals, extending the
   horizon, deploying to a real smart-home gateway.
