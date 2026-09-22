# Smart Home Energy Forecasting REST API

## Description

This service exposes the gradient-boosting regressor trained in the diploma
project on the UCI *Individual Household Electric Power Consumption* (IHEPC)
dataset for the task of one-hour-ahead prediction of household active power
consumption (kW). The model is an XGBoost regressor evaluated chronologically
on the held-out test split (RMSE = 0.4402 kW, R² = 0.6156, MAE = 0.3008 kW).
At inference time the API accepts between 168 and 336 consecutive hourly
readings of the seven raw measurements (Global_active_power, Global_reactive_power,
Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3),
derives the 69-feature vector (56 causal lag features, 6 rolling mean/std
features on the target with `.shift(1)` to prevent leakage, and 7 cyclical
time features) and returns the predicted active power for the hour
immediately after the last submitted reading.

## Quick start

```powershell
# 1. Install dependencies into the active environment
pip install -r requirements_api.txt

# 2. Start the service (working directory: deploy/)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 3. Open the interactive documentation
#    http://localhost:8000/docs   (Swagger UI)
#    http://localhost:8000/redoc  (ReDoc)
```

## Endpoints

| Method | Path                | Tag        | Purpose                                                                  |
| ------ | ------------------- | ---------- | ------------------------------------------------------------------------ |
| GET    | `/health`           | Health     | Liveness probe; reports load state, feature count, uptime in seconds.    |
| GET    | `/model/info`       | Model      | Static model metadata: RMSE, MAE, R², MAPE, training/test periods, etc.  |
| GET    | `/model/features`   | Model      | Top-N (default 20) SHAP feature importances with rank, name, group.      |
| POST   | `/predict`          | Prediction | Returns predicted active power (kW) for the hour after the last reading. |
| GET    | `/predict/example`  | Prediction | Returns a 168-row request body ready to be POSTed back to `/predict`.    |

The full request/response schemas (Pydantic v2) are available in the
auto-generated OpenAPI document at `/openapi.json` and rendered at `/docs`.

## Testing

```powershell
# Full test suite (5 feature tests + 9 API tests)
pytest tests/ -v

# Subset
pytest tests/test_features.py -v
pytest tests/test_api.py -v
```

Coverage includes:

- Structural assertions on the engineered 69-feature vector (count, shape,
  no NaN, column order matches `outputs/splits/test.parquet`).
- A causal-leakage assertion that `rmean_w/rstd_w` use only the prior `w`
  readings and that `lag_1h`/`lag_168h` resolve to the expected source rows.
- HTTP-level checks for `/health`, `/model/info` (RMSE = 0.4402, R² = 0.6156),
  `/model/features` (20 items), `/predict` (success, output range),
  schema rejection of short input (HTTP 422) and non-chronological timestamps
  (HTTP 422), and the round-trip `/predict/example` → `/predict` workflow.

## Demo

```powershell
python demo/run_demo.py
```

`demo/run_demo.py` boots uvicorn on port 8001 in a subprocess, waits for
`/health` to become healthy, POSTs 168 readings drawn from the tail of
`outputs/data/hourly_clean.parquet` to `/predict`, prints the full JSON
response and a one-line summary, persists the response to
`demo/demo_output.json`, and shuts the server down cleanly. `demo/sample_input.json`
is the response of `/predict/example` and serves as appendix evidence in
the thesis.

## End-to-end correctness

For the last test-set timestamp `2010-11-26 20:00:00`, the API's prediction
(when the readings end one hour earlier) is identical to the notebook's
`outputs/predictions/pred_xgb.npy[-1]` (difference = 0.000000 kW). The
engineered feature row matches `outputs/splits/test.parquet` to within
2.3 × 10⁻¹³ on every column, confirming parity of feature engineering
between training and inference.

## Thesis reference

This API constitutes the prototype information system described in
Section 2.12 of the diploma project and satisfies the IT-programme
requirement п.23 of ДП-AITU-19 (running system with UML architecture,
testable endpoints, results matching the original problem statement).
