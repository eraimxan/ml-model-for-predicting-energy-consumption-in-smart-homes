from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.features import FEATURE_NAMES, build_features
from api.model import model_manager
from api.schemas import (
    FeatureImportanceItem,
    FeatureImportanceResponse,
    HealthResponse,
    HourlyReading,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent
MODEL_PATH = DATA_ROOT / "outputs" / "models" / "xgb_model.json"
SCALER_PATH = DATA_ROOT / "outputs" / "models" / "scaler.joblib"
TEST_PARQUET = DATA_ROOT / "outputs" / "splits" / "test.parquet"
HOURLY_PARQUET = DATA_ROOT / "outputs" / "data" / "hourly_clean.parquet"
STATIC_DIR = PROJECT_ROOT / "static"
INDEX_HTML = STATIC_DIR / "index.html"

API_DESCRIPTION = (
    "REST API for short-horizon (1-hour-ahead) prediction of household active "
    "power consumption (kW). The service exposes the XGBoost regressor trained "
    "on the UCI Individual Household Electric Power Consumption (IHEPC) dataset "
    "(RMSE = 0.4402 kW, R² = 0.6156 on the held-out test split). Inputs are "
    "168 to 336 consecutive hourly readings of seven raw measurements; the service "
    "derives the 69-feature vector (56 lag, 6 rolling, 7 cyclical time) used at "
    "training time and returns the prediction for the hour immediately after the "
    "last submitted reading. This API constitutes the prototype information system "
    "referenced in Section 2.12 of the diploma project (п.23 ДП-AITU-19)."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smart-home-energy-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.load(str(MODEL_PATH), str(SCALER_PATH))
    log.info("Model and scaler loaded successfully")
    yield


app = FastAPI(
    title="Smart Home Energy Forecasting API",
    description=API_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _request_logging(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - t0) * 1000.0
    log.info(
        "%s %s -> %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        scaler_loaded=model_manager._scaler is not None,
        n_features=model_manager.n_features,
        uptime_seconds=model_manager.uptime_seconds,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info() -> ModelInfoResponse:
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    info = model_manager.get_model_info()
    info["rmse_kw"] = round(info["rmse_kw"], 4)
    info["r2"] = round(info["r2"], 4)
    return ModelInfoResponse(**info)


@app.get("/model/features", response_model=FeatureImportanceResponse, tags=["Model"])
def model_features(top_n: int = 20) -> FeatureImportanceResponse:
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    items = [FeatureImportanceItem(**x) for x in model_manager.get_shap_features(top_n=top_n)]
    return FeatureImportanceResponse(features=items, total=len(items))


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        X = build_features(req.readings)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    nan_mask = pd.isna(X[0])
    if nan_mask.any():
        missing = [FEATURE_NAMES[i] for i, m in enumerate(nan_mask) if m]
        raise HTTPException(
            status_code=422,
            detail=f"feature vector contains NaN at: {missing[:5]}",
        )

    pred_kw = model_manager.predict(X)
    last_ts = req.readings[-1].timestamp
    ts_predicted = last_ts + pd.Timedelta(hours=1)

    return PredictResponse(
        predicted_kw=round(pred_kw, 4),
        timestamp_predicted=ts_predicted,
        warning=None,
    )


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(str(INDEX_HTML))


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/predict/example", tags=["Prediction"])
def predict_example() -> dict:
    hourly = pd.read_parquet(HOURLY_PARQUET)
    window = hourly.iloc[-168:]
    readings = []
    for ts, row in window.iterrows():
        readings.append(
            HourlyReading(
                timestamp=ts,
                Global_active_power=float(row["Global_active_power"]),
                Global_reactive_power=float(row["Global_reactive_power"]),
                Voltage=float(row["Voltage"]),
                Global_intensity=float(row["Global_intensity"]),
                Sub_metering_1=float(row["Sub_metering_1"]),
                Sub_metering_2=float(row["Sub_metering_2"]),
                Sub_metering_3=float(row["Sub_metering_3"]),
            ).model_dump(mode="json")
        )
    return {"readings": readings}
