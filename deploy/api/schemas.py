from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class HourlyReading(BaseModel):
    timestamp: datetime
    Global_active_power: float = Field(ge=0)
    Global_reactive_power: float
    Voltage: float
    Global_intensity: float = Field(ge=0)
    Sub_metering_1: float = Field(ge=0)
    Sub_metering_2: float = Field(ge=0)
    Sub_metering_3: float = Field(ge=0)


class PredictRequest(BaseModel):
    readings: List[HourlyReading] = Field(min_length=168, max_length=336)

    @model_validator(mode="after")
    def _timestamps_strictly_ascending(self) -> "PredictRequest":
        ts = [r.timestamp for r in self.readings]
        for i in range(1, len(ts)):
            if not (ts[i] > ts[i - 1]):
                raise ValueError(
                    f"timestamps must be strictly ascending; "
                    f"violation at index {i}: {ts[i - 1]!s} -> {ts[i]!s}"
                )
        return self


class PredictResponse(BaseModel):
    predicted_kw: float
    timestamp_predicted: datetime
    model: str = "XGBoost"
    rmse_kw: float = 0.4402
    r2: float = 0.6156
    feature_count: int = 69
    warning: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    n_features: int
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    rmse_kw: float
    mae_kw: float
    r2: float
    mape_pct: float
    n_features: int
    n_training_rows: int
    training_period: str
    test_period: str
    top_feature: str
    top_feature_shap: float


class FeatureImportanceItem(BaseModel):
    rank: int
    name: str
    mean_abs_shap: float
    group: str


class FeatureImportanceResponse(BaseModel):
    features: List[FeatureImportanceItem]
    total: int
