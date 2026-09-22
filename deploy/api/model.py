from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from api.features import FEATURE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent
RESULTS_PATH = DATA_ROOT / "outputs" / "results" / "final_results.csv"
SHAP_PATH = DATA_ROOT / "outputs" / "results" / "shap_top_features.json"
TRAIN_PATH = DATA_ROOT / "outputs" / "splits" / "train.parquet"
TEST_PATH = DATA_ROOT / "outputs" / "splits" / "test.parquet"

MIN_KW = 0.0
MAX_KW = 20.0
MODEL_NAME = "XGBoost (default)"
MODEL_VERSION = "1.0.0"


def _group_for(name: str) -> str:
    if "_lag_" in name:
        return "lag"
    if "_rmean_" in name or "_rstd_" in name:
        return "rolling"
    return "time"


class ModelManager:
    def __init__(self) -> None:
        self._model: Optional[xgb.XGBRegressor] = None
        self._scaler = None
        self._load_time: Optional[float] = None
        self._start_time: float = time.time()

    def load(self, model_path: str, scaler_path: str) -> None:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        self._model = model
        self._scaler = joblib.load(scaler_path)
        self._load_time = time.time()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._scaler is not None

    @property
    def n_features(self) -> int:
        if self._scaler is None:
            return 0
        return int(getattr(self._scaler, "n_features_in_", len(FEATURE_NAMES)))

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def load_time(self) -> Optional[float]:
        return self._load_time

    def predict(self, X_raw: np.ndarray) -> float:
        if not self.is_loaded:
            raise RuntimeError("ModelManager.predict called before load()")
        if X_raw.shape != (1, 69):
            raise ValueError(f"expected X shape (1, 69), got {X_raw.shape}")
        y = self._model.predict(X_raw)
        return float(np.clip(y, MIN_KW, MAX_KW)[0])

    def get_shap_features(self, top_n: int = 20) -> List[dict]:
        with open(SHAP_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        ranking = payload["ranking"] if isinstance(payload, dict) else payload
        out: List[dict] = []
        for i, item in enumerate(ranking[:top_n], start=1):
            name = item["feature"]
            out.append(
                {
                    "rank": i,
                    "name": name,
                    "mean_abs_shap": float(item["mean_abs_shap"]),
                    "group": _group_for(name),
                }
            )
        return out

    def get_model_info(self) -> dict:
        results = pd.read_csv(RESULTS_PATH)
        row = results[results["Model"] == MODEL_NAME].iloc[0]
        rmse_kw = float(row["RMSE"])
        mae_kw = float(row["MAE"])
        r2 = float(row["R2"])
        mape_pct = float(row["MAPE"])

        train = pd.read_parquet(TRAIN_PATH)
        n_training_rows = int(len(train))
        training_period = f"{train.index.min().date()} to {train.index.max().date()}"

        test = pd.read_parquet(TEST_PATH)
        test_period = f"{test.index.min().date()} to {test.index.max().date()}"

        with open(SHAP_PATH, "r", encoding="utf-8") as f:
            shap_payload = json.load(f)
        ranking = shap_payload["ranking"] if isinstance(shap_payload, dict) else shap_payload
        top = ranking[0]

        return {
            "model_name": MODEL_NAME,
            "version": MODEL_VERSION,
            "rmse_kw": rmse_kw,
            "mae_kw": mae_kw,
            "r2": r2,
            "mape_pct": mape_pct,
            "n_features": self.n_features,
            "n_training_rows": n_training_rows,
            "training_period": training_period,
            "test_period": test_period,
            "top_feature": top["feature"],
            "top_feature_shap": float(top["mean_abs_shap"]),
        }


model_manager = ModelManager()
