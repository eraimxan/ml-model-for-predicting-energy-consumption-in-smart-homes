from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent
HOURLY_PATH = DATA_ROOT / "outputs" / "data" / "hourly_clean.parquet"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def readings_168() -> list[dict]:
    df = pd.read_parquet(HOURLY_PATH).iloc[-168:]
    out: list[dict] = []
    for ts, row in df.iterrows():
        out.append(
            {
                "timestamp": ts.isoformat(),
                "Global_active_power": float(row["Global_active_power"]),
                "Global_reactive_power": float(row["Global_reactive_power"]),
                "Voltage": float(row["Voltage"]),
                "Global_intensity": float(row["Global_intensity"]),
                "Sub_metering_1": float(row["Sub_metering_1"]),
                "Sub_metering_2": float(row["Sub_metering_2"]),
                "Sub_metering_3": float(row["Sub_metering_3"]),
            }
        )
    return out


def test_health_returns_200_and_model_loaded(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["model_loaded"] is True
    assert data["scaler_loaded"] is True
    assert data["n_features"] == 69


def test_model_info_returns_correct_rmse(client: TestClient) -> None:
    r = client.get("/model/info")
    assert r.status_code == 200
    assert r.json()["rmse_kw"] == 0.4402


def test_model_info_returns_correct_r2(client: TestClient) -> None:
    r = client.get("/model/info")
    assert r.status_code == 200
    assert r.json()["r2"] == 0.6156


def test_model_features_returns_20_items(client: TestClient) -> None:
    r = client.get("/model/features")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 20
    assert len(data["features"]) == 20
    first = data["features"][0]
    assert first["rank"] == 1
    assert isinstance(first["name"], str)
    assert isinstance(first["mean_abs_shap"], float)
    assert first["group"] in {"lag", "rolling", "time"}


def test_predict_with_valid_168_readings_returns_float(
    client: TestClient, readings_168: list[dict]
) -> None:
    r = client.post("/predict", json={"readings": readings_168})
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data["predicted_kw"], float)
    assert data["feature_count"] == 69
    assert data["model"] == "XGBoost"


def test_predict_rejects_fewer_than_168_readings(
    client: TestClient, readings_168: list[dict]
) -> None:
    r = client.post("/predict", json={"readings": readings_168[:50]})
    assert r.status_code == 422


def test_predict_rejects_non_chronological_timestamps(
    client: TestClient, readings_168: list[dict]
) -> None:
    bad = [dict(x) for x in readings_168]
    bad[10], bad[11] = bad[11], bad[10]
    r = client.post("/predict", json={"readings": bad})
    assert r.status_code == 422


def test_predict_output_in_valid_range(
    client: TestClient, readings_168: list[dict]
) -> None:
    r = client.post("/predict", json={"readings": readings_168})
    assert r.status_code == 200
    pred = r.json()["predicted_kw"]
    assert 0.0 <= pred <= 20.0


def test_predict_example_returns_valid_json(client: TestClient) -> None:
    r = client.get("/predict/example")
    assert r.status_code == 200
    data = r.json()
    assert "readings" in data
    assert len(data["readings"]) == 168
    sample = data["readings"][0]
    for key in (
        "timestamp",
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ):
        assert key in sample
