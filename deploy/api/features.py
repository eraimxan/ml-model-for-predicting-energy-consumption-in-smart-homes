from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

TARGET = "Global_active_power"
COVARIATES: List[str] = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
VARIABLES: List[str] = [TARGET] + COVARIATES
LAGS: List[int] = [1, 2, 3, 6, 12, 24, 48, 168]
WINDOWS: List[int] = [3, 6, 24]


def _build_feature_names() -> List[str]:
    names: List[str] = []
    for var in VARIABLES:
        for h in LAGS:
            names.append(f"{var}_lag_{h}h")
    for w in WINDOWS:
        names.append(f"{TARGET}_rmean_{w}h")
        names.append(f"{TARGET}_rstd_{w}h")
    names.extend(
        ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "mon_sin", "mon_cos", "is_weekend"]
    )
    return names


FEATURE_NAMES: List[str] = _build_feature_names()
assert len(FEATURE_NAMES) == 69, f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries, expected 69"


def _readings_to_dataframe(readings: Sequence) -> pd.DataFrame:
    rows = []
    for r in readings:
        if hasattr(r, "model_dump"):
            rows.append(r.model_dump())
        elif isinstance(r, dict):
            rows.append(r)
        else:
            rows.append(
                {
                    "timestamp": r.timestamp,
                    "Global_active_power": r.Global_active_power,
                    "Global_reactive_power": r.Global_reactive_power,
                    "Voltage": r.Voltage,
                    "Global_intensity": r.Global_intensity,
                    "Sub_metering_1": r.Sub_metering_1,
                    "Sub_metering_2": r.Sub_metering_2,
                    "Sub_metering_3": r.Sub_metering_3,
                }
            )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[VARIABLES].astype(float)


def build_features(readings: Sequence) -> np.ndarray:
    if len(readings) < 168:
        raise ValueError(
            f"build_features requires at least 168 readings (got {len(readings)})"
        )
    df = _readings_to_dataframe(readings)

    next_ts = df.index[-1] + pd.Timedelta(hours=1)
    placeholder = pd.DataFrame({v: [0.0] for v in VARIABLES}, index=[next_ts])
    placeholder.index.name = df.index.name
    df = pd.concat([df, placeholder])

    for var in VARIABLES:
        for h in LAGS:
            df[f"{var}_lag_{h}h"] = df[var].shift(h)

    shifted_target = df[TARGET].shift(1)
    for w in WINDOWS:
        df[f"{TARGET}_rmean_{w}h"] = shifted_target.rolling(w).mean()
        df[f"{TARGET}_rstd_{w}h"] = shifted_target.rolling(w).std()

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["mon_sin"] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    df["mon_cos"] = np.cos(2 * np.pi * (df.index.month - 1) / 12)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(np.int8)

    last = df.iloc[[-1]][FEATURE_NAMES]
    X = last.to_numpy(dtype=np.float64, copy=True)
    return X.reshape(1, 69)


def validate_feature_vector(X: np.ndarray) -> None:
    if X.shape != (1, 69):
        raise AssertionError(f"feature vector shape {X.shape} != (1, 69)")
    if np.isnan(X).any():
        idx = np.where(np.isnan(X[0]))[0].tolist()
        names = [FEATURE_NAMES[i] for i in idx]
        raise AssertionError(f"feature vector contains NaN at positions {idx} ({names})")
