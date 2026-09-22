"""
Weather-features extension to the diploma pipeline.

Mirrors diploma_pipeline.ipynb's Phases 2-8 with 7 hourly weather variables
from Open-Meteo Historical API added as 42 new features (35 lags + 7 current).
Identical XGBoost hyperparameters as Phase 4 (Cell 30) and identical split
boundaries as Phase 3 (Cell 17). Original outputs are not touched; all new
artifacts go under outputs/weather_extension/.
"""
from __future__ import annotations

import json
import os
import random
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import openmeteo_requests
import requests_cache
from retry_requests import retry

# ---------------------------------------------------------------------------
# Section 0 - Setup & reused utilities (verbatim from diploma_pipeline.ipynb)
# ---------------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path.cwd()
OUT = PROJECT_ROOT / "outputs" / "weather_extension"
OUT_SPLITS = OUT / "splits"
OUT.mkdir(parents=True, exist_ok=True)
OUT_SPLITS.mkdir(parents=True, exist_ok=True)

ORIGINAL_DATA = PROJECT_ROOT / "outputs" / "data" / "hourly_clean.parquet"
ORIGINAL_SPLITS = PROJECT_ROOT / "outputs" / "splits"

# --- Original feature-engineering constants (Cell 12) ---------------------
TARGET = "Global_active_power"
COVARIATES = [
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
LAG_VARS = [TARGET] + COVARIATES       # 7
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]   # 8
ROLLING_WINDOWS = [3, 6, 24]

# --- Weather config -------------------------------------------------------
LAT, LON = 48.7769, 2.2903                       # Sceaux, France
WEATHER_START, WEATHER_END = "2006-12-16", "2010-11-30"
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
]
WEATHER_LAGS = [1, 2, 3, 6, 24]                  # 5 lag horizons

# --- Evaluation helpers (verbatim from Cell 22) ---------------------------
MAPE_EPSILON_KW = 0.10  # Pirbazari et al., 2020 - matches original baseline


def safe_mape(y_true, y_pred, eps: float = MAPE_EPSILON_KW) -> float:
    """MAPE in percent with epsilon guard on the denominator."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def evaluate(y_true, y_pred) -> dict:
    """Returns RMSE, MAE, R2, MAPE - the four metrics reported throughout."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
        "MAPE": safe_mape(y_true, y_pred),
    }


# ===========================================================================
# Section 2 - Fetch Open-Meteo weather
# ===========================================================================
def step2_fetch_weather() -> pd.DataFrame:
    weather_path = OUT / "weather_raw.parquet"
    if weather_path.exists():
        print(f"[step2] cache hit: {weather_path}")
        df = pd.read_parquet(weather_path)
    else:
        print(f"[step2] fetching {WEATHER_START} -> {WEATHER_END} for "
              f"({LAT}, {LON})")
        cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        client = openmeteo_requests.Client(session=retry_session)
        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": WEATHER_START,
            "end_date": WEATHER_END,
            "hourly": WEATHER_VARS,
            "timezone": "Europe/Paris",
        }
        url = "https://archive-api.open-meteo.com/v1/archive"
        responses = client.weather_api(url, params=params)
        response = responses[0]

        hourly = response.Hourly()
        # Build the timestamp index from start/end/interval (UTC seconds).
        idx = pd.date_range(
            start=pd.to_datetime(hourly.Time(),    unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(),   unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ).tz_convert("Europe/Paris")
        data = {
            var: hourly.Variables(i).ValuesAsNumpy()
            for i, var in enumerate(WEATHER_VARS)
        }
        df = pd.DataFrame(data, index=idx)
        df.index.name = "timestamp"
        df.to_parquet(weather_path)
        print(f"[step2] saved -> {weather_path}")

    # --- Verification ---
    # 2006-12-16 to 2010-11-30 inclusive = 1446 days * 24h = 34,704 hours;
    # allow a tolerance for DST clock adjustments around the boundary.
    assert 34500 <= len(df) <= 36500, f"row count {len(df)} outside 34500-36500"
    assert df.shape[1] == 7, f"col count {df.shape[1]} != 7"
    assert df.index.tz is not None, "index must be tz-aware"
    assert str(df.index.tz) == "Europe/Paris", f"tz={df.index.tz} != Europe/Paris"
    assert df.index.is_monotonic_increasing, "index not monotonic"
    for c in df.columns:
        assert not df[c].isna().all(), f"column {c} all NaN"
    print(f"[step2] shape={df.shape}  tz={df.index.tz}  "
          f"range={df.index.min()} -> {df.index.max()}")
    print(f"[step2] head(3):\n{df.head(3)}")
    print(f"[step2] nulls per column:\n{df.isna().sum().to_dict()}")
    return df


# ===========================================================================
# Section 3 - Merge with hourly UCI data
# ===========================================================================
def step3_merge(weather: pd.DataFrame) -> pd.DataFrame:
    merged_path = OUT / "hourly_merged.parquet"
    if merged_path.exists():
        print(f"[step3] cache hit: {merged_path}")
        merged = pd.read_parquet(merged_path)
    else:
        uci = pd.read_parquet(ORIGINAL_DATA)
        # The UCI hourly index has no DST artifacts: 2007-03-25 02:00 exists
        # (spring-forward hour, skipped in Paris local time) and 2007-10-28 02:00
        # appears only once (autumn fallback, would be duplicated in Paris local).
        # Therefore the UCI timestamps are fixed CET (UTC+1, no DST), not Paris
        # wall-clock. Localize as Etc/GMT-1 (= UTC+1 fixed) and align weather to
        # the same offset to merge by absolute moment.
        if uci.index.tz is None:
            uci.index = uci.index.tz_localize("Etc/GMT-1")
        else:
            uci.index = uci.index.tz_convert("Etc/GMT-1")
        # Weather was fetched with timezone="Europe/Paris" (DST-aware). Convert
        # to fixed UTC+1 so timestamps match UCI absolute moments.
        weather_aligned = weather.tz_convert("Etc/GMT-1")

        merged = uci.join(weather_aligned, how="inner")

        # Forward-fill weather gaps of <= 2 hours, if any, then assert clean.
        weather_cols = list(weather.columns)
        merged[weather_cols] = merged[weather_cols].ffill(limit=2)
        merged.to_parquet(merged_path)
        print(f"[step3] saved -> {merged_path}")

    # --- Verification ---
    assert merged.shape[1] == 14, f"col count {merged.shape[1]} != 14"
    assert merged.index.is_monotonic_increasing
    assert not merged.index.has_duplicates
    assert merged.index.tz is not None
    assert 33000 <= len(merged) <= 35000, f"row count {len(merged)} outside 33000-35000"
    assert merged[list(weather.columns)].isna().sum().sum() == 0, "weather NaN remain"
    print(f"[step3] shape={merged.shape}  range={merged.index.min()} -> "
          f"{merged.index.max()}")
    print(f"[step3] cols={list(merged.columns)}")
    print(f"[step3] head(3):\n{merged.head(3)}")
    return merged


# ===========================================================================
# Section 4 - Feature engineering (re-run Cell 12 + add weather features)
# ===========================================================================
def step4_features(merged: pd.DataFrame) -> pd.DataFrame:
    feat_path = OUT / "features_weather.parquet"
    if feat_path.exists():
        print(f"[step4] cache hit: {feat_path}")
        df = pd.read_parquet(feat_path)
    else:
        # ---- Replicate Cell 12 verbatim on the merged UCI columns. -------
        base = merged[[TARGET] + COVARIATES].copy()

        # 56 UCI lag features
        for var in LAG_VARS:
            for h in LAGS:
                base[f"{var}_lag_{h}h"] = base[var].shift(h)

        # 6 rolling stats on TARGET, shifted by 1 to avoid leakage
        shifted = base[TARGET].shift(1)
        for w in ROLLING_WINDOWS:
            base[f"{TARGET}_rmean_{w}h"] = shifted.rolling(w).mean()
            base[f"{TARGET}_rstd_{w}h"]  = shifted.rolling(w).std()

        # 7 cyclical time features
        idx = base.index
        base["hour_sin"]   = np.sin(2 * np.pi * idx.hour       / 24)
        base["hour_cos"]   = np.cos(2 * np.pi * idx.hour       / 24)
        base["dow_sin"]    = np.sin(2 * np.pi * idx.dayofweek  / 7)
        base["dow_cos"]    = np.cos(2 * np.pi * idx.dayofweek  / 7)
        base["mon_sin"]    = np.sin(2 * np.pi * (idx.month - 1) / 12)
        base["mon_cos"]    = np.cos(2 * np.pi * (idx.month - 1) / 12)
        base["is_weekend"] = (idx.dayofweek >= 5).astype(np.int8)

        # Drop raw covariates - only their lags are features.
        base = base.drop(columns=COVARIATES)

        # ---- Weather features --------------------------------------------
        for v in WEATHER_VARS:
            for h in WEATHER_LAGS:
                base[f"{v}_lag_{h}h"] = merged[v].shift(h)
            base[f"{v}_now"] = merged[v].values  # current-hour, no shift

        df = base.dropna()
        df.to_parquet(feat_path)
        print(f"[step4] saved -> {feat_path}")

    # --- Verification ---
    n_cols = df.shape[1]
    feature_cols = [c for c in df.columns if c != TARGET]
    n_features = len(feature_cols)
    # Group counts
    n_uci_lags = sum(c.endswith(tuple(f"_lag_{h}h" for h in LAGS))
                     and not any(c.startswith(v + "_") for v in WEATHER_VARS)
                     for c in feature_cols)
    n_weather_lags = sum(any(c == f"{v}_lag_{h}h"
                              for v in WEATHER_VARS for h in WEATHER_LAGS)
                          for c in feature_cols)
    n_weather_now = sum(c.endswith("_now") for c in feature_cols)
    print(f"[step4] shape={df.shape}  features={n_features}  "
          f"(uci_lags={n_uci_lags}, weather_lags={n_weather_lags}, "
          f"weather_now={n_weather_now})")
    assert n_features == 111, f"expected 111 features, got {n_features}"
    assert n_weather_lags == 35, f"expected 35 weather lags, got {n_weather_lags}"
    assert n_weather_now == 7,   f"expected 7 weather_now, got {n_weather_now}"
    assert df.isna().sum().sum() == 0, "NaN values remain"
    # Spot-check 10 random rows: temperature_2m_now == merged temperature_2m at same ts
    rng = np.random.RandomState(SEED)
    sample_ts = rng.choice(df.index, size=10, replace=False)
    for ts in sample_ts:
        a = df.loc[ts, "temperature_2m_now"]
        b = merged.loc[ts, "temperature_2m"]
        assert np.isclose(a, b, atol=1e-9), f"now-vs-merged mismatch at {ts}: {a} vs {b}"
    # Weather lag direction sanity (one sample): temperature_2m_lag_1h at t == temperature_2m at t-1h
    t = df.index[100]
    t_minus = t - pd.Timedelta(hours=1)
    if t_minus in merged.index:
        assert np.isclose(df.loc[t, "temperature_2m_lag_1h"],
                          merged.loc[t_minus, "temperature_2m"], atol=1e-9), \
            f"lag direction wrong at {t}"
    print(f"[step4] verification passed (range {df.index.min()} -> "
          f"{df.index.max()})")
    return df


# ===========================================================================
# Section 5 - Split using original boundaries + fit scaler
# ===========================================================================
def step5_split(features: pd.DataFrame):
    train_p = OUT_SPLITS / "train_w.parquet"
    val_p   = OUT_SPLITS / "val_w.parquet"
    test_p  = OUT_SPLITS / "test_w.parquet"
    scaler_p = OUT / "scaler_weather.joblib"

    # Extract boundaries from ORIGINAL splits (timestamp based). Original splits
    # are naive timestamps matching the UCI hourly_clean.parquet index, which we
    # interpret as fixed UTC+1 (see step3_merge comment).
    orig_val = pd.read_parquet(ORIGINAL_SPLITS / "val.parquet")
    orig_test = pd.read_parquet(ORIGINAL_SPLITS / "test.parquet")
    val_start  = pd.Timestamp(orig_val.index.min()).tz_localize("Etc/GMT-1")
    test_start = pd.Timestamp(orig_test.index.min()).tz_localize("Etc/GMT-1")
    print(f"[step5] boundaries: val_start={val_start}  test_start={test_start}")

    feature_cols = [c for c in features.columns if c != TARGET]
    train = features.loc[features.index < val_start]
    val   = features.loc[(features.index >= val_start) & (features.index < test_start)]
    test  = features.loc[features.index >= test_start]

    train.to_parquet(train_p)
    val.to_parquet(val_p)
    test.to_parquet(test_p)

    scaler = StandardScaler()
    scaler.fit(train[feature_cols].to_numpy())
    joblib.dump(scaler, scaler_p)

    # --- Verification ---
    assert len(train) + len(val) + len(test) == len(features), \
        "split rows do not sum to total"
    assert train.index.max() < val.index.min(),  "train overlaps val"
    assert val.index.max()   < test.index.min(), "val overlaps test"
    assert 23000 <= len(train) <= 24500, f"train size {len(train)} outside 23000-24500"
    assert 4800  <= len(val)   <= 5200,  f"val size {len(val)} outside 4800-5200"
    assert 4800  <= len(test)  <= 5200,  f"test size {len(test)} outside 4800-5200"
    assert scaler.n_features_in_ == 111

    temp_now_idx = feature_cols.index("temperature_2m_now")
    temp_mean = scaler.mean_[temp_now_idx]
    assert 5.0 <= temp_mean <= 15.0, \
        f"Paris temperature_2m_now train mean {temp_mean:.2f} outside 5-15 C"
    print(f"[step5] train={len(train)} val={len(val)} test={len(test)}")
    print(f"[step5] train range: {train.index.min()} -> {train.index.max()}")
    print(f"[step5] val   range: {val.index.min()} -> {val.index.max()}")
    print(f"[step5] test  range: {test.index.min()} -> {test.index.max()}")
    print(f"[step5] temperature_2m_now train mean = {temp_mean:.3f} C "
          f"(Paris annual mean ~12 C: OK)")
    print(f"[step5] scaler n_features_in_ = {scaler.n_features_in_}")
    return train, val, test, scaler


# ===========================================================================
# Section 6 - Train XGBoost (same hyperparameters as Cell 30)
# ===========================================================================
def step6_train(train, val, test, scaler):
    pred_p  = OUT / "pred_xgb_weather.npy"
    model_p = OUT / "xgb_weather_model.json"
    cmp_p   = OUT / "comparison.csv"

    feature_cols = [c for c in train.columns if c != TARGET]
    X_train = scaler.transform(train[feature_cols].to_numpy())
    X_val   = scaler.transform(val[feature_cols].to_numpy())
    X_test  = scaler.transform(test[feature_cols].to_numpy())
    y_train = train[TARGET].to_numpy()
    y_val   = val[TARGET].to_numpy()
    y_test  = test[TARGET].to_numpy()

    if model_p.exists() and pred_p.exists():
        print(f"[step6] cache hit: {model_p} + {pred_p}")
        xgb_weather = xgb.XGBRegressor()
        xgb_weather.load_model(str(model_p))
        pred_test = np.load(pred_p)
    else:
        # Original Cell 30 hyperparameters (NOT the brief's 1.0/1.0 variant)
        xgb_weather = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric="rmse",
        )
        xgb_weather.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        pred_test = np.clip(xgb_weather.predict(X_test), 0.0, 20.0)
        np.save(pred_p, pred_test)
        xgb_weather.save_model(str(model_p))
        print(f"[step6] trained: best_iter={xgb_weather.best_iteration} "
              f"best_val_rmse={xgb_weather.best_score:.4f}")

    # --- Aligned 5,075-hour window: drop first 24 ---
    SEQ_LEN = 24
    y_test_aligned    = y_test[SEQ_LEN:]
    pred_test_aligned = pred_test[SEQ_LEN:]
    metrics_full     = evaluate(y_test, pred_test)
    metrics_aligned  = evaluate(y_test_aligned, pred_test_aligned)

    # --- Baseline (from original benchmark_comparison.csv) ---
    baseline = {"RMSE": 0.4402, "MAE": 0.3008, "R2": 0.6156, "MAPE": 39.36}

    cmp_df = pd.DataFrame([
        {"model": "XGBoost (original 69 features)",        "n_features": 69,  **baseline},
        {"model": "XGBoost (with weather 111 features)",   "n_features": 111, **metrics_aligned},
    ])
    cmp_df.to_csv(cmp_p, index=False)

    # --- Verification ---
    assert xgb_weather.best_iteration > 0, "model did not train"
    assert pred_test.shape == y_test.shape
    assert 0.0 < metrics_aligned["RMSE"] < 0.60, \
        f"aligned RMSE {metrics_aligned['RMSE']:.4f} outside (0, 0.60)"
    assert 0.0 < metrics_aligned["R2"] < 1.0
    print(f"[step6] full test ({len(y_test)} rows):    {metrics_full}")
    print(f"[step6] aligned ({len(y_test_aligned)} rows): {metrics_aligned}")
    delta_rmse = metrics_aligned["RMSE"] - baseline["RMSE"]
    delta_r2   = metrics_aligned["R2"]   - baseline["R2"]
    pct_rmse   = 100 * delta_rmse / baseline["RMSE"]
    pct_r2     = 100 * delta_r2   / baseline["R2"]
    print(f"[step6] ΔRMSE = {delta_rmse:+.4f} ({pct_rmse:+.2f}%)")
    print(f"[step6] ΔR²   = {delta_r2:+.4f} ({pct_r2:+.2f}%)")
    print(f"[step6] saved comparison.csv:\n{cmp_df.to_string(index=False)}")
    return xgb_weather, X_test, y_test_aligned, pred_test_aligned, metrics_aligned, baseline, feature_cols


# ===========================================================================
# Section 7 - SHAP on 111-feature model
# ===========================================================================
def step7_shap(xgb_weather, test, feature_cols):
    import shap
    shap_json = OUT / "shap_weather_features.json"
    shap_png  = OUT / "shap_weather_bar.png"

    rng = np.random.RandomState(SEED)
    n_sample = 500
    sample_idx = np.sort(rng.choice(len(test), size=n_sample, replace=False))
    X_sample = test[feature_cols].iloc[sample_idx].to_numpy()
    # Scale to match the trained model's input space
    scaler = joblib.load(OUT / "scaler_weather.joblib")
    X_sample_scaled = scaler.transform(X_sample)

    explainer = shap.TreeExplainer(xgb_weather)
    shap_values = explainer.shap_values(X_sample_scaled)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ranking = [{"feature": feature_cols[i], "mean_abs_shap": float(mean_abs[i])}
               for i in order]
    with open(shap_json, "w", encoding="utf-8") as f:
        json.dump({"sample_size": n_sample, "ranking": ranking}, f, indent=2)

    # --- Extended colour map ---
    def colour_for(name: str) -> str:
        if name.endswith("_now"):                     return "#d62728"  # weather_now: red
        if any(name.startswith(v + "_lag_") for v in WEATHER_VARS):
            return "#17becf"                                            # weather_lag: teal
        if "_lag_" in name:                          return "#1f77b4"   # uci_lag
        if "rmean" in name or "rstd" in name:        return "#ff7f0e"   # rolling
        return "#2ca02c"                                                # time

    def category_for(name: str) -> str:
        if name.endswith("_now"):                     return "weather_now"
        if any(name.startswith(v + "_lag_") for v in WEATHER_VARS):
            return "weather_lag"
        if "_lag_" in name:                          return "lag"
        if "rmean" in name or "rstd" in name:        return "rolling"
        return "time"

    top_n = 20
    top = ranking[:top_n][::-1]
    top_feats  = [r["feature"] for r in top]
    top_values = [r["mean_abs_shap"] for r in top]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top_feats, top_values, color=[colour_for(n) for n in top_feats])
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(f"Top {top_n} features by mean |SHAP| "
                 f"(XGBoost + weather, n={n_sample} samples)")
    ax.grid(axis="x", alpha=0.3)
    legend_keys = [
        ("lag features (UCI)",     "#1f77b4"),
        ("rolling stats",          "#ff7f0e"),
        ("time features",          "#2ca02c"),
        ("weather lag features",   "#17becf"),
        ("weather current-hour",   "#d62728"),
    ]
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, fc=c) for _, c in legend_keys],
        [n for n, _ in legend_keys],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(shap_png, dpi=150)
    plt.close(fig)

    # --- Verification ---
    assert len(ranking) == 111, f"SHAP ranking has {len(ranking)} != 111"
    top20_feats = [r["feature"] for r in ranking[:20]]
    weather_in_top20 = [f for f in top20_feats if category_for(f) in ("weather_lag", "weather_now")]
    if not weather_in_top20:
        print("[step7] WARNING: no weather features in top 20 - check merge alignment")
    assert shap_json.exists() and shap_json.stat().st_size > 0
    assert shap_png.exists() and shap_png.stat().st_size > 20_000, \
        f"shap_weather_bar.png too small: {shap_png.stat().st_size} bytes"
    print(f"[step7] top 10:")
    for i, r in enumerate(ranking[:10], 1):
        cat = category_for(r["feature"])
        print(f"  {i:2d}. {r['feature']:35s}  mean|SHAP|={r['mean_abs_shap']:.4f}  [{cat}]")
    print(f"[step7] weather features in top 20: {len(weather_in_top20)}")
    return ranking, weather_in_top20


# ===========================================================================
# Section 8 - Improvement report + comparison plot
# ===========================================================================
def step8_report(metrics_aligned, baseline, ranking, weather_in_top20):
    report_p = OUT / "improvement_report.txt"
    plot_p   = OUT / "model_comparison_weather.png"

    delta = {k: metrics_aligned[k] - baseline[k] for k in baseline}
    pct = {k: 100.0 * delta[k] / baseline[k] for k in baseline}

    d_r2 = delta["R2"]
    if d_r2 > 0.05:
        interp = ("Weather features provide significant accuracy improvement. "
                  "Temperature- and radiation-driven load components are now "
                  "captured in addition to the autoregressive signal.")
    elif d_r2 > 0.02:
        interp = ("Weather features provide moderate accuracy improvement. "
                  "Adds value on top of the autoregressive lag structure but "
                  "the latter remains dominant.")
    else:
        interp = ("Weather features provide marginal improvement; the dominant "
                  "signal remains autoregressive lag structure of the household "
                  "load itself.")

    top_w = [r for r in ranking if any(r["feature"].startswith(v + "_") or r["feature"] == v + "_now"
                                         for v in WEATHER_VARS)][:10]

    lines = []
    lines.append("=== WEATHER EXTENSION RESULTS ===")
    lines.append("")
    lines.append("Dataset:         UCI IHEPC, Sceaux, France, hourly (2006-2010)")
    lines.append("Weather source:  Open-Meteo Historical API (archive-api.open-meteo.com)")
    lines.append("Weather vars:    temperature_2m, relative_humidity_2m, apparent_temperature,")
    lines.append("                 precipitation, wind_speed_10m, shortwave_radiation, cloud_cover")
    lines.append("Weather lags:    {1, 2, 3, 6, 24} hours + current-hour values")
    lines.append("New features:    42 (35 lag + 7 current-hour)")
    lines.append("Total features:  111 (69 original + 42 weather)")
    lines.append("")
    lines.append("--- Test Set Results (aligned 5,075-hour window) ---")
    lines.append("")
    lines.append("Metric       | Original (69 feat) | Weather (111 feat) | Delta    | Pct change")
    lines.append("-------------|--------------------|--------------------|----------|----------")
    lines.append(f"RMSE (kW)    | {baseline['RMSE']:<18.4f} | {metrics_aligned['RMSE']:<18.4f} | {delta['RMSE']:+8.4f} | {pct['RMSE']:+7.2f}%")
    lines.append(f"MAE  (kW)    | {baseline['MAE']:<18.4f} | {metrics_aligned['MAE']:<18.4f} | {delta['MAE']:+8.4f} | {pct['MAE']:+7.2f}%")
    lines.append(f"R^2          | {baseline['R2']:<18.4f} | {metrics_aligned['R2']:<18.4f} | {delta['R2']:+8.4f} | {pct['R2']:+7.2f}%")
    lines.append(f"MAPE (%)     | {baseline['MAPE']:<18.4f} | {metrics_aligned['MAPE']:<18.4f} | {delta['MAPE']:+8.4f} | {pct['MAPE']:+7.2f}%")
    lines.append("")
    lines.append("--- Top weather features by SHAP importance ---")
    lines.append("Rank  Feature                                Mean |SHAP|")
    for i, r in enumerate(top_w, 1):
        lines.append(f"{i:>4d}  {r['feature']:<37s}  {r['mean_abs_shap']:.4f}")
    lines.append("")
    lines.append("--- Interpretation ---")
    lines.append(interp)
    lines.append("")
    lines.append("=== END OF REPORT ===")
    report_text = "\n".join(lines)
    report_p.write_text(report_text, encoding="utf-8")

    # --- Comparison plot (same dual-axis style as 03_model_comparison.png) ---
    cmp_df = pd.DataFrame([
        {"Model": "XGBoost (69 features)",  "RMSE": baseline["RMSE"],         "R2": baseline["R2"]},
        {"Model": "XGBoost + weather (111)", "RMSE": metrics_aligned["RMSE"], "R2": metrics_aligned["R2"]},
    ])
    fig, ax1 = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(cmp_df))
    bars = ax1.bar(xs, cmp_df["RMSE"], color="#4c72b0", alpha=0.85)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(cmp_df["Model"], rotation=10, ha="right")
    ax1.set_ylabel("RMSE (kW)", color="#4c72b0")
    ax1.tick_params(axis="y", labelcolor="#4c72b0")
    ax1.set_ylim(0, cmp_df["RMSE"].max() * 1.25)
    for b, v in zip(bars, cmp_df["RMSE"]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01,
                 f"{v:.4f}", ha="center", fontsize=10)
    ax2 = ax1.twinx()
    ax2.plot(xs, cmp_df["R2"], color="#dd8452", marker="o", linewidth=2)
    for x, v in zip(xs, cmp_df["R2"]):
        ax2.text(x, v + 0.01, f"{v:.4f}", ha="center", fontsize=10, color="#dd8452")
    ax2.set_ylabel("R^2", color="#dd8452")
    ax2.tick_params(axis="y", labelcolor="#dd8452")
    ax2.set_ylim(min(cmp_df["R2"]) - 0.10, 1.0)
    ax1.set_title("XGBoost accuracy improvement with weather features")
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_p, dpi=150)
    plt.close(fig)

    # --- Final file-existence asserts ---
    expected = [
        "weather_raw.parquet",
        "hourly_merged.parquet",
        "features_weather.parquet",
        "splits/train_w.parquet",
        "splits/val_w.parquet",
        "splits/test_w.parquet",
        "scaler_weather.joblib",
        "pred_xgb_weather.npy",
        "xgb_weather_model.json",
        "shap_weather_features.json",
        "shap_weather_bar.png",
        "comparison.csv",
        "improvement_report.txt",
        "model_comparison_weather.png",
    ]
    missing = [p for p in expected if not (OUT / p).exists()]
    assert not missing, f"missing artifacts: {missing}"
    print("[step8] ALL OUTPUT FILES VERIFIED")
    print()
    print(report_text)


# ===========================================================================
# Orchestration
# ===========================================================================
def main():
    # CLI: weather_extension.py [step]  where step in {2..8, all}.
    # Runs all steps up to and including the requested one (so earlier outputs
    # are present). Re-runs are cheap because each step is parquet-cached.
    step_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    last = 8 if step_arg == "all" else int(step_arg)
    assert 2 <= last <= 8, f"step must be 2..8 or 'all', got {step_arg}"

    weather  = step2_fetch_weather()                       if last >= 2 else None
    merged   = step3_merge(weather)                        if last >= 3 else None
    features = step4_features(merged)                      if last >= 4 else None
    if last < 5:
        return
    train, val, test, scaler = step5_split(features)
    if last < 6:
        return
    xgb_weather, _X_test, _y_aligned, _pred_aligned, m_aligned, baseline, fcols = \
        step6_train(train, val, test, scaler)
    if last < 7:
        return
    ranking, weather_top = step7_shap(xgb_weather, test, fcols)
    if last < 8:
        return
    step8_report(m_aligned, baseline, ranking, weather_top)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
