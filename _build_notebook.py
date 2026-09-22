"""Build weather_extension.ipynb from the verified weather_extension.py script.

The notebook mirrors the 8-section structure of the script. Each section gets a
markdown header (matching diploma_pipeline.ipynb style) and a runnable code
cell. All artifacts persist to outputs/weather_extension/ so re-runs are fast.
"""
from pathlib import Path
import nbformat as nbf

NB = nbf.v4.new_notebook()
NB.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Weather-Features Extension to the Diploma Pipeline

**AITU, June 2026 — Yerassyl Raimkhan**

This notebook extends `diploma_pipeline.ipynb` with 7 hourly weather variables
fetched from the Open-Meteo Historical API for Sceaux, France
(48.7769° N, 2.2903° E), the location of the UCI household. It builds 42 new
features (35 lag + 7 current-hour) on top of the existing 69, re-trains XGBoost
with the same Phase 4 hyperparameters, and quantifies the improvement on the
same aligned 5,075-hour test window.

**Constraints (from the project brief):**
- Identical split boundaries as `outputs/splits/{train,val,test}.parquet`.
- `StandardScaler` refit on train only over the 111 features.
- XGBoost config identical to `diploma_pipeline.ipynb` Cell 30 (the published
  baseline that produced RMSE = 0.4402 / R² = 0.6156 / MAPE = 39.36).
- `safe_mape` uses `MAPE_EPSILON_KW = 0.10` (same as the baseline).
- All outputs go under `outputs/weather_extension/`; the original `outputs/`
  tree is read-only and untouched by this notebook.

**Timezone note.** The UCI hourly index has no DST artifacts
(2007-03-25 02:00 exists despite being skipped in Paris local; 2007-10-28 02:00
appears only once despite being duplicated in Paris local), so we interpret it
as **fixed UTC+1 (CET, no DST)** via `Etc/GMT-1`. Open-Meteo weather is fetched
in Europe/Paris and converted to the same fixed offset before merging.
""")

md("## Section 0 — Imports, constants, and reused utilities")

code("""from __future__ import annotations

import json, os, random, sys, warnings
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)

PROJECT_ROOT = Path.cwd()
OUT = PROJECT_ROOT / "outputs" / "weather_extension"
OUT_SPLITS = OUT / "splits"
OUT.mkdir(parents=True, exist_ok=True)
OUT_SPLITS.mkdir(parents=True, exist_ok=True)

ORIGINAL_DATA = PROJECT_ROOT / "outputs" / "data" / "hourly_clean.parquet"
ORIGINAL_SPLITS = PROJECT_ROOT / "outputs" / "splits"

# --- Original feature-engineering constants (diploma_pipeline.ipynb Cell 12) -
TARGET = "Global_active_power"
COVARIATES = [
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
LAG_VARS = [TARGET] + COVARIATES       # 7
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]   # 8
ROLLING_WINDOWS = [3, 6, 24]

# --- Weather config ---------------------------------------------------------
LAT, LON = 48.7769, 2.2903
WEATHER_START, WEATHER_END = "2006-12-16", "2010-11-30"
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "wind_speed_10m", "shortwave_radiation", "cloud_cover",
]
WEATHER_LAGS = [1, 2, 3, 6, 24]

# --- Evaluation helpers (verbatim from diploma_pipeline.ipynb Cell 22) ------
MAPE_EPSILON_KW = 0.10  # Pirbazari et al., 2020

def safe_mape(y_true, y_pred, eps: float = MAPE_EPSILON_KW) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
        "MAPE": safe_mape(y_true, y_pred),
    }

print("setup OK; SEED =", SEED, "; OUT =", OUT)
""")

md("""## Section 1 — Dependency check

The three Open-Meteo client libraries (`openmeteo-requests`, `requests-cache`,
`retry-requests`) are required for Section 2. If not already installed, run
`pip install openmeteo-requests requests-cache retry-requests` from a terminal.""")

code("""import importlib.metadata as md
for p in ("openmeteo-requests", "requests-cache", "retry-requests"):
    print(f"{p:<22s} {md.version(p)}")
print("imports OK")
""")

md("""## Section 2 — Fetch historical weather

Open-Meteo's archive (ERA5 reanalysis) returns 1,446 days × 24 h = 34,704 hourly
rows for the 2006-12-16 → 2010-11-30 window. The cached parquet means re-runs
are instant.""")

code("""weather_path = OUT / "weather_raw.parquet"
if weather_path.exists():
    weather = pd.read_parquet(weather_path)
    print(f"cache hit: {weather_path}")
else:
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": WEATHER_START, "end_date": WEATHER_END,
        "hourly": WEATHER_VARS, "timezone": "Europe/Paris",
    }
    response = client.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )[0]
    hourly = response.Hourly()
    idx = pd.date_range(
        start=pd.to_datetime(hourly.Time(),  unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_convert("Europe/Paris")
    weather = pd.DataFrame(
        {var: hourly.Variables(i).ValuesAsNumpy() for i, var in enumerate(WEATHER_VARS)},
        index=idx,
    )
    weather.index.name = "timestamp"
    weather.to_parquet(weather_path)
    print(f"saved -> {weather_path}")

# --- Verification ---
assert 34_500 <= len(weather) <= 36_500, f"unexpected row count {len(weather)}"
assert weather.shape[1] == 7
assert str(weather.index.tz) == "Europe/Paris"
assert weather.index.is_monotonic_increasing
assert weather.isna().sum().sum() == 0
print(f"shape={weather.shape}  range={weather.index.min()} -> {weather.index.max()}")
weather.head(3)
""")

md("""## Section 3 — Merge with UCI hourly data

The UCI hourly index is naive and has no DST artifacts → interpret as fixed
UTC+1 via `Etc/GMT-1`. Convert the Paris-local weather to the same fixed offset
so the inner join matches absolute moments in time.""")

code("""merged_path = OUT / "hourly_merged.parquet"
if merged_path.exists():
    merged = pd.read_parquet(merged_path)
    print(f"cache hit: {merged_path}")
else:
    uci = pd.read_parquet(ORIGINAL_DATA)
    if uci.index.tz is None:
        uci.index = uci.index.tz_localize("Etc/GMT-1")  # fixed UTC+1, no DST
    else:
        uci.index = uci.index.tz_convert("Etc/GMT-1")
    weather_aligned = weather.tz_convert("Etc/GMT-1")
    merged = uci.join(weather_aligned, how="inner")
    # Forward-fill any weather gaps of <= 2 hours; assert clean afterwards.
    merged[list(weather.columns)] = merged[list(weather.columns)].ffill(limit=2)
    merged.to_parquet(merged_path)
    print(f"saved -> {merged_path}")

# --- Verification ---
assert merged.shape[1] == 14
assert merged.index.is_monotonic_increasing and not merged.index.has_duplicates
assert merged.index.tz is not None
assert 33_000 <= len(merged) <= 35_000
assert merged[list(weather.columns)].isna().sum().sum() == 0
print(f"shape={merged.shape}  range={merged.index.min()} -> {merged.index.max()}")
print("nulls per column:", merged.isna().sum().to_dict())
merged.head(3)
""")

md("""## Section 4 — Feature engineering (69 original + 42 weather)

Re-runs the diploma Cell 12 lag/rolling/cyclical engineering on the merged
frame, then appends 35 weather lags ({1, 2, 3, 6, 24} h × 7 variables) and 7
current-hour weather values. Result: **33,989 rows × 112 columns** (1 target +
111 features), the same row count as the original `features.parquet` (the 168 h
lag remains the binding constraint).""")

code("""feat_path = OUT / "features_weather.parquet"
if feat_path.exists():
    features = pd.read_parquet(feat_path)
    print(f"cache hit: {feat_path}")
else:
    base = merged[[TARGET] + COVARIATES].copy()
    # 56 UCI lag features
    for var in LAG_VARS:
        for h in LAGS:
            base[f"{var}_lag_{h}h"] = base[var].shift(h)
    # 6 rolling stats on TARGET, shift(1) to avoid leakage
    shifted = base[TARGET].shift(1)
    for w in ROLLING_WINDOWS:
        base[f"{TARGET}_rmean_{w}h"] = shifted.rolling(w).mean()
        base[f"{TARGET}_rstd_{w}h"]  = shifted.rolling(w).std()
    # 7 cyclical time features
    idx = base.index
    base["hour_sin"]   = np.sin(2 * np.pi * idx.hour      / 24)
    base["hour_cos"]   = np.cos(2 * np.pi * idx.hour      / 24)
    base["dow_sin"]    = np.sin(2 * np.pi * idx.dayofweek / 7)
    base["dow_cos"]    = np.cos(2 * np.pi * idx.dayofweek / 7)
    base["mon_sin"]    = np.sin(2 * np.pi * (idx.month - 1) / 12)
    base["mon_cos"]    = np.cos(2 * np.pi * (idx.month - 1) / 12)
    base["is_weekend"] = (idx.dayofweek >= 5).astype(np.int8)
    # Drop raw covariates - only their lags are features.
    base = base.drop(columns=COVARIATES)
    # 35 weather lags + 7 weather current-hour
    for v in WEATHER_VARS:
        for h in WEATHER_LAGS:
            base[f"{v}_lag_{h}h"] = merged[v].shift(h)
        base[f"{v}_now"] = merged[v].values
    features = base.dropna()
    features.to_parquet(feat_path)
    print(f"saved -> {feat_path}")

feature_cols = [c for c in features.columns if c != TARGET]
n_weather_lags = sum(any(c == f"{v}_lag_{h}h" for v in WEATHER_VARS for h in WEATHER_LAGS)
                     for c in feature_cols)
n_weather_now  = sum(c.endswith("_now") for c in feature_cols)
print(f"shape={features.shape}  features={len(feature_cols)}  "
      f"(weather_lags={n_weather_lags}, weather_now={n_weather_now})")
assert len(feature_cols) == 111
assert n_weather_lags == 35 and n_weather_now == 7
assert features.isna().sum().sum() == 0
# Spot-check 10 random rows: temperature_2m_now must equal merged temperature_2m
rng = np.random.RandomState(SEED)
sample_ts = rng.choice(features.index, size=10, replace=False)
for ts in sample_ts:
    assert np.isclose(features.loc[ts, "temperature_2m_now"],
                      merged.loc[ts, "temperature_2m"], atol=1e-9)
print("verification passed")
""")

md("""## Section 5 — Chronological split + scaler

Reuses the exact boundary timestamps from `outputs/splits/{val,test}.parquet`
so the test window is identical to the original. Fits a fresh `StandardScaler`
on the 111 train features only.""")

code("""orig_val  = pd.read_parquet(ORIGINAL_SPLITS / "val.parquet")
orig_test = pd.read_parquet(ORIGINAL_SPLITS / "test.parquet")
val_start  = pd.Timestamp(orig_val.index.min()).tz_localize("Etc/GMT-1")
test_start = pd.Timestamp(orig_test.index.min()).tz_localize("Etc/GMT-1")
print(f"boundaries: val_start={val_start}  test_start={test_start}")

train = features.loc[features.index < val_start]
val   = features.loc[(features.index >= val_start) & (features.index < test_start)]
test  = features.loc[features.index >= test_start]
train.to_parquet(OUT_SPLITS / "train_w.parquet")
val.to_parquet  (OUT_SPLITS / "val_w.parquet")
test.to_parquet (OUT_SPLITS / "test_w.parquet")

scaler = StandardScaler().fit(train[feature_cols].to_numpy())
joblib.dump(scaler, OUT / "scaler_weather.joblib")

# --- Verification ---
assert len(train) + len(val) + len(test) == len(features)
assert train.index.max() < val.index.min() < val.index.max() < test.index.min()
assert scaler.n_features_in_ == 111
temp_mean = scaler.mean_[feature_cols.index("temperature_2m_now")]
assert 5.0 <= temp_mean <= 15.0, f"Paris temp train mean {temp_mean:.2f} outside 5-15 C"
print(f"train={len(train)}  val={len(val)}  test={len(test)}")
print(f"train range: {train.index.min()} -> {train.index.max()}")
print(f"val   range: {val.index.min()} -> {val.index.max()}")
print(f"test  range: {test.index.min()} -> {test.index.max()}")
print(f"temperature_2m_now train mean = {temp_mean:.3f} C  (Paris annual ~12 C)")
""")

md("""## Section 6 — Train XGBoost with the original Phase 4 config

Exact same hyperparameters as `diploma_pipeline.ipynb` Cell 30 — only the
feature set changes (69 → 111). The test predictions drop the first 24 rows so
the metrics are computed on the same aligned 5,075-hour window reported in
`outputs/results/final_results.csv`.""")

code("""X_train = scaler.transform(train[feature_cols].to_numpy())
X_val   = scaler.transform(val[feature_cols].to_numpy())
X_test  = scaler.transform(test[feature_cols].to_numpy())
y_train = train[TARGET].to_numpy()
y_val   = val[TARGET].to_numpy()
y_test  = test[TARGET].to_numpy()

model_p = OUT / "xgb_weather_model.json"
pred_p  = OUT / "pred_xgb_weather.npy"
if model_p.exists() and pred_p.exists():
    xgb_weather = xgb.XGBRegressor()
    xgb_weather.load_model(str(model_p))
    pred_test = np.load(pred_p)
    print("cache hit: model + predictions")
else:
    xgb_weather = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.0, reg_lambda=1.0,
        tree_method="hist", random_state=SEED, n_jobs=-1,
        early_stopping_rounds=50, eval_metric="rmse",
    )
    xgb_weather.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_test = np.clip(xgb_weather.predict(X_test), 0.0, 20.0)
    np.save(pred_p, pred_test)
    xgb_weather.save_model(str(model_p))
    print(f"trained: best_iter={xgb_weather.best_iteration}  "
          f"best_val_rmse={xgb_weather.best_score:.4f}")

# Aligned 5,075-hour window (drop first 24 rows, matching Phase 8 Cell 64)
SEQ_LEN = 24
y_test_aligned    = y_test[SEQ_LEN:]
pred_test_aligned = pred_test[SEQ_LEN:]
m_aligned = evaluate(y_test_aligned, pred_test_aligned)

baseline = {"RMSE": 0.4402, "MAE": 0.3008, "R2": 0.6156, "MAPE": 39.36}
cmp_df = pd.DataFrame([
    {"model": "XGBoost (original 69 features)",       "n_features": 69,  **baseline},
    {"model": "XGBoost (with weather 111 features)",  "n_features": 111, **m_aligned},
])
cmp_df.to_csv(OUT / "comparison.csv", index=False)

# --- Verification ---
assert m_aligned["RMSE"] < 0.60
print(f"aligned ({len(y_test_aligned)} rows): {m_aligned}")
for k in baseline:
    d = m_aligned[k] - baseline[k]; p = 100 * d / baseline[k]
    print(f"  Δ{k:<5s} = {d:+.4f}  ({p:+.2f}%)")
cmp_df
""")

md("""## Section 7 — SHAP analysis (TreeExplainer, 500 random test samples)

Identifies which of the 111 features dominate XGBoost's predictions, with the
weather features highlighted in two new colour groups
(`weather_lag` = teal, `weather_now` = red) to extend the original Cell 68 bar
chart.""")

code("""import shap

rng = np.random.RandomState(SEED)
n_sample = 500
sample_idx = np.sort(rng.choice(len(test), size=n_sample, replace=False))
X_sample_raw = test[feature_cols].iloc[sample_idx].to_numpy()
X_sample = scaler.transform(X_sample_raw)

explainer = shap.TreeExplainer(xgb_weather)
shap_values = explainer.shap_values(X_sample)
mean_abs = np.abs(shap_values).mean(axis=0)
order = np.argsort(mean_abs)[::-1]
ranking = [{"feature": feature_cols[i], "mean_abs_shap": float(mean_abs[i])}
           for i in order]
with open(OUT / "shap_weather_features.json", "w", encoding="utf-8") as f:
    json.dump({"sample_size": n_sample, "ranking": ranking}, f, indent=2)

def colour_for(name: str) -> str:
    if name.endswith("_now"):                                return "#d62728"
    if any(name.startswith(v + "_lag_") for v in WEATHER_VARS): return "#17becf"
    if "_lag_" in name:                                       return "#1f77b4"
    if "rmean" in name or "rstd" in name:                    return "#ff7f0e"
    return "#2ca02c"

def category_for(name: str) -> str:
    if name.endswith("_now"):                                return "weather_now"
    if any(name.startswith(v + "_lag_") for v in WEATHER_VARS): return "weather_lag"
    if "_lag_" in name:                                       return "lag"
    if "rmean" in name or "rstd" in name:                    return "rolling"
    return "time"

top_n = 20
top = ranking[:top_n][::-1]
top_feats  = [r["feature"] for r in top]
top_values = [r["mean_abs_shap"] for r in top]

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(top_feats, top_values, color=[colour_for(n) for n in top_feats])
ax.set_xlabel("mean(|SHAP value|)")
ax.set_title(f"Top {top_n} features by mean |SHAP|  (XGBoost + weather, n={n_sample} samples)")
ax.grid(axis="x", alpha=0.3)
legend_keys = [
    ("lag features (UCI)", "#1f77b4"),
    ("rolling stats",      "#ff7f0e"),
    ("time features",      "#2ca02c"),
    ("weather lag",        "#17becf"),
    ("weather current-hour","#d62728"),
]
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=c) for _, c in legend_keys],
          [n for n, _ in legend_keys], loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "shap_weather_bar.png", dpi=150)
plt.show()

top20 = ranking[:20]
weather_top20 = [r["feature"] for r in top20
                  if category_for(r["feature"]) in ("weather_lag", "weather_now")]
assert len(ranking) == 111
if not weather_top20:
    print("WARNING: no weather feature in top 20 - check merge alignment")
print(f"weather features in top 20: {len(weather_top20)}")
print("\\nTop 10:")
for i, r in enumerate(ranking[:10], 1):
    print(f"  {i:2d}. {r['feature']:<35s}  mean|SHAP|={r['mean_abs_shap']:.4f}  [{category_for(r['feature'])}]")
""")

md("""## Section 8 — Improvement report + side-by-side comparison plot

Final per-metric comparison table and dual-axis RMSE/R² plot mirroring the
style of `outputs/figures/03_model_comparison.png`. The interpretation tier is
auto-selected from the actual ΔR² value.""")

code("""delta = {k: m_aligned[k] - baseline[k] for k in baseline}
pct   = {k: 100.0 * delta[k] / baseline[k] for k in baseline}

d_r2 = delta["R2"]
if d_r2 > 0.05:
    interp = ("Weather features provide significant accuracy improvement. "
              "Temperature- and radiation-driven load components are now captured "
              "in addition to the autoregressive signal.")
elif d_r2 > 0.02:
    interp = ("Weather features provide moderate accuracy improvement. "
              "Adds value on top of the autoregressive lag structure but the "
              "latter remains dominant.")
else:
    interp = ("Weather features provide marginal improvement; the dominant signal "
              "remains autoregressive lag structure of the household load itself.")

top_w = [r for r in ranking
         if any(r["feature"].startswith(v + "_") or r["feature"] == v + "_now"
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
lines.append(f"RMSE (kW)    | {baseline['RMSE']:<18.4f} | {m_aligned['RMSE']:<18.4f} | {delta['RMSE']:+8.4f} | {pct['RMSE']:+7.2f}%")
lines.append(f"MAE  (kW)    | {baseline['MAE']:<18.4f} | {m_aligned['MAE']:<18.4f} | {delta['MAE']:+8.4f} | {pct['MAE']:+7.2f}%")
lines.append(f"R^2          | {baseline['R2']:<18.4f} | {m_aligned['R2']:<18.4f} | {delta['R2']:+8.4f} | {pct['R2']:+7.2f}%")
lines.append(f"MAPE (%)     | {baseline['MAPE']:<18.4f} | {m_aligned['MAPE']:<18.4f} | {delta['MAPE']:+8.4f} | {pct['MAPE']:+7.2f}%")
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
report_text = "\\n".join(lines)
(OUT / "improvement_report.txt").write_text(report_text, encoding="utf-8")

# --- Side-by-side comparison plot (dual-axis, same style as Cell 68) -------
plot_p = OUT / "model_comparison_weather.png"
cmp_plot_df = pd.DataFrame([
    {"Model": "XGBoost (69 features)",   "RMSE": baseline["RMSE"],   "R2": baseline["R2"]},
    {"Model": "XGBoost + weather (111)", "RMSE": m_aligned["RMSE"],  "R2": m_aligned["R2"]},
])
fig, ax1 = plt.subplots(figsize=(10, 6))
xs = np.arange(len(cmp_plot_df))
bars = ax1.bar(xs, cmp_plot_df["RMSE"], color="#4c72b0", alpha=0.85)
ax1.set_xticks(xs)
ax1.set_xticklabels(cmp_plot_df["Model"], rotation=10, ha="right")
ax1.set_ylabel("RMSE (kW)", color="#4c72b0")
ax1.tick_params(axis="y", labelcolor="#4c72b0")
ax1.set_ylim(0, cmp_plot_df["RMSE"].max() * 1.25)
for b, v in zip(bars, cmp_plot_df["RMSE"]):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
ax2 = ax1.twinx()
ax2.plot(xs, cmp_plot_df["R2"], color="#dd8452", marker="o", linewidth=2)
for x, v in zip(xs, cmp_plot_df["R2"]):
    ax2.text(x, v + 0.01, f"{v:.4f}", ha="center", fontsize=10, color="#dd8452")
ax2.set_ylabel("R^2", color="#dd8452")
ax2.tick_params(axis="y", labelcolor="#dd8452")
ax2.set_ylim(min(cmp_plot_df["R2"]) - 0.10, 1.0)
ax1.set_title("XGBoost accuracy improvement with weather features")
ax1.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(plot_p, dpi=150)
plt.show()

# --- Final 14-artifact existence assertion (mirrors brief lines 383-397) ---
expected = [
    "weather_raw.parquet", "hourly_merged.parquet", "features_weather.parquet",
    "splits/train_w.parquet", "splits/val_w.parquet", "splits/test_w.parquet",
    "scaler_weather.joblib", "pred_xgb_weather.npy", "xgb_weather_model.json",
    "shap_weather_features.json", "shap_weather_bar.png", "comparison.csv",
    "improvement_report.txt", "model_comparison_weather.png",
]
missing = [p for p in expected if not (OUT / p).exists()]
assert not missing, f"missing artifacts: {missing}"
print("ALL OUTPUT FILES VERIFIED")
print()
print(report_text)
""")

NB["cells"] = cells
nb_path = Path("weather_extension.ipynb")
nbf.write(NB, nb_path)
print(f"wrote {nb_path}  ({len(cells)} cells)")
