# PHASE 02 — Feature Engineering
# Input:  outputs/data/hourly_clean.parquet
# Output: outputs/data/features.parquet
# Notebook section: "Phase 2: Feature Engineering"

---

## WHAT THIS PHASE DOES

Builds a leakage-free 69-feature matrix from the clean hourly data and saves it
as a parquet file. No models. No splits. No training. Feature engineering only.

---

## DOMAIN KNOWLEDGE FOR THIS PHASE

**Why 69 features, and how they are counted:**

Lag features: 7 columns (target + 6 covariates) × 8 lag values = 56 features
Rolling features: 3 windows × 2 statistics (mean, std) = 6 features
Time features: hour_sin, hour_cos, dow_sin, dow_cos, mon_sin, mon_cos, is_weekend = 7 features
Total: 56 + 6 + 7 = 69 features

The previous version (diploma_v2.pdf) stated "48 features" — that was wrong.
The correct count is 69. The thesis must state 69.

**Target variable:**
Global_active_power. This column stays in the feature DataFrame alongside the 69
features (it is the label). It is NEVER included as one of the 69 feature columns.

**Covariates (variables used to build lag features alongside the target):**
Global_reactive_power, Voltage, Global_intensity,
Sub_metering_1, Sub_metering_2, Sub_metering_3

**Lag values:** 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h

The 168h (one week) lag is the single most important feature in this dataset.
Household consumption patterns are highly weekly-periodic — what happened last
Tuesday at 6pm is a strong predictor of this Tuesday at 6pm.
This is consistent with Pirbazari et al. (2020) who found weekly lags most predictive.

**Rolling windows:** 3h, 6h, 24h (mean and std of target only)

Rolling statistics capture the recent consumption trend and volatility.
The 24h rolling mean captures the "normal level" for this household.
Rolling std captures whether this is a stable or variable period.

**Cyclical time encodings:**
Raw hour-of-day (0–23) is a discontinuous integer — the model sees 0 and 23 as far
apart when they are adjacent. Sin/cos encoding maps them onto a circle, preserving
continuity. This is standard in the literature: Teslyuk et al. (2025), Salman (2026),
Han & Wang (2023) all use cyclical time features.

Formula: feature_sin = sin(2π × k / K), feature_cos = cos(2π × k / K)
where k is the raw value and K is the period (24 for hour, 7 for day-of-week, 12 for month).

**is_weekend:**
Binary flag: 1 if day-of-week is Saturday or Sunday, 0 otherwise.
Household consumption is systematically higher on weekends (people are home).
Moosbrugger et al. (2025) and Almughram et al. (2021) both note this pattern.

---

## THE LEAKAGE PROBLEM — CRITICAL FOR THIS PHASE

**What leakage means here:**
A feature at row t has leakage if it uses any information from time t or later.
If a model sees the target value at t as a feature while predicting t,
it trivially achieves near-zero error — this is not real performance.

**Why leakage is subtle in rolling windows:**
`df[TARGET].rolling(3).mean()` at row t computes the mean of rows t-2, t-1, t.
This INCLUDES t itself → leakage.

The correct form: `df[TARGET].shift(1).rolling(3).mean()` at row t computes
the mean of rows t-3, t-2, t-1 → no leakage.

**Cyclical time features are leakage-free:**
`hour_sin = sin(2π × index.hour / 24)` encodes WHEN this row is, not WHAT its
value is. Knowing it is 6pm does not tell you the consumption at 6pm.

**Lag features are trivially leakage-free:**
`df[TARGET].shift(h)` at row t gives the value at t-h.
For h ≥ 1, this is strictly in the past.

**The leakage verification that must be automated:**
At a specific row i, the lag_1h feature must exactly equal TARGET at row i-1.
This is a runtime-testable assertion that catches implementation errors.

---

## WHAT TESTS MUST PASS (write these first — TDD)

Tests go in `tests/test_phase02.py` AND as a test cell in the notebook.
Tests load `outputs/data/features.parquet` — they do not use in-memory variables.

**Structure tests:**
- File exists at expected path
- Loads as DataFrame with DatetimeIndex
- Has exactly 70 columns (69 features + TARGET)
- TARGET column is present
- All 69 expected feature names are present (check by name pattern, not position)
- Zero NaN values anywhere

**Feature count tests:**
- Exactly 56 lag feature columns (7 variables × 8 lags)
- Exactly 6 rolling feature columns (3 windows × 2 stats)
- Exactly 7 time feature columns

**Leakage tests — these are the most important:**
- For a random sample of 100 rows, verify that `{TARGET}_lag_1h` equals `TARGET`
  from the previous row (within floating-point tolerance)
- For a random sample of 100 rows, verify that `{TARGET}_lag_168h` equals `TARGET`
  from 168 rows earlier
- For a random sample of 100 rows, verify that `{TARGET}_rmean_3h` equals the mean
  of `{TARGET}_lag_1h`, `{TARGET}_lag_2h`, `{TARGET}_lag_3h` (within tolerance)
  — this confirms the rolling window is past-only
- Verify that TARGET itself is NOT in the list of feature columns

**Size test:**
- Row count is between 33,000 and 35,000 (first 168 rows dropped due to deepest lag)

**Value range tests:**
- All lag features of Global_active_power are ≥ 0
- hour_sin and hour_cos are in [-1, 1]
- is_weekend is binary (only values 0 and 1)

---

## WHAT TO SAVE

One file: `outputs/data/features.parquet`

This contains all 69 feature columns PLUS the TARGET column (70 columns total).
The TARGET column is the label — it stays in the DataFrame for convenient splitting
in Phase 3, but it is never used as a feature column during training.

Print in the notebook:
- Total feature count (must be 69)
- List all 69 feature names grouped by type (lag / rolling / time)
- Row count before and after dropna
- Confirmation that leakage assertions passed

---


