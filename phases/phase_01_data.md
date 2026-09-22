# PHASE 01 — Data Loading and Cleaning
# Input:  household_power_consumption.txt
# Output: outputs/data/hourly_clean.parquet
# Notebook section: "Phase 1: Data Loading and Cleaning"

---

## WHAT THIS PHASE DOES

Loads the raw UCI dataset, handles missing values, resamples to hourly resolution,
and saves a clean hourly parquet file. Nothing else. No features. No splits. No models.

---

## DOMAIN KNOWLEDGE FOR THIS PHASE

**Dataset origin:**
Hébrail, G. & Berard, A. (2012). Individual household electric power consumption.
UCI Machine Learning Repository. This is the de facto benchmark dataset for
residential STLF — used by Hammou Ou Ali (2024), Salman et al. (2026),
Han & Wang (2023), Olu-Ajayi & Alaka (2022), and many others.

**Why the data is the way it is:**
The data was collected from a single French household near Paris using a smart meter.
The missing values (~1.25%) occur in short clustered runs that the original authors
attribute to communication losses between the meter and the logging system.
These are not random — they tend to cluster in a few multi-hour outages spread
across the 4-year collection period.

**Why hourly and not minute-level:**
Minute-level data is too noisy for a 1-hour-ahead HEMS forecast and too expensive
to train deep models on. The literature consistently resamples to hourly:
Teslyuk et al. (2025), Hammou Ou Ali (2024), Han & Wang (2023) all use hourly.
The standard approach: average rate variables (power, voltage, current),
sum energy variables (sub-metering channels).

**Why sub-metering channels are summed, not averaged:**
Sub_metering_1/2/3 are in Wh (watt-hours) — they represent energy consumed in each
minute. When aggregating to hourly, the total energy consumed in the hour is the SUM
of the per-minute values. Averaging would understate consumption by 60×.

**Why rate variables (active_power, voltage, intensity) are averaged:**
Global_active_power is in kW — it is a rate (power at that instant). The hourly
average power is the arithmetic mean of the per-minute readings, which correctly
represents the average load level during the hour.

**Missing value treatment from the literature:**
The standard approach (used in diploma_v2.pdf) is linear interpolation for short gaps
(≤ 6 consecutive minutes) and dropping rows for longer gaps. This is defensible
because: (a) 6 minutes represents 10% of an hour — acceptable for a 1-hour forecast,
(b) longer gaps indicate actual data loss, not communication glitches.

**Quality filter on hourly rows:**
Any hourly row built from fewer than 30 minute-level observations should be dropped.
This prevents hours where more than half the data was missing from appearing as if
they were fully observed. Mehdipour Pirbazari et al. (2020) discuss this kind of
data quality filtering for smart meter studies.

**Expected outputs after cleaning:**
- Approximately 34,000–35,000 hourly rows
- Date range: 16 December 2006 → 26 November 2010
- Zero NaN values in the output
- All values non-negative (negative power is physically impossible for this meter)
- Index: DatetimeIndex at hourly frequency, monotonically increasing

---

## WHAT TESTS MUST PASS (write these first — TDD)

Tests go in `tests/test_phase01.py` AND as a test cell in the notebook.
Tests load `outputs/data/hourly_clean.parquet` — they do not use in-memory variables.

The following properties must be verified:

**Structure tests:**
- The output file exists at the expected path
- It loads as a DataFrame with a DatetimeIndex
- It has exactly 7 columns (the 7 measurement columns, no Date/Time columns)
- Column names match the exact names from the raw file

**Data quality tests:**
- Zero NaN values anywhere in the DataFrame
- All values in Global_active_power are ≥ 0
- All values in Sub_metering_1/2/3 are ≥ 0
- Index is monotonically increasing
- Index frequency is hourly (or close to it — gaps from dropped rows are acceptable)

**Size tests:**
- Row count is between 30,000 and 36,000 (sanity range for this dataset)
- The earliest timestamp is in December 2006
- The latest timestamp is in November 2010

**Resampling correctness tests:**
- Pick a specific hour that is fully observed in the raw data (verify manually first)
- Assert that Global_active_power for that hour in the output equals the mean of
  the corresponding minute-level rows in the raw data (within floating-point tolerance)
- Assert that Sub_metering_1 for that hour equals the sum of the corresponding
  minute-level rows (within floating-point tolerance)

---

## EDA PLOTS FOR THIS PHASE

The notebook should include exploratory plots after the cleaning cell.
These plots become Figure 9 in the final outputs and a thesis EDA section.

**Plot A: Full time series of Global_active_power (hourly)**
Shows the complete 4-year series. The reader immediately sees seasonality,
weekly patterns, and the variance of the target variable.

**Plot B: Average consumption by hour of day**
Bar chart across 24 hours. Should show clear diurnal pattern:
low overnight (~0.3 kW), morning peak (~1.5 kW), evening peak (~1.8 kW).
This motivates the hour_sin/cos cyclical features in Phase 2.

**Plot C: Average consumption by day of week**
Bar chart across 7 days. Should show higher consumption on weekends.
This motivates the dow_sin/cos and is_weekend features in Phase 2.

**Plot D: Missing value timeline**
A timeline showing WHERE the missing data occurred across the 4-year period.
This shows the commission that you understood the data quality issue,
not just mechanically filled it.

**Plot E: Distribution of Global_active_power**
Histogram with log scale on y-axis. Shows the heavy right tail and near-zero mass
overnight. This motivates the safe_mape with epsilon guard in Phase 4+.

Save all plots to `outputs/figures/09_eda_patterns.png` (composite figure).

---

## WHAT TO SAVE

After all tests pass, save exactly one file:
`outputs/data/hourly_clean.parquet`

This parquet file is the single source of truth for all downstream phases.
Phases 2–8 load this file — they never touch the raw txt file.

Also print a cleaning report in the notebook with:
- Raw row count
- Missing value count and percentage
- Rows dropped (gap > 6 min)
- Rows dropped (< 30 min observations per hour)
- Final hourly row count
- Date range of output
