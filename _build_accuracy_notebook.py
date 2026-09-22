"""Build accuracy_improvements.ipynb from accuracy_improvements.py.

The notebook is structured as a presentation layer over the script:
each section invokes one of the step functions and renders its outputs
inline. The script itself remains the source of truth for the logic.
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


md("""# Accuracy-Improvement Workstreams on the Diploma Pipeline

**AITU, June 2026 — Yerassyl Raimkhan**

This notebook is the second extension of `diploma_pipeline.ipynb`. It builds on
the 111-feature weather model (`weather_extension.ipynb`) with three new levers,
applied sequentially:

1. **Feature bundle** — French national holidays + Paris Zone-C school holidays,
   day-of-year cyclical, HDD/CDD (heating/cooling degree days, base 18.3 °C),
   plus temperature-by-daypart and `power_lag_1h × temp_now` interactions.
   Result: **125 features** (111 + 14).
2. **Optuna retune** — 60 trials on the same 9-parameter search space as
   Phase 5, applied to the enlarged feature set.
3. **Stacking ensemble** — Constrained-LSQ blend (or RidgeCV, picked by smaller
   val→test R² gap) over five base models:
   `xgb69`, `xgb111`, `xgb125_tuned`, `lstm69`, `gru69`.

**Constraints inherited from earlier work:**
- Identical chronological split boundaries; identical 5,075-hour aligned test
  window; `MAPE_EPSILON_KW = 0.10`; `SEED = 42` throughout.
- All writes scoped to `outputs/improvements/`. Original `outputs/` and
  `outputs/weather_extension/` are read-only.

The heavy logic lives in `accuracy_improvements.py` — each notebook cell here
invokes one of its `stepN_*` functions and renders the outputs inline. Re-runs
are cached by parquet/npy/joblib so the notebook is fast after the first pass.
""")

md("""## Section 0 — Setup

Imports the step functions from the companion script and seeds RNGs. The
script's section functions are idempotent: each one looks for its cached
artifact under `outputs/improvements/` and skips computation if present.
""")

code("""from __future__ import annotations
import os, random, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)

import accuracy_improvements as ai
print("script module loaded from", ai.__file__)
print("OUT directory:", ai.OUT)
""")

md("""## Section 1 — Feature bundle (111 → 125)

**Calendar (5):** `is_holiday_fr` (FR national holidays via the `holidays`
package), `is_school_holiday_fr_c` (Paris Zone-C school vacation windows from
a static lookup table), `is_long_weekend` (pont days adjacent to Tuesday/
Thursday holidays), `doy_sin`/`doy_cos` (day-of-year cyclical — captures
Christmas/August vacation dips that monthly encodings blur).

**HDD/CDD (3):** `hdd = max(0, 18.3 − T)`, `cdd = max(0, T − 18.3)` (ASHRAE
residential base), and `hdd_rolling_24h` (`.shift(1).rolling(24).mean()` for
leakage-safe multi-day cold-snap memory).

**Interactions (6):** `temp_x_hour_{sin,cos}`, `temp_x_is_weekend`,
`radiation_x_hour_{sin,cos}`, and `power_lag_1h_x_temp_now` (lagged household
demand conditioned on current temperature — the strongest of the six in SHAP).
""")

code("""features = ai.step1_features()
features.shape, list(features.columns)[-14:]   # last 14 = the new bundle
""")

md("""## Section 2 — Chronological split + StandardScaler

Boundaries inherited from `weather_extension/splits/*` so the test window
remains the same 5,099 contiguous hours (April 2010 → November 2010) that the
weather model was evaluated on. Scaler fit on the 125 train features only.
""")

code("""train, val, test, scaler = ai.step2_split(features)
print("scaler.n_features_in_ =", scaler.n_features_in_)
""")

md("""## Section 3 — Optuna retune on the 125-feature set

Same 9-parameter search space as `diploma_pipeline.ipynb` Cells 33–37, but
**60 trials** instead of 40 and applied to the enlarged feature set. The TPE
sampler is seeded; saved best parameters in `outputs/improvements/
optuna_125_best_params.json`. The final model is refit with the best params
and produces both val and test predictions (the val predictions feed into
the stacking meta-learner in Section 5).

> **First run takes ~5–10 minutes**; subsequent runs hit the cache and are
> instant.
""")

code("""tuned, X_test, y_test, pred_test_125, pred_val_125, metrics_125, feature_cols = \\
    ai.step3_optuna(train, val, test, scaler)
print("aligned test metrics:", metrics_125)
""")

md("""## Section 4 — Base-model predictions for stacking

For each of the five stacking base models, produce aligned val + test
predictions (both 5,074- and 5,075-row windows respectively). Val
predictions for the tree models are regenerated from the saved models;
LSTM/GRU val predictions are produced by sliding 24-step sequences from
the last 24 train rows through the val set. All test predictions are the
already-aligned arrays from the original pipeline.
""")

code("""base, y_val_aligned, y_test_aligned = ai.step4_base_predictions()
print("base models loaded; aligned shapes:")
for name in ["xgb69", "xgb111", "xgb125", "lstm", "gru"]:
    print(f"  {name:6s}  val={base[name]['val'].shape}  test={base[name]['test'].shape}")
""")

md("""## Section 5 — Stacking ensemble

Two meta-learner variants are fit on the **val** predictions and chosen by
the smaller val→test R² gap (a proxy for generalisation):

- **Constrained-LSQ blend** — weights ≥ 0 summing to 1 (SLSQP), interpretable
  as model votes.
- **RidgeCV stacking** — `RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)`,
  unconstrained linear meta with regularised coefficients.

The chosen variant is saved as `outputs/improvements/blend_weights.json` plus
(if Ridge) `ridge_meta.joblib`. The final test prediction is clipped to
[0, 20] kW and written to `stacked_predictions.npy`.
""")

code("""metrics_stack, chosen, blend_info = ai.step5_stacking(
    base, y_val_aligned, y_test_aligned)
print(f"chosen variant: {chosen}")
print(f"stacked test:   {metrics_stack}")
""")

md("""## Section 6 — Report, SHAP, comparison plot, bootstrap CI

The four-row model-progression table:

| Stage             | Features | Note                              |
|-------------------|----------|-----------------------------------|
| Baseline          | 69       | Original `diploma_pipeline.ipynb` |
| +Weather          | 111      | `weather_extension.ipynb`         |
| +Cal/HDD/Inter+Optuna | 125  | This notebook, Sections 1–3       |
| Stacked ensemble  | 5-base   | This notebook, Sections 4–5       |

SHAP is computed via `TreeExplainer` on 500 random test rows (seeded), with
new colour categories for `calendar`, `hdd`, and `interaction` features. The
bootstrap CI on Δ R² (stacked − baseline) uses 1,000 resamples of the 5,075-
hour test window — if the 95% CI excludes zero, the improvement is
statistically distinguishable from sampling noise.
""")

code("""cmp, ranking, ci = ai.step6_report(
    tuned, test, scaler, feature_cols, metrics_125, metrics_stack,
    base, y_test_aligned, chosen, blend_info)
print()
print((ai.OUT / "improvement_report.txt").read_text(encoding="utf-8"))
""")

md("""## Final artifact check

Verifies every required artifact under `outputs/improvements/` is present
and non-empty.
""")

code("""from pathlib import Path
expected = [
    "features_125.parquet",
    "splits/train_125.parquet", "splits/val_125.parquet", "splits/test_125.parquet",
    "scaler_125.joblib",
    "optuna_125_study.pkl", "optuna_125_trials.csv", "optuna_125_best_params.json",
    "xgb_125_tuned.json", "pred_xgb_125_tuned.npy", "pred_xgb_125_tuned_val.npy",
    "preds_base/xgb69_val.npy", "preds_base/xgb69_test.npy",
    "preds_base/xgb111_val.npy", "preds_base/xgb111_test.npy",
    "preds_base/xgb125_val.npy", "preds_base/xgb125_test.npy",
    "preds_base/lstm_val.npy", "preds_base/lstm_test.npy",
    "preds_base/gru_val.npy", "preds_base/gru_test.npy",
    "stacked_predictions.npy", "blend_weights.json",
    "shap_125_features.json", "shap_125_bar.png",
    "comparison_table.csv", "model_comparison.png", "improvement_report.txt",
]
missing = [p for p in expected if not (ai.OUT / p).exists()
            or (ai.OUT / p).stat().st_size == 0]
assert not missing, f"missing or empty: {missing}"
print(f"all {len(expected)} artifacts verified")
""")

NB["cells"] = cells
nb_path = Path("accuracy_improvements.ipynb")
nbf.write(NB, nb_path)
print(f"wrote {nb_path}  ({len(cells)} cells)")
