# PHASE 05 — Optuna Bayesian Hyperparameter Optimisation
# Input:  outputs/splits/train.parquet, val.parquet, test.parquet
#         outputs/predictions/pred_xgb.npy (default XGBoost result for comparison)
# Output: outputs/results/optuna_best_params.json
#         outputs/results/optuna_study_trials.csv
# Notebook section: "Phase 5: Optuna Hyperparameter Optimisation"

---

## WHAT THIS PHASE DOES

Runs a 40-trial Bayesian hyperparameter search over XGBoost using the Optuna
framework. Compares the tuned model to the default configuration from Phase 4.
This is a methodological validation — the finding is the comparison, not a new model.

---

## DOMAIN KNOWLEDGE FOR THIS PHASE

**Why Bayesian optimisation and not grid search:**

Grid search over 9 parameters is combinatorially infeasible.
Even with 3 values per parameter: 3^9 = 19,683 trials.

Bergstra & Bengio (2012) proved random search is provably more efficient than
grid search at the same computational budget — any randomly selected trial covers
more of the relevant hyperparameter space than a grid trial.

Bayesian optimisation (specifically TPE — Tree-structured Parzen Estimator) is
even better: it builds a probabilistic model of which hyperparameter regions are
likely to be good and concentrates evaluations there.
Akiba et al. (2019) introduced Optuna with TPE as an open-source framework.

Academic precedent:
Devanathan & Lakshmanan (2026) apply Optuna+GBM on residential data.
Salman et al. (2026) use Brown Bear Optimization (BBO) — a metaheuristic approach
conceptually similar to Bayesian optimisation.
Parizad et al. (2024) use Particle Swarm Optimization (PSO) and show 36.6% RMSE
improvement over untuned XGBoost on a different household dataset.

**Why 40 trials:**
The literature on Bayesian optimisation suggests that for spaces of 5-15 parameters,
20-50 trials is sufficient to find near-optimal configurations.
The Optuna documentation recommends 30-100 trials for this parameter count.
40 trials balances thoroughness with computational feasibility.

**The 9-parameter search space (from diploma_v2.pdf, confirmed correct):**
- n_estimators: number of trees [200, 1200] step 100
- max_depth: tree depth [3, 10]
- learning_rate: shrinkage [5e-3, 0.3] log-scale
- subsample: row sampling fraction [0.5, 1.0]
- colsample_bytree: column sampling fraction [0.5, 1.0]
- min_child_weight: minimum leaf weight [1, 10]
- reg_alpha: L1 regularisation [1e-3, 3.0] log-scale
- reg_lambda: L2 regularisation [1e-3, 6.0] log-scale
- gamma: minimum split gain [1e-6, 1.0] log-scale

**Why log-scale for learning_rate, regularisation, and gamma:**
These parameters span multiple orders of magnitude. Sampling uniformly in log-space
gives equal probability to [0.001, 0.01] and [0.01, 0.1] — each decade is equally
explored. Sampling uniformly in linear space would concentrate most evaluations near
the upper end of the range, missing the important small values.

**What the Optuna objective function does:**
1. Sample hyperparameters from the search space
2. Train XGBoost on X_train with those parameters
3. Use early stopping on X_val with patience=50 — this is CRITICAL (was missing in v1)
4. Evaluate RMSE on X_val
5. Return validation RMSE to Optuna (Optuna minimises this)

**Why early stopping must be inside the objective:**
Without early stopping, every trial trains for the full n_estimators regardless
of convergence. This wastes compute on over-fitted configurations and makes
the search less reliable. Adding early_stopping_rounds=50 inside the objective
makes each trial self-terminating when val RMSE stops improving.
This was a confirmed bug in v1.

**What to do with the best params after the study:**
Refit the best configuration on train+val combined (not train alone), then
evaluate once on the held-out test set. This is the standard Optuna protocol
and matches what Devanathan et al. (2026) describe.

**The expected finding:**
In v1, the Optuna-tuned model achieved test RMSE = 0.4348 kW while the default
achieved 0.4345 kW — a difference of 0.0003 kW (0.07%). This is statistically
negligible. The thesis interpretation: "the default XGBoost configuration is
already near-optimal for this engineered feature space. Further accuracy gains
require richer features, not finer hyperparameter search."
This is itself a meaningful methodological contribution — it falsifies the
common assumption that hyperparameter tuning always helps significantly.

**MedianPruner:**
Optuna's MedianPruner terminates underperforming trials early. After n_warmup_steps
trials, any trial performing worse than the median of completed trials is pruned.
This further concentrates compute on promising hyperparameter regions.

---

## WHAT TESTS MUST PASS (write these first — TDD)

Tests in `tests/test_phase05.py` AND a test cell in the notebook.

**File existence tests:**
- outputs/results/optuna_best_params.json exists and loads as valid JSON
- outputs/results/optuna_study_trials.csv exists

**Best params structure test:**
- JSON contains all 9 expected parameter keys
- n_estimators is an integer between 200 and 1200
- max_depth is an integer between 3 and 10
- learning_rate is a float between 5e-3 and 0.3
- subsample is a float between 0.5 and 1.0
- All other parameters are within their specified search ranges

**Study quality tests:**
- CSV contains at least 40 rows (40 completed trials)
- CSV has a 'value' column (validation RMSE per trial)
- Best validation RMSE in the study is less than 0.55 (sanity check)
- The study explored diverse regions: std of n_estimators across trials > 100

**Comparison test:**
- Load pred_xgb.npy from Phase 4 (default XGBoost test predictions)
- Compute RMSE of default on test set
- Compute RMSE of Optuna-tuned on test set
- Assert |Optuna RMSE - default RMSE| < 0.05 kW
  (the finding should be that Optuna barely helps — if it helps a lot, something
  is wrong with the Phase 4 default configuration)

**Refit correctness test:**
- The Optuna model was refit on train+val (combined), not train alone
- Verify this by checking that the tuned model's training data size equals
  len(train) + len(val)

---

## WHAT TO SAVE

- `outputs/results/optuna_best_params.json` — the 9 best hyperparameter values
- `outputs/results/optuna_study_trials.csv` — all 40 trial results for transparency

Print in notebook:
- Trial progress (show_progress_bar=True during study)
- Best validation RMSE achieved
- Default model validation RMSE (for comparison)
- Best parameter values in a clean table
- Test set comparison: default vs tuned RMSE, Δ RMSE, interpretation
- Visualisation: Optuna optimisation history plot (value vs trial number)

The Optuna optimisation history plot is a standard figure to include and shows
the commission that the search was thorough and convergent.
