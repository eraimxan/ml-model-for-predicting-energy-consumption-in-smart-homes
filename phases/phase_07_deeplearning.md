# PHASE 07 — Deep Learning: LSTM and GRU
# Input:  outputs/splits/train.parquet, val.parquet, test.parquet
#         outputs/models/scaler.joblib
# Output: outputs/predictions/pred_lstm.npy
#         outputs/predictions/pred_gru.npy
#         outputs/models/lstm_model.keras
#         outputs/models/gru_model.keras
#         outputs/results/lstm_history.json
#         outputs/results/gru_history.json
# Notebook section: "Phase 7: Deep Learning (LSTM and GRU)"

---

## WHAT THIS PHASE DOES

Builds multivariate input sequences from the 69-feature matrix, trains a 2-layer
LSTM and a 2-layer GRU, evaluates both on the test set, and saves models,
predictions, and training histories. This is the most technically complex phase.

---

## DOMAIN KNOWLEDGE FOR THIS PHASE

**Why multivariate sequences — the key correction from v1:**

v1 fed LSTM a 1-dimensional sequence of raw Global_active_power values
while XGBoost received all 69 engineered features. This is an unfair comparison —
LSTM was information-starved relative to XGBoost.

The correct approach: build sequences from the same 69-feature matrix that
XGBoost and RF use. Each sequence at time t contains the 69-feature rows
from t-SEQ_LEN to t-1 (the past 24 hours of all features). The target is y at time t.

This makes the comparison fair: all models receive the same information,
just in different formats (tabular for trees, sequence for DL).

**Why SEQ_LEN = 24:**

24 hours captures the full diurnal cycle — the most important seasonal pattern
in residential consumption. The 24h lag is the strongest single predictor
in XGBoost feature importance (from the literature and expected from v1 results).
Han & Wang (2023) use 24-step sequences on Irish CER data.

**The sequence construction and its leakage proof:**

Sequence at position i:
  X[i] = feature_matrix[i - SEQ_LEN : i]  →  shape (24, 69)
  y[i] = target[i]

feature_matrix row at position j contains:
  lag_1h = target[j-1]
  lag_2h = target[j-2]
  ...
  rolling_mean_3h = mean of target[j-3], target[j-2], target[j-1]
  hour_sin = sin(2π × hour_of_day / 24)

Therefore feature_matrix[j] contains NO information from time j or later.
The sequence X[i] = feature_matrix[i-24:i] contains rows j = i-24 to j = i-1.
None of these rows contain information from time i or later.
Target y[i] is the value AT time i.
Conclusion: NO LEAKAGE.

**Why the scaler from Phase 3 must be reused (not refitted):**

The StandardScaler was fitted on X_train only in Phase 3 and saved.
In Phase 7, load that scaler and apply it to X_train, X_val, X_test.
Do NOT refit the scaler on any new data in Phase 7.

This ensures: (a) consistency — the same transformation applied throughout,
(b) no leakage — the scaler parameters were derived from X_train only.

**Sequence split — critical:**

Build sequences SEPARATELY from the already-split train/val/test sets.
Do NOT build one giant sequence array and then split by index — this would
create overlap at the boundaries (the last SEQ_LEN rows of train would appear
in the val sequences as input context).

Correct approach:
  train_seq: sequences from X_train_scaled, y_train
  val_seq:   sequences from X_val_scaled, y_val
  test_seq:  sequences from X_test_scaled, y_test

The first SEQ_LEN rows of each split cannot form a complete sequence
(they have insufficient history). They are dropped. This means len(test_seq)
= len(test) - SEQ_LEN. When comparing DL predictions to tree predictions on
the test set, use only the rows of the test set that have DL predictions.

**LSTM architecture — from the literature:**

2-layer stacked LSTM is the standard architecture in recent residential STLF:
Han & Wang (2023) use 3 layers with 64 units.
Teslyuk et al. (2025) use 1 layer with 100 units.
Hammou Ou Ali (2024) use CNN-LSTM hybrid.
Salman et al. (2026) use LSTM + GRU both with 200 units.

A 2-layer architecture (128 → 64 units) balances capacity and regularisation
for a dataset of ~24,000 training rows. More layers risk overfitting.

**GRU architecture:**

GRU (Gated Recurrent Unit, Cho et al. 2014) is a simplified LSTM that uses
only reset gate and update gate (no separate cell state).

Mathematical comparison:
LSTM: 4 gates (forget, input, output, cell update) → 4 × (n² + n×m) parameters
GRU:  2 gates (reset, update) + 1 candidate → 3 × (n² + n×m) parameters
GRU is approximately 25% fewer parameters than equivalent LSTM.

On small datasets, fewer parameters can reduce overfitting.
Salman et al. (2026) show GRU RMSE = 0.4921 vs LSTM RMSE = 0.4909 on Kaggle
smart home data — essentially identical. The comparison establishes which
architecture is more suitable for this specific dataset size and feature space.

**EarlyStopping and ReduceLROnPlateau:**

EarlyStopping: stop training when val_loss stops improving.
patience=10 means the model trains for 10 additional epochs after the best
val_loss before stopping. restore_best_weights=True reverts to the epoch with
the best validation loss, not the last epoch.

ReduceLROnPlateau: when val_loss stops improving for 5 epochs, multiply
the learning rate by 0.5. This allows the model to initially train with a
large learning rate (fast convergence) and then fine-tune with a smaller one.

Both callbacks are standard in the deep learning for energy forecasting literature.
Teslyuk et al. (2025) use early stopping with their LSTM.

**Batch size and epochs:**

batch_size=64: standard for this dataset size. Larger batches are faster but
may converge to sharper minima (worse generalisation).
epochs=100: a ceiling, not a target — early stopping will likely trigger much earlier.
The number of epochs actually trained is reported via the training history.

**Random seed for reproducibility:**

TensorFlow/Keras training involves randomness from:
- Weight initialisation
- Dropout masks
- Mini-batch shuffling

Setting tf.random.set_seed(42) and np.random.seed(42) before model instantiation
and training makes results reproducible across runs on the same hardware.

---

## WHAT TESTS MUST PASS (write these first — TDD)

Tests in `tests/test_phase07.py` AND a test cell in the notebook.

**File existence tests:**
- outputs/predictions/pred_lstm.npy exists
- outputs/predictions/pred_gru.npy exists
- outputs/models/lstm_model.keras exists
- outputs/models/gru_model.keras exists
- outputs/results/lstm_history.json exists (contains loss curves)
- outputs/results/gru_history.json exists

**Prediction shape tests:**
- len(pred_lstm) == len(test) - SEQ_LEN  (SEQ_LEN rows dropped from test sequences)
- len(pred_gru) == len(pred_lstm)
- No NaN values in either prediction
- No infinite values

**Prediction range tests:**
- All LSTM predictions ≥ 0
- All GRU predictions ≤ 20 kW

**Training history tests:**
- lstm_history has 'loss' and 'val_loss' keys
- Number of epochs trained is between 10 and 100 (early stopping fired)
- val_loss at best epoch is less than val_loss at epoch 1
  (the model actually learned something)

**Model architecture tests:**
- Load lstm_model.keras and check:
  - It has exactly 6 layers (LSTM → Dropout → LSTM → Dropout → Dense → Dense)
  - Input shape matches (SEQ_LEN, 69)
  - Output shape matches (1,) for regression
- Same checks for gru_model.keras

**Metric sanity tests:**
- Compute RMSE of LSTM on test_seq targets
- Compute RMSE of GRU on test_seq targets
- Both RMSE values are between 0.3 and 0.75 (sanity range)
- LSTM and GRU RMSE values differ by less than 0.1 kW
  (they use the same features/sequences — large difference suggests a bug)

**Fairness test — the key correction from v1:**
- Verify that X_test_seq has shape (n, SEQ_LEN, 69) — not (n, SEQ_LEN, 1)
  Shape (n, SEQ_LEN, 1) would mean only the raw target was used (v1 bug)
  Shape (n, SEQ_LEN, 69) means all 69 features were used (correct)

---

## WHAT TO SAVE

Predictions:
- `outputs/predictions/pred_lstm.npy`
- `outputs/predictions/pred_gru.npy`

Models:
- `outputs/models/lstm_model.keras`
- `outputs/models/gru_model.keras`

Training histories (for plot in Phase 8):
- `outputs/results/lstm_history.json` — {'loss': [...], 'val_loss': [...]}
- `outputs/results/gru_history.json` — {'loss': [...], 'val_loss': [...]}

Print in notebook:
- LSTM model summary (layer sizes, parameter counts)
- GRU model summary
- Training: epochs trained, best val_loss epoch, final train loss, final val loss
- Test set metrics for LSTM and GRU (RMSE, MAE, R², MAPE)
- Note: DL metrics computed on test_seq (len = len(test) - SEQ_LEN)
  while tabular metrics from Phase 4 use full test set — acknowledge this difference
