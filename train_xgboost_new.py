"""
train_xgboost.py — Train XGBoost FOMO detector với Optuna hyperparameter tuning.

Pipeline:
    1. Load fomo_features.csv (đã gộp features + snorkel labels, dropped all-abstain)
    2. Tính sample_weight từ fomo_prob
    3. Train/test split 80/20
    4. Optuna tìm hyperparameters tốt nhất (minimize logloss trên val set)
    5. Train final model với best params
    6. Evaluate trên test set
    7. Export predictions + SHAP values ra CSV
    8. Save model

Input:    fomo_features.csv  — output của feature_engineering.py
          all-abstain đã được drop, fomo_prob là soft label từ Snorkel
          Features: behavioral + market context, KHÔNG có LF-derived features

Target:   fomo_prob (continuous [0,1]) — reg:logistic
Metric:   logloss + RMSE
Output:
    data/output/fomo_predictions.csv   — tx_id + fomo_score + fomo_level
    data/output/shap_values.csv        — SHAP per feature per transaction
    data/models/fomo_xgboost.json      — trained model

Chạy:
    python train_xgboost.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from constants import (
    OUTPUT_DIR,
    MODEL_DIR,
    XGBOOST_MODEL_FILE,
)

FEATURES_FILE    = f"{OUTPUT_DIR}/fomo_features.csv"
PREDICTIONS_FILE = f"{OUTPUT_DIR}/fomo_predictions.csv"
SHAP_FILE        = f"{OUTPUT_DIR}/shap_values.csv"

# ── Config ────────────────────────────────────────────────────────────────
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
OPTUNA_TRIALS    = 50
OPTUNA_TIMEOUT   = 300
WEIGHT_EPSILON   = 0.01

FOMO_HIGH_THRESH   = 0.65
FOMO_MEDIUM_THRESH = 0.40

# Columns không phải feature
NON_FEATURE_COLS = [
    "tx_id", "investor_id", "timestamp",
    "fomo_prob", "momentum_acceleration",  # momentum_acceleration 100% NaN
]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & Merge
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

df = pd.read_csv(FEATURES_FILE, parse_dates=["timestamp"])
print(f"  fomo_features: {df.shape}")
print(f"  Investors    : {df['investor_id'].nunique():,}")
print(f"  fomo_prob NaN: {df['fomo_prob'].isna().sum()}")
print(f"  fomo_prob mean: {df['fomo_prob'].mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Prepare features, target, sample weight
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Preparing features & weights...")
print("=" * 60)

feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
print(f"  Feature columns: {len(feature_cols)}")

X = df[feature_cols].copy()
y = df["fomo_prob"].values

# Sample weight: |fomo_prob - 0.5| * 2 + epsilon
# Rows gần 0.5 (uncertain/all-abstain) → weight thấp
# Rows gần 0 hoặc 1 (chắc chắn) → weight cao
sample_weight = np.abs(y - 0.5) * 2 + WEIGHT_EPSILON

print(f"  Target y — mean: {y.mean():.4f}, std: {y.std():.4f}")
print(f"  Sample weight   — min: {sample_weight.min():.3f}, max: {sample_weight.max():.3f}, mean: {sample_weight.mean():.3f}")

# Fill NaN còn sót lại (nếu có) bằng median
X = X.fillna(X.median())


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Train/Test Split
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Train/Test split...")
print("=" * 60)

# Lưu lại index để map predictions sau này
idx = np.arange(len(df))

(X_train, X_test,
 y_train, y_test,
 w_train, w_test,
 idx_train, idx_test) = train_test_split(
    X, y, sample_weight, idx,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"  Train: {len(X_train):,} rows")
print(f"  Test : {len(X_test):,} rows")
print(f"  Train fomo_prob mean: {y_train.mean():.4f}")
print(f"  Test  fomo_prob mean: {y_test.mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Optuna Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"STEP 4: Optuna tuning ({OPTUNA_TRIALS} trials, timeout={OPTUNA_TIMEOUT}s)...")
print("=" * 60)

# Tách val set từ train để Optuna evaluate
X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
    X_train, y_train, w_train,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

def objective(trial):
    params = {
        "objective"        : "reg:logistic",
        "eval_metric"      : "logloss",
        "tree_method"      : "hist",
        "random_state"     : RANDOM_STATE,
        "verbosity"        : 0,
        # Hyperparameters được tune
        "n_estimators"     : trial.suggest_int("n_estimators", 100, 800),
        "max_depth"        : trial.suggest_int("max_depth", 3, 8),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight" : trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma"            : trial.suggest_float("gamma", 0.0, 5.0),
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )

    preds = model.predict(X_val)
    preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
    # logloss với soft label
    loss = log_loss(
        (y_val > 0.5).astype(int),
        preds_clipped,
        sample_weight=w_val,
    )
    return loss

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(
    objective,
    n_trials=OPTUNA_TRIALS,
    timeout=OPTUNA_TIMEOUT,
    show_progress_bar=True,
)

best_params = study.best_params
best_val_loss = study.best_value
print(f"\n  Best val logloss: {best_val_loss:.6f}")
print(f"  Best params:")
for k, v in best_params.items():
    print(f"    {k:<22s}: {v}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Train Final Model trên toàn bộ train set
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Training final model on full train set...")
print("=" * 60)

final_params = {
    "objective"  : "reg:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "verbosity"  : 0,
    **best_params,
}

final_model = xgb.XGBRegressor(**final_params)
final_model.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_test, y_test)],
    sample_weight_eval_set=[w_test],
    verbose=50,
)
print("  Final model trained.")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Evaluate trên Test Set
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Evaluation on test set...")
print("=" * 60)

y_pred_test = final_model.predict(X_test)
y_pred_test = np.clip(y_pred_test, 1e-7, 1 - 1e-7)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_test, sample_weight=w_test))
test_logloss = log_loss(
    (y_test > 0.5).astype(int),
    y_pred_test,
    sample_weight=w_test,
)

print(f"  RMSE (weighted)    : {rmse:.6f}")
print(f"  LogLoss (weighted) : {test_logloss:.6f}")

# Distribution của predictions
print(f"\n  Prediction distribution:")
for lo, hi, lbl in [(0.0, 0.4, "Low FOMO   (< 0.40)"),
                    (0.4, 0.65,"Medium FOMO (0.40-0.65)"),
                    (0.65, 1.01,"High FOMO  (>= 0.65)")]:
    mask = (y_pred_test >= lo) & (y_pred_test < hi)
    print(f"    {lbl}: {mask.sum():,} ({mask.mean()*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — Predict toàn bộ dataset
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Predicting full dataset...")
print("=" * 60)

y_pred_all = final_model.predict(X)
y_pred_all = np.clip(y_pred_all, 1e-7, 1 - 1e-7)

def assign_fomo_level(score):
    if score >= FOMO_HIGH_THRESH:
        return "High"
    elif score >= FOMO_MEDIUM_THRESH:
        return "Medium"
    else:
        return "Low"

fomo_levels = [assign_fomo_level(s) for s in y_pred_all]

predictions_df = pd.DataFrame({
    "tx_id"             : df["tx_id"].values,
    "investor_id"       : df["investor_id"].values,
    "timestamp"         : df["timestamp"].values,
    "fomo_prob_snorkel" : y,
    "fomo_score"        : y_pred_all,
    "fomo_level"        : fomo_levels,
    "split"             : ["test" if i in set(idx_test) else "train" for i in range(len(df))],
})

print(f"  Total predictions: {len(predictions_df):,}")
print(f"  fomo_level distribution:")
print(predictions_df["fomo_level"].value_counts().to_string())


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — SHAP Values
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Computing SHAP values (sample 10,000 rows)...")
print("=" * 60)

# SHAP trên sample để tiết kiệm thời gian
shap_sample_size = min(10_000, len(X))
shap_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), size=shap_sample_size, replace=False)
X_shap = X.iloc[shap_idx].copy()

explainer   = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_shap)

shap_df = pd.DataFrame(shap_values, columns=feature_cols)
shap_df.insert(0, "tx_id", df["tx_id"].iloc[shap_idx].values)

# Feature importance summary
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=feature_cols,
).sort_values(ascending=False)

print(f"\n  Top 15 features by mean |SHAP|:")
print(mean_abs_shap.head(15).round(6).to_string())


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9 — Save outputs
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: Saving outputs...")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

predictions_df.to_csv(PREDICTIONS_FILE, index=False)
print(f"  ✓ Predictions : {PREDICTIONS_FILE}  ({predictions_df.shape})")

shap_df.to_csv(SHAP_FILE, index=False)
print(f"  ✓ SHAP values : {SHAP_FILE}  ({shap_df.shape})")

final_model.save_model(XGBOOST_MODEL_FILE)
print(f"  ✓ Model       : {XGBOOST_MODEL_FILE}")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Total BUY transactions : {len(df):,}")
print(f"  Features used          : {len(feature_cols)}")
print(f"  Optuna best val loss   : {best_val_loss:.6f}")
print(f"  Test RMSE (weighted)   : {rmse:.6f}")
print(f"  Test LogLoss (weighted): {test_logloss:.6f}")
print(f"\n  fomo_level breakdown (full dataset):")
for lvl in ["High", "Medium", "Low"]:
    n = (predictions_df["fomo_level"] == lvl).sum()
    print(f"    {lvl:<8}: {n:,} ({n/len(predictions_df)*100:.1f}%)")
print(f"\n  Top 5 FOMO features (SHAP):")
for feat, val in mean_abs_shap.head(5).items():
    print(f"    {feat:<40s}: {val:.6f}")
print("\n✓ Done. Next: visualize fomo_predictions.csv trên dashboard.")
