"""
train_xgboost_new.py — Train XGBoost FOMO detector với Optuna hyperparameter tuning.

Split strategy: Regime-based time split
    Bull bucket  = Bull data + Sideways nửa đầu → time split 80/20
    Bear bucket  = Bear data + Sideways nửa sau → time split 80/20
    Train = Bull train + Bear train
    Test  = Bull test  + Bear test

    Optuna val:
    Bull val = 20% cuối Bull train
    Bear val = 20% cuối Bear train
    → Optuna val = Bull val + Bear val (nhất quán với regime split)

Chạy:
    python train_xgboost_new.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
from sklearn.metrics import (
    mean_squared_error, log_loss,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, fbeta_score
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from constants import OUTPUT_DIR, MODEL_DIR, XGBOOST_MODEL_FILE

FEATURES_FILE    = f"{OUTPUT_DIR}/fomo_features.csv"
PREDICTIONS_FILE = f"{OUTPUT_DIR}/fomo_predictions.csv"
SHAP_FILE        = f"{OUTPUT_DIR}/shap_values.csv"

# ── Config ────────────────────────────────────────────────────────────────
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
OPTUNA_TRIALS    = 50
OPTUNA_TIMEOUT   = 300
WEIGHT_EPSILON   = 0.01
CV_FOLDS         = 5

FOMO_HIGH_THRESH   = 0.71
FOMO_LOW_THRESH    = 0.35
FOMO_MEDIUM_THRESH = 0.40

# ── Market regime boundaries ──────────────────────────────────────────────
BULL_START     = '2020-11-01'
BULL_END       = '2021-06-30'
SIDEWAYS_START = '2021-07-01'
SIDEWAYS_END   = '2022-01-31'
BEAR_START     = '2022-02-01'
BEAR_END       = '2022-07-31'

NON_FEATURE_COLS = [
    "tx_id", "investor_id", "timestamp",
    "fomo_prob", "momentum_acceleration",
    # ── DROP leakage ──────────────────────────────────────────────
    "trade_gap_days",
    "total_value_pctrank_asset",
    "rolling_avg_position_size_last_10",
    "position_size_to_volatility_ratio",
    "position_size_ratio",
    "trades_per_investor_per_day",
    "same_day_multiple_flag",
    "return_1d",
    "market_fomo_pressure_score",
    "asset_popularity_zscore",
    # ── DROP redundant ────────────────────────────────────────────
    "volatility_regime",
    "rolling_trade_freq_5",
    # ── DROP leakage — biến mới ───────────────────────────────────
    "price_distance_high",
    "macd",
    "macd_hist",
    "macd_signal",
]


def time_split_bucket(bucket, test_size=TEST_SIZE):
    """Split 1 bucket theo thời gian — 80% đầu train, 20% cuối test."""
    bucket = bucket.sort_values("timestamp").reset_index(drop=True)
    idx    = int(len(bucket) * (1 - test_size))
    return bucket.iloc[:idx].copy(), bucket.iloc[idx:].copy()


def assign_fomo_level(score):
    if score >= FOMO_HIGH_THRESH:     return "High"
    elif score >= FOMO_MEDIUM_THRESH: return "Medium"
    else:                             return "Low"


def prepare(subset, feature_cols):
    X = subset[feature_cols].fillna(subset[feature_cols].median())
    y = subset["fomo_prob"].values
    w = np.abs(y - 0.5) * 2 + WEIGHT_EPSILON
    return X, y, w


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load data
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: Loading data...")
print("=" * 65)

df = pd.read_csv(FEATURES_FILE, parse_dates=["timestamp"])
print(f"  fomo_features: {df.shape}")
print(f"  Investors    : {df['investor_id'].nunique():,}")
print(f"  fomo_prob mean: {df['fomo_prob'].mean():.4f}")

feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
print(f"  Features ({len(feature_cols)}): {feature_cols}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Chia regime + tạo bucket
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: Phân chia theo Market Regime...")
print("=" * 65)

bull_df     = df[(df['timestamp'] >= BULL_START)     & (df['timestamp'] <= BULL_END)]
sideways_df = df[(df['timestamp'] >= SIDEWAYS_START) & (df['timestamp'] <= SIDEWAYS_END)]
bear_df     = df[(df['timestamp'] >= BEAR_START)     & (df['timestamp'] <= BEAR_END)]
outside     = len(df) - len(bull_df) - len(sideways_df) - len(bear_df)

print(f"  Bull     ({BULL_START} → {BULL_END}):     {len(bull_df):,} rows")
print(f"  Sideways ({SIDEWAYS_START} → {SIDEWAYS_END}): {len(sideways_df):,} rows")
print(f"  Bear     ({BEAR_START} → {BEAR_END}):     {len(bear_df):,} rows")
print(f"  Outside regimes (bỏ qua):                      {outside:,} rows")

# Sideways chia đều → nửa đầu vào Bull, nửa sau vào Bear
sideways_sorted  = sideways_df.sort_values("timestamp").reset_index(drop=True)
mid_idx          = len(sideways_sorted) // 2
sideways_to_bull = sideways_sorted.iloc[:mid_idx]
sideways_to_bear = sideways_sorted.iloc[mid_idx:]

bull_bucket = pd.concat([bull_df, sideways_to_bull]).sort_values("timestamp").reset_index(drop=True)
bear_bucket = pd.concat([bear_df, sideways_to_bear]).sort_values("timestamp").reset_index(drop=True)

print(f"\n  Bull bucket (Bull + Sideways nửa đầu): {len(bull_bucket):,} rows")
print(f"  Bear bucket (Bear + Sideways nửa sau): {len(bear_bucket):,} rows")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Time split trong từng bucket
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3: Time-based split trong từng bucket (80/20)...")
print("=" * 65)

bull_train, bull_test = time_split_bucket(bull_bucket)
bear_train, bear_test = time_split_bucket(bear_bucket)

print(f"  Bull  → train: {len(bull_train):,} | test: {len(bull_test):,}")
print(f"  Bear  → train: {len(bear_train):,} | test: {len(bear_test):,}")

train_df = pd.concat([bull_train, bear_train]).sort_values("timestamp").reset_index(drop=True)
test_df  = pd.concat([bull_test,  bear_test ]).sort_values("timestamp").reset_index(drop=True)
full_df  = pd.concat([bull_bucket, bear_bucket]).sort_values("timestamp").reset_index(drop=True)

print(f"\n  Total train: {len(train_df):,} | test: {len(test_df):,}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Prepare X, y, weights
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: Preparing features & weights...")
print("=" * 65)

X_train, y_train, w_train = prepare(train_df, feature_cols)
X_test,  y_test,  w_test  = prepare(test_df,  feature_cols)
X_full,  y_full,  _       = prepare(full_df,  feature_cols)

print(f"  X_train: {X_train.shape} | y mean: {y_train.mean():.4f}")
print(f"  X_test : {X_test.shape}  | y mean: {y_test.mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Optuna Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"STEP 5: Optuna tuning ({OPTUNA_TRIALS} trials)...")
print("=" * 65)

# Optuna val: lấy 20% cuối của TỪNG bucket riêng → gộp lại
# Đảm bảo val có đủ cả Bull lẫn Bear signal
bull_val_idx  = int(len(bull_train) * (1 - TEST_SIZE))
bear_val_idx  = int(len(bear_train) * (1 - TEST_SIZE))

bull_tr_df    = bull_train.iloc[:bull_val_idx]
bull_val_df   = bull_train.iloc[bull_val_idx:]
bear_tr_df    = bear_train.iloc[:bear_val_idx]
bear_val_df   = bear_train.iloc[bear_val_idx:]

optuna_tr_df  = pd.concat([bull_tr_df,  bear_tr_df ]).sort_values("timestamp").reset_index(drop=True)
optuna_val_df = pd.concat([bull_val_df, bear_val_df]).sort_values("timestamp").reset_index(drop=True)

X_tr,  y_tr,  w_tr  = prepare(optuna_tr_df,  feature_cols)
X_val, y_val, w_val = prepare(optuna_val_df, feature_cols)

print(f"  Optuna train: {len(X_tr):,} | val: {len(X_val):,}")
print(f"  Val Bull: {len(bull_val_df):,} | Val Bear: {len(bear_val_df):,}")

def objective(trial):
    params = {
        "objective"        : "reg:logistic",
        "eval_metric"      : "logloss",
        "tree_method"      : "hist",
        "random_state"     : RANDOM_STATE,
        "verbosity"        : 0,
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
    model.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_val, y_val)],
              sample_weight_eval_set=[w_val],
              verbose=False)
    preds = np.clip(model.predict(X_val), 1e-7, 1 - 1e-7)
    return log_loss((y_val > 0.5).astype(int), preds, sample_weight=w_val)

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective, n_trials=OPTUNA_TRIALS,
               timeout=OPTUNA_TIMEOUT, show_progress_bar=True)

best_params   = study.best_params
best_val_loss = study.best_value
print(f"\n  Best val logloss: {best_val_loss:.6f}")
print(f"  Best params:")
for k, v in best_params.items():
    print(f"    {k:<22s}: {v}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Train Final Model
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6: Training final model...")
print("=" * 65)

final_model = xgb.XGBRegressor(
    objective="reg:logistic", eval_metric="logloss",
    tree_method="hist", random_state=RANDOM_STATE,
    verbosity=0, **best_params
)
final_model.fit(
    X_train, y_train, sample_weight=w_train,
    eval_set=[(X_test, y_test)],
    sample_weight_eval_set=[w_test],
    verbose=50,
)
print("  Final model trained.")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — Evaluation
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 7: Evaluation...")
print("=" * 65)

y_pred_test = np.clip(final_model.predict(X_test), 1e-7, 1 - 1e-7)

# ── Tầng 1: CV AUC ────────────────────────────────────────────────────────
print("\n  [Tầng 1] CV AUC — Label Learnability")
print("  " + "-" * 50)

df_cv = full_df[
    (full_df["fomo_prob"] > FOMO_HIGH_THRESH) |
    (full_df["fomo_prob"] < FOMO_LOW_THRESH)
].copy()
df_cv["fomo_label"] = (df_cv["fomo_prob"] > FOMO_HIGH_THRESH).astype(int)
df_cv = df_cv.sort_values("timestamp").reset_index(drop=True)

X_cv = df_cv[feature_cols].fillna(df_cv[feature_cols].median())
y_cv = df_cv["fomo_label"].values

print(f"  CV data: {len(df_cv):,} rows "
      f"(FOMO=1: {y_cv.sum():,} | FOMO=0: {(y_cv==0).sum():,})")

tscv    = TimeSeriesSplit(n_splits=CV_FOLDS)
cv_aucs = []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_cv)):
    X_cv_tr, X_cv_te = X_cv.iloc[tr_idx], X_cv.iloc[te_idx]
    y_cv_tr, y_cv_te = y_cv[tr_idx],      y_cv[te_idx]

    cv_model = xgb.XGBRegressor(
        objective="reg:logistic", tree_method="hist",
        random_state=RANDOM_STATE, verbosity=0,
        **best_params
    )
    cv_model.fit(X_cv_tr, y_cv_tr)
    y_cv_prob = np.clip(cv_model.predict(X_cv_te), 1e-7, 1 - 1e-7)

    if len(np.unique(y_cv_te)) < 2:
        print(f"  Fold {fold+1}: Skip (only 1 class)")
        continue

    auc = roc_auc_score(y_cv_te, y_cv_prob)
    cv_aucs.append(auc)
    print(f"  Fold {fold+1}: AUC={auc:.4f} "
          f"(n={len(y_cv_te):,}, FOMO%={y_cv_te.mean()*100:.1f}%)")

cv_auc_mean = np.mean(cv_aucs)
cv_auc_std  = np.std(cv_aucs)
print(f"\n  Mean CV AUC = {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
if cv_auc_mean > 0.75:
    print(f"  ✓ Learnable tốt (AUC > 0.75)")
elif cv_auc_mean > 0.65:
    print(f"  ~ Learnable vừa phải (AUC > 0.65)")
else:
    print(f"  ✗ Khó học (AUC < 0.65)")

# ── Tầng 2: Precision-focused metrics ─────────────────────────────────────
print("\n  [Tầng 2] Precision-focused Metrics")
print("  " + "-" * 50)

test_hc_mask = (y_test > FOMO_HIGH_THRESH) | (y_test < FOMO_LOW_THRESH)
y_test_hc    = (y_test[test_hc_mask] > FOMO_HIGH_THRESH).astype(int)
y_pred_hc    = y_pred_test[test_hc_mask]
y_pred_label = (y_pred_hc >= 0.71).astype(int)

print(f"  High-confidence test: {test_hc_mask.sum():,} rows "
      f"(FOMO=1: {y_test_hc.sum():,} | FOMO=0: {(y_test_hc==0).sum():,})")

pr_auc = average_precision_score(y_test_hc, y_pred_hc)
f05    = fbeta_score(y_test_hc, y_pred_label, beta=0.5, zero_division=0)
prec   = precision_score(y_test_hc, y_pred_label, zero_division=0)
rec    = recall_score(y_test_hc, y_pred_label, zero_division=0)

print(f"\n  PR-AUC    : {pr_auc:.4f}")
print(f"  F0.5      : {f05:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")

print(f"\n  Precision@K:")
for k_pct in [0.05, 0.10, 0.20]:
    k      = max(1, int(len(y_pred_hc) * k_pct))
    top_k  = np.argsort(y_pred_hc)[::-1][:k]
    prec_k = y_test_hc[top_k].mean()
    print(f"    Top {k_pct*100:.0f}% (n={k:,}): Precision@K = {prec_k:.4f}")

print(f"\n  Performance per regime:")
for regime, lo, hi in [("Bull", BULL_START, BULL_END), ("Bear", BEAR_START, BEAR_END)]:
    mask = (test_df['timestamp'] >= lo) & (test_df['timestamp'] <= hi)
    if mask.sum() == 0:
        continue
    y_r  = y_test[mask.values]
    p_r  = y_pred_test[mask.values]
    hc_r = (y_r > FOMO_HIGH_THRESH) | (y_r < FOMO_LOW_THRESH)
    if hc_r.sum() == 0 or len(np.unique((y_r[hc_r] > FOMO_HIGH_THRESH).astype(int))) < 2:
        continue
    auc_r = roc_auc_score((y_r[hc_r] > FOMO_HIGH_THRESH).astype(int), p_r[hc_r])
    pr_r  = average_precision_score((y_r[hc_r] > FOMO_HIGH_THRESH).astype(int), p_r[hc_r])
    print(f"    {regime:<10}: n={mask.sum():,}  AUC={auc_r:.4f}  PR-AUC={pr_r:.4f}")

print(f"\n  Prediction distribution (full test):")
for lo, hi, lbl in [
    (0.0,  0.40, "Low FOMO    (< 0.40)"),
    (0.40, 0.65, "Medium FOMO (0.40-0.65)"),
    (0.65, 1.01, "High FOMO   (>= 0.65)"),
]:
    mask = (y_pred_test >= lo) & (y_pred_test < hi)
    bar  = "█" * int(mask.mean() * 40)
    print(f"  {lbl}  {bar:<40} {mask.sum():>5,} ({mask.mean()*100:.1f}%)")
# ── Optuna threshold tuning — tối ưu Precision với ràng buộc Recall ──────
print(f"\n  Optuna threshold tuning (maximize Precision, Recall < {rec:.4f}):")

baseline_recall = rec  # recall tại threshold 0.5



# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — Predict full dataset
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 8: Predicting full dataset...")
print("=" * 65)

y_pred_all = np.clip(final_model.predict(X_full), 1e-7, 1 - 1e-7)
train_ids  = set(train_df['tx_id'].values)

predictions_df = pd.DataFrame({
    "tx_id"             : full_df["tx_id"].values,
    "investor_id"       : full_df["investor_id"].values,
    "timestamp"         : full_df["timestamp"].values,
    "fomo_prob_snorkel" : y_full,
    "fomo_score"        : y_pred_all,
    "fomo_level"        : [assign_fomo_level(s) for s in y_pred_all],
    "split"             : ["train" if t in train_ids else "test"
                           for t in full_df["tx_id"].values],
})

print(f"  Total predictions: {len(predictions_df):,}")
print(predictions_df["fomo_level"].value_counts().to_string())


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9 — SHAP
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 9: Computing SHAP values...")
print("=" * 65)

shap_idx = np.random.RandomState(RANDOM_STATE).choice(
    len(X_full), size=min(10_000, len(X_full)), replace=False
)
X_shap = X_full.iloc[shap_idx].copy()

try:
    explainer = shap.TreeExplainer(final_model)
except (ValueError, TypeError):
    booster = final_model.get_booster()
    config  = json.loads(booster.save_config())
    param   = config["learner"]["learner_model_param"]
    raw     = param.get("base_score", "0.5")
    if isinstance(raw, str) and raw.startswith("["):
        param["base_score"] = raw.strip("[]")
    booster.load_config(json.dumps(config))
    fixed_model          = xgb.XGBRegressor()
    fixed_model._Booster = booster
    explainer = shap.TreeExplainer(fixed_model)

shap_values   = explainer.shap_values(X_shap)
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), index=feature_cols
).sort_values(ascending=False)

total_shap = mean_abs_shap.sum()
print(f"\n  Features by mean |SHAP|:")
for feat, val in mean_abs_shap.items():
    bar = "█" * int(val / mean_abs_shap.iloc[0] * 25)
    print(f"    {feat:<45s}: {val:.4f} ({val/total_shap*100:.1f}%)  {bar}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 10 — Save
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 10: Saving outputs...")
print("=" * 65)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

predictions_df.to_csv(PREDICTIONS_FILE, index=False)
print(f"  ✓ Predictions : {PREDICTIONS_FILE}  ({predictions_df.shape})")

shap_df = pd.DataFrame(shap_values, columns=feature_cols)
shap_df.insert(0, "tx_id", full_df["tx_id"].iloc[shap_idx].values)
shap_df.to_csv(SHAP_FILE, index=False)
print(f"  ✓ SHAP values : {SHAP_FILE}  ({shap_df.shape})")

final_model.save_model(XGBOOST_MODEL_FILE)
print(f"  ✓ Model       : {XGBOOST_MODEL_FILE}")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Split strategy  : Regime-based time split")
print(f"  Bull bucket     : {len(bull_bucket):,} → train {len(bull_train):,} / test {len(bull_test):,}")
print(f"  Bear bucket     : {len(bear_bucket):,} → train {len(bear_train):,} / test {len(bear_test):,}")
print(f"  Total train     : {len(train_df):,} | test: {len(test_df):,}")
print(f"  Features        : {len(feature_cols)}")
print(f"  Optuna best     : {best_val_loss:.6f}")
print(f"\n  [Tầng 1] Label Learnability:")
print(f"    CV AUC = {cv_auc_mean:.4f} ± {cv_auc_std:.4f} ({CV_FOLDS}-fold TimeSeriesSplit)")
print(f"\n  [Tầng 2] Precision-focused:")
print(f"    PR-AUC    = {pr_auc:.4f}")
print(f"    F0.5      = {f05:.4f}")
print(f"    Precision = {prec:.4f}")
print(f"    Recall    = {rec:.4f}")
print(f"\n  fomo_level breakdown:")
for lvl in ["High", "Medium", "Low"]:
    n = (predictions_df["fomo_level"] == lvl).sum()
    print(f"    {lvl:<8}: {n:,} ({n/len(predictions_df)*100:.1f}%)")
print(f"\n  Top 5 FOMO features (SHAP):")
for feat, val in mean_abs_shap.head(5).items():
    print(f"    {feat:<45s}: {val:.4f} ({val/total_shap*100:.1f}%)")
print("\n✓ Done.")