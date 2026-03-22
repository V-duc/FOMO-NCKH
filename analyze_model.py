"""
analyze_model.py — SHAP analysis + feature importance sau khi train XGBoost.

Load model đã train từ file, chạy SHAP, export kết quả.
NON_FEATURE_COLS và split strategy giống hệt train_xgboost_new.py.

Input:
    data/models/fomo_xgboost.json
    data/output/fomo_features.csv

Output:
    data/output/shap_values.csv
    data/output/feature_importance.csv

Chạy:
    python analyze_model.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import log_loss, mean_squared_error

warnings.filterwarnings("ignore")

from constants import OUTPUT_DIR, MODEL_DIR, XGBOOST_MODEL_FILE

FEATURES_FILE    = f"{OUTPUT_DIR}/fomo_features.csv"
SHAP_FILE        = f"{OUTPUT_DIR}/shap_values.csv"
IMPORTANCE_FILE  = f"{OUTPUT_DIR}/feature_importance.csv"

RANDOM_STATE     = 42
TEST_SIZE        = 0.2
WEIGHT_EPSILON   = 0.01
SHAP_SAMPLE_SIZE = 10_000

FOMO_HIGH_THRESH   = 0.65
FOMO_LOW_THRESH    = 0.35
FOMO_MEDIUM_THRESH = 0.40

# ── Market regime boundaries ──────────────────────────────────────────────
BULL_START     = '2020-11-01'
BULL_END       = '2021-06-30'
SIDEWAYS_START = '2021-07-01'
SIDEWAYS_END   = '2022-01-31'
BEAR_START     = '2022-02-01'
BEAR_END       = '2022-07-31'

# ── NON_FEATURE_COLS — giống hệt train_xgboost_new.py ────────────────────
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

# ── SHAP groups ───────────────────────────────────────────────────────────
SHAP_GROUPS = {
    "Investor Profile" : ["risk_level", "investment_capacity_ordinal", "investor_trade_index"],
    "Trading Habit"    : ["rolling_avg_trade_gap_last_10", "digital_trade_flag",
                          "consecutive_buy_streak", "rolling_buy_ratio_last_5",
                          "rolling_buy_ratio_last_20"],
    "Position Sizing"  : ["position_size_spike_flag", "capital_acceleration_ratio"],
    "Asset Switching"  : ["is_new_asset", "asset_diversity_last_10"],
    "Crowd Alignment"  : ["investor_alignment_with_crowd"],
    "Market Context"   : ["volatility_10d", "volatility_ratio",
                          "market_breadth", "ema_12", "ema_26"],
    "Asset Statistical": ["volatility_10d_pctrank_asset"],
    "Behavioral Std"   : ["return_1d_rolling_std_10", "volatility_5d_rolling_std_10"],
}


def sep(title="", width=65):
    print(f"\n{'='*width}")
    if title:
        print(f"  {title}")
        print(f"{'='*width}")


def time_split_bucket(bucket, test_size=TEST_SIZE):
    bucket = bucket.sort_values("timestamp").reset_index(drop=True)
    idx    = int(len(bucket) * (1 - test_size))
    return bucket.iloc[:idx].copy(), bucket.iloc[idx:].copy()


def load_model_with_fix(model_path):
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    try:
        explainer = shap.TreeExplainer(model)
        print("  TreeExplainer: OK (no fix needed)")
        return model, explainer
    except (ValueError, TypeError):
        print("  TreeExplainer failed — patching booster config...")
        booster = model.get_booster()
        config  = json.loads(booster.save_config())
        param   = config["learner"]["learner_model_param"]
        raw     = param.get("base_score", "0.5")
        if isinstance(raw, str) and raw.startswith("["):
            param["base_score"] = raw.strip("[]")
        booster.load_config(json.dumps(config))
        fixed_model          = xgb.XGBRegressor()
        fixed_model._Booster = booster
        explainer = shap.TreeExplainer(fixed_model)
        print("  TreeExplainer: OK (booster config patched)")
        return fixed_model, explainer


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load data & model
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 1: Loading data & model")

df = pd.read_csv(FEATURES_FILE, parse_dates=["timestamp"])
print(f"  fomo_features: {df.shape}")

# Dùng cột từ fomo_features.csv sau khi loại NON_FEATURE_COLS
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
print(f"  Features ({len(feature_cols)}): {feature_cols}")

# ── Regime-based time split — giống hệt train_xgboost_new.py ─────────────
bull_df     = df[(df['timestamp'] >= BULL_START)     & (df['timestamp'] <= BULL_END)]
sideways_df = df[(df['timestamp'] >= SIDEWAYS_START) & (df['timestamp'] <= SIDEWAYS_END)]
bear_df     = df[(df['timestamp'] >= BEAR_START)     & (df['timestamp'] <= BEAR_END)]

sideways_sorted  = sideways_df.sort_values("timestamp").reset_index(drop=True)
mid_idx          = len(sideways_sorted) // 2
sideways_to_bull = sideways_sorted.iloc[:mid_idx]
sideways_to_bear = sideways_sorted.iloc[mid_idx:]

bull_bucket = pd.concat([bull_df, sideways_to_bull]).sort_values("timestamp").reset_index(drop=True)
bear_bucket = pd.concat([bear_df, sideways_to_bear]).sort_values("timestamp").reset_index(drop=True)

bull_train, bull_test = time_split_bucket(bull_bucket)
bear_train, bear_test = time_split_bucket(bear_bucket)

train_df = pd.concat([bull_train, bear_train]).sort_values("timestamp").reset_index(drop=True)
test_df  = pd.concat([bull_test,  bear_test ]).sort_values("timestamp").reset_index(drop=True)
full_df  = pd.concat([bull_bucket, bear_bucket]).sort_values("timestamp").reset_index(drop=True)

def prepare(subset):
    X = subset[feature_cols].fillna(subset[feature_cols].median())
    y = subset["fomo_prob"].values
    w = np.abs(y - 0.5) * 2 + WEIGHT_EPSILON
    return X, y, w

X_train, y_train, w_train = prepare(train_df)
X_test,  y_test,  w_test  = prepare(test_df)
X_full,  y_full,  _       = prepare(full_df)

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Full: {len(X_full):,}")
print(f"  Train period: {train_df['timestamp'].min().date()} → {train_df['timestamp'].max().date()}")
print(f"  Test  period: {test_df['timestamp'].min().date()} → {test_df['timestamp'].max().date()}")

print(f"\n  Loading model: {XGBOOST_MODEL_FILE}")
model, explainer = load_model_with_fix(XGBOOST_MODEL_FILE)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Evaluate trên test set + Baseline comparison
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 2: Evaluation on test set")

y_pred = np.clip(model.predict(X_test), 1e-7, 1 - 1e-7)

rmse    = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=w_test))
logloss = log_loss((y_test > FOMO_HIGH_THRESH).astype(int), y_pred, sample_weight=w_test)

# ── Baseline ──────────────────────────────────────────────────────────────
fomo_ratio = (y_test > FOMO_HIGH_THRESH).mean()

# Baseline RMSE: predict mean (naive mean predictor)
baseline_rmse = np.sqrt(mean_squared_error(
    y_test,
    np.full_like(y_test, y_test.mean()),
    sample_weight=w_test
))

# Baseline LogLoss: predict prior probability
p = fomo_ratio
baseline_logloss = -(p * np.log(p + 1e-8) + (1-p) * np.log(1-p + 1e-8))

print(f"\n  FOMO ratio (threshold={FOMO_HIGH_THRESH}): {fomo_ratio*100:.1f}%")
print(f"\n  {'Metric':<22} {'Baseline':>12} {'Model':>12}  {'Beat?':>8}  {'Improvement':>12}")
print(f"  {'-'*72}")

for metric, base_val, model_val in [
    ("RMSE (weighted)",    baseline_rmse,    rmse),
    ("LogLoss (weighted)", baseline_logloss, logloss),
]:
    beat  = model_val < base_val
    delta = base_val - model_val
    print(f"  {metric:<22} {base_val:>12.6f} {model_val:>12.6f}  "
          f"{'✓ Yes' if beat else '✗ No':>8}  {delta:>+12.6f}")

if rmse < baseline_rmse and logloss < baseline_logloss:
    print(f"\n  ✓ Model vượt baseline trên cả 2 metrics")
elif rmse < baseline_rmse or logloss < baseline_logloss:
    print(f"\n  ~ Model vượt baseline trên 1/2 metrics")
else:
    print(f"\n  ✗ Model không vượt baseline")

# Prediction distribution
print(f"\n  Prediction distribution (test set):")
for lo, hi, lbl in [
    (0.0,  0.40, "Low FOMO    (< 0.40)"),
    (0.40, 0.65, "Medium FOMO (0.40-0.65)"),
    (0.65, 1.01, "High FOMO   (>= 0.65)"),
]:
    mask = (y_pred >= lo) & (y_pred < hi)
    bar  = "█" * int(mask.mean() * 40)
    print(f"  {lbl}  {bar:<40} {mask.sum():>5,} ({mask.mean()*100:.1f}%)")

# Per regime
print(f"\n  Performance per regime:")
for regime, lo, hi in [("Bull", BULL_START, BULL_END), ("Bear", BEAR_START, BEAR_END)]:
    mask = (test_df['timestamp'] >= lo) & (test_df['timestamp'] <= hi)
    if mask.sum() == 0:
        continue
    y_r = y_test[mask.values]
    p_r = y_pred[mask.values]
    r   = np.sqrt(mean_squared_error(y_r, p_r, sample_weight=w_test[mask.values]))
    l   = log_loss((y_r > FOMO_HIGH_THRESH).astype(int),
                   np.clip(p_r, 1e-7, 1-1e-7),
                   sample_weight=w_test[mask.values])
    print(f"    {regime:<10}: n={mask.sum():,}  RMSE={r:.4f}  LogLoss={l:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — XGBoost built-in feature importance
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 3: XGBoost built-in feature importance (Gain)")

importance_df = pd.DataFrame({"feature": feature_cols})
for imp_type in ["weight", "gain", "cover"]:
    scores = model.get_booster().get_score(importance_type=imp_type)
    importance_df[imp_type] = importance_df["feature"].map(scores).fillna(0)

importance_df["gain_normalized"] = importance_df["gain"] / importance_df["gain"].sum()
importance_df = importance_df.sort_values("gain", ascending=False).reset_index(drop=True)

print(f"\n  {'Rank':<5} {'Feature':<45} {'Gain%':>8}  {'Weight':>8}  {'Cover':>8}")
print(f"  {'-'*5} {'-'*45} {'-'*8}  {'-'*8}  {'-'*8}")
for i, row in importance_df.iterrows():
    print(f"  {i+1:<5} {row['feature']:<45} "
          f"{row['gain_normalized']*100:>7.2f}%  "
          f"{row['weight']:>8.0f}  "
          f"{row['cover']:>8.0f}")

zero_gain = importance_df[importance_df["gain"] == 0]
if len(zero_gain) > 0:
    print(f"\n  ⚠️  {len(zero_gain)} features không được dùng (gain=0):")
    for _, row in zero_gain.iterrows():
        print(f"     {row['feature']}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — SHAP values
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 4: SHAP values")

shap_size = min(SHAP_SAMPLE_SIZE, len(X_full))
rng       = np.random.RandomState(RANDOM_STATE)
shap_idx  = rng.choice(len(X_full), size=shap_size, replace=False)
X_shap    = X_full.iloc[shap_idx].copy()

print(f"  Computing SHAP on {shap_size:,} rows...")
shap_values = explainer.shap_values(X_shap)
shap_arr    = np.array(shap_values)
print(f"  Done. SHAP shape: {shap_arr.shape}")

mean_abs_shap = pd.Series(
    np.abs(shap_arr).mean(axis=0),
    index=feature_cols,
).sort_values(ascending=False)

total_shap = mean_abs_shap.sum()
print(f"\n  Features by mean |SHAP|:")
print(f"  {'Rank':<5} {'Feature':<45} {'Mean |SHAP|':>12}  {'% total':>8}")
for i, (feat, val) in enumerate(mean_abs_shap.items()):
    bar = "█" * int(val / mean_abs_shap.iloc[0] * 30)
    print(f"  {i+1:<5} {feat:<45} {val:>12.6f}  {val/total_shap*100:>7.1f}%  {bar}")

# SHAP by group
print(f"\n  SHAP importance by feature group:")
group_shap = {}
for group, feats in SHAP_GROUPS.items():
    present = [f for f in feats if f in mean_abs_shap.index]
    if present:
        group_shap[group] = mean_abs_shap[present].sum()

group_shap = pd.Series(group_shap).sort_values(ascending=False)
print(f"\n  {'Group':<20} {'Total SHAP':>12}  {'% total':>8}")
print(f"  {'-'*20} {'-'*12}  {'-'*8}")
for group, val in group_shap.items():
    bar = "█" * int(val / group_shap.iloc[0] * 30)
    print(f"  {group:<20} {val:>12.6f}  {val/total_shap*100:>7.1f}%  {bar}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Save outputs
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 5: Saving outputs")

shap_df = pd.DataFrame(shap_arr, columns=feature_cols)
shap_df.insert(0, "tx_id", full_df["tx_id"].iloc[shap_idx].values)
shap_df.to_csv(SHAP_FILE, index=False)
print(f"  ✓ SHAP values      : {SHAP_FILE}  {shap_df.shape}")

importance_final = importance_df.merge(
    mean_abs_shap.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}),
    on="feature", how="left"
).sort_values("mean_abs_shap", ascending=False)
importance_final.to_csv(IMPORTANCE_FILE, index=False)
print(f"  ✓ Feature importance: {IMPORTANCE_FILE}  {importance_final.shape}")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
sep("SUMMARY")
print(f"  Model          : {XGBOOST_MODEL_FILE}")
print(f"  Split          : Regime-based time split")
print(f"  Features       : {len(feature_cols)}")
print(f"  Train/Test     : {len(train_df):,} / {len(test_df):,}")
print(f"\n  Baseline vs Model:")
print(f"    RMSE    baseline={baseline_rmse:.4f}  model={rmse:.4f}  "
      f"({'✓ beat' if rmse < baseline_rmse else '✗ not beat'})")
print(f"    LogLoss baseline={baseline_logloss:.4f}  model={logloss:.4f}  "
      f"({'✓ beat' if logloss < baseline_logloss else '✗ not beat'})")
print(f"\n  Top 5 FOMO features (SHAP):")
market_feats = ["volatility_10d", "volatility_10d_pctrank_asset",
                "volatility_ratio", "market_breadth", "ema_12", "ema_26"]
behavioral   = [f for f in mean_abs_shap.index if f not in market_feats]
for i, feat in enumerate(behavioral[:5]):
    print(f"    {i+1}. {feat:<45} {mean_abs_shap[feat]:.6f} ({mean_abs_shap[feat]/total_shap*100:.1f}%)")
print(f"\n  Dominant group: {group_shap.index[0]} "
      f"({group_shap.iloc[0]/total_shap*100:.1f}% of total SHAP)")
print(f"\n✓ Done.")