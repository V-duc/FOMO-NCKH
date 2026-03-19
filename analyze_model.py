"""
analyze_model.py — SHAP analysis + feature importance sau khi train XGBoost.

Load model đã train từ file, chạy SHAP, export kết quả.
File này chạy độc lập với train_xgboost.py — không cần retrain.

Input:
    data/models/fomo_xgboost.json   — trained model
    data/output/fomo_features.csv   — features + fomo_prob
    data/output/fomo_predictions.csv — predictions đã có từ training

Output:
    data/output/shap_values.csv          — SHAP per feature per transaction
    data/output/feature_importance.csv   — mean |SHAP| per feature, ranked

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

FEATURES_FILE     = f"{OUTPUT_DIR}/fomo_features.csv"
PREDICTIONS_FILE  = f"{OUTPUT_DIR}/fomo_predictions.csv"
SHAP_FILE         = f"{OUTPUT_DIR}/shap_values.csv"
IMPORTANCE_FILE   = f"{OUTPUT_DIR}/feature_importance.csv"

# ── Hardcode từ Optuna output (trial 44) ─────────────────────────────────
BEST_PARAMS = {
    "n_estimators"    : 687,
    "max_depth"       : 8,
    "learning_rate"   : 0.08052465159589915,
    "subsample"       : 0.7485827300611674,
    "colsample_bytree": 0.5886086220101581,
    "min_child_weight": 1,
    "reg_alpha"       : 0.5522680449800129,
    "reg_lambda"      : 3.4816341279910063e-07,
    "gamma"           : 0.22843311562189395,
}

RANDOM_STATE     = 42
TEST_SIZE        = 0.2
WEIGHT_EPSILON   = 0.01
SHAP_SAMPLE_SIZE = 10_000

FOMO_HIGH_THRESH   = 0.65
FOMO_MEDIUM_THRESH = 0.40

NON_FEATURE_COLS = [
    "tx_id", "investor_id", "timestamp",
    "fomo_prob", "momentum_acceleration",
]


def sep(title="", width=60):
    print(f"\n{'='*width}")
    if title:
        print(f"  {title}")
        print(f"{'='*width}")


def load_model_with_fix(model_path):
    """
    Load XGBoost model với fix cho lỗi base_score format.
    XGBoost >= 2.0 serialize base_score dạng '[3.45E-1]' —
    shap.TreeExplainer không parse được → patch config trực tiếp vào booster.
    """
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    try:
        explainer = shap.TreeExplainer(model)
        print("  TreeExplainer: OK (no fix needed)")
        return model, explainer
    except (ValueError, TypeError):
        print("  TreeExplainer failed — patching booster config...")

        # Lấy config từ booster dưới dạng JSON string
        booster = model.get_booster()
        config  = json.loads(booster.save_config())

        # Fix base_score: strip brackets
        param = config["learner"]["learner_model_param"]
        raw   = param.get("base_score", "0.5")
        if isinstance(raw, str) and raw.startswith("["):
            param["base_score"] = raw.strip("[]")

        # Load config đã fix lại vào booster
        booster.load_config(json.dumps(config))

        # Wrap lại thành XGBRegressor để SHAP nhận dạng đúng
        fixed_model = xgb.XGBRegressor()
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

feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
X = df[feature_cols].fillna(df[feature_cols].median())
y = df["fomo_prob"].values
print(f"  Features: {len(feature_cols)}")
print(f"  Feature list: {feature_cols}")

# Recreate train/test split giống hệt train_xgboost.py
from sklearn.model_selection import train_test_split
idx = np.arange(len(df))
sample_weight = np.abs(y - 0.5) * 2 + WEIGHT_EPSILON

(X_train, X_test,
 y_train, y_test,
 w_train, w_test,
 idx_train, idx_test) = train_test_split(
    X, y, sample_weight, idx,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# Load model
print(f"\n  Loading model: {XGBOOST_MODEL_FILE}")
model, explainer = load_model_with_fix(XGBOOST_MODEL_FILE)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Re-evaluate trên test set (sanity check)
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 2: Evaluation on test set")

y_pred = np.clip(model.predict(X_test), 1e-7, 1 - 1e-7)

rmse     = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=w_test))
logloss  = log_loss((y_test > 0.5).astype(int), y_pred, sample_weight=w_test)

print(f"  RMSE (weighted)   : {rmse:.6f}")
print(f"  LogLoss (weighted): {logloss:.6f}")

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


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — XGBoost built-in feature importance
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 3: XGBoost built-in feature importance")

importance_types = ["weight", "gain", "cover"]
importance_df    = pd.DataFrame({"feature": feature_cols})

for imp_type in importance_types:
    scores = model.get_booster().get_score(importance_type=imp_type)
    importance_df[imp_type] = importance_df["feature"].map(scores).fillna(0)

# Normalize gain (most meaningful metric)
importance_df["gain_normalized"] = (
    importance_df["gain"] / importance_df["gain"].sum()
)
importance_df = importance_df.sort_values("gain", ascending=False).reset_index(drop=True)

print(f"\n  Top 20 features by gain:")
print(f"  {'Rank':<5} {'Feature':<45} {'Gain%':>8}  {'Weight':>8}  {'Cover':>8}")
print(f"  {'-'*5} {'-'*45} {'-'*8}  {'-'*8}  {'-'*8}")
for i, row in importance_df.head(20).iterrows():
    print(f"  {i+1:<5} {row['feature']:<45} "
          f"{row['gain_normalized']*100:>7.2f}%  "
          f"{row['weight']:>8.0f}  "
          f"{row['cover']:>8.0f}")

# Features với gain = 0 (không được dùng)
zero_gain = importance_df[importance_df["gain"] == 0]
if len(zero_gain) > 0:
    print(f"\n  ⚠️  {len(zero_gain)} features không được dùng (gain=0):")
    for _, row in zero_gain.iterrows():
        print(f"     {row['feature']}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — SHAP values
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 4: SHAP values")

shap_size = min(SHAP_SAMPLE_SIZE, len(X))
rng       = np.random.RandomState(RANDOM_STATE)
shap_idx  = rng.choice(len(X), size=shap_size, replace=False)
X_shap    = X.iloc[shap_idx].copy()

print(f"  Computing SHAP on {shap_size:,} rows...")
shap_values = explainer.shap_values(X_shap)
shap_arr    = np.array(shap_values)
print(f"  Done. SHAP shape: {shap_arr.shape}")

# Mean absolute SHAP per feature
mean_abs_shap = pd.Series(
    np.abs(shap_arr).mean(axis=0),
    index=feature_cols,
).sort_values(ascending=False)

print(f"\n  Top 20 features by mean |SHAP|:")
print(f"  {'Rank':<5} {'Feature':<45} {'Mean |SHAP|':>12}  {'% total':>8}")
total_shap = mean_abs_shap.sum()
for i, (feat, val) in enumerate(mean_abs_shap.head(20).items()):
    bar = "█" * int(val / mean_abs_shap.iloc[0] * 30)
    print(f"  {i+1:<5} {feat:<45} {val:>12.6f}  {val/total_shap*100:>7.1f}%  {bar}")

# Group by nhóm feature
print(f"\n  SHAP importance by feature group:")
groups = {
    "Investor Profile"   : ["risk_level", "investment_capacity_ordinal", "investor_trade_index"],
    "Trading Habit"      : ["trade_gap_days", "rolling_avg_trade_gap_last_10",
                            "trades_per_investor_per_day", "digital_trade_flag",
                            "consecutive_buy_streak", "rolling_buy_ratio_last_5",
                            "rolling_buy_ratio_last_20", "rolling_trade_freq_5"],
    "Position Sizing"    : ["position_size_ratio", "position_size_spike_flag",
                            "capital_acceleration_ratio", "rolling_avg_position_size_last_10",
                            "position_size_to_volatility_ratio"],
    "Asset Switching"    : ["is_new_asset", "asset_diversity_last_10", "same_day_multiple_flag"],
    "Crowd Alignment"    : ["investor_alignment_with_crowd", "asset_popularity_zscore"],
    "Market Context"     : ["return_1d", "volatility_10d", "volatility_regime"],
    "Asset Statistical"  : ["total_value_pctrank_asset", "volatility_10d_pctrank_asset",
                            "market_fomo_pressure_score"],
    "Behavioral Std"     : ["return_1d_rolling_std_10", "volatility_5d_rolling_std_10"],
}

group_shap = {}
for group, feats in groups.items():
    present = [f for f in feats if f in mean_abs_shap.index]
    if present:
        group_shap[group] = mean_abs_shap[present].sum()

group_shap = pd.Series(group_shap).sort_values(ascending=False)
print(f"\n  {'Group':<25} {'Total SHAP':>12}  {'% total':>8}")
print(f"  {'-'*25} {'-'*12}  {'-'*8}")
for group, val in group_shap.items():
    bar = "█" * int(val / group_shap.iloc[0] * 30)
    print(f"  {group:<25} {val:>12.6f}  {val/total_shap*100:>7.1f}%  {bar}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Save outputs
# ════════════════════════════════════════════════════════════════════════════
sep("STEP 5: Saving outputs")

# SHAP values per transaction
shap_df = pd.DataFrame(shap_arr, columns=feature_cols)
shap_df.insert(0, "tx_id", df["tx_id"].iloc[shap_idx].values)
shap_df.to_csv(SHAP_FILE, index=False)
print(f"  ✓ SHAP values      : {SHAP_FILE}  {shap_df.shape}")

# Feature importance: merge SHAP + built-in gain
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
print(f"  Model     : {XGBOOST_MODEL_FILE}")
print(f"  Test RMSE : {rmse:.6f}")
print(f"  Test Loss : {logloss:.6f}")
print(f"\n  Top 5 FOMO behavioral signals (SHAP):")
behavioral_feats = [f for f in mean_abs_shap.index
                    if f not in ["return_1d", "volatility_10d", "volatility_regime",
                                 "total_value_pctrank_asset", "volatility_10d_pctrank_asset",
                                 "market_fomo_pressure_score", "asset_popularity_zscore"]]
for i, feat in enumerate(behavioral_feats[:5]):
    print(f"    {i+1}. {feat:<40} {mean_abs_shap[feat]:.6f}")

print(f"\n  Dominant group: {group_shap.index[0]} "
      f"({group_shap.iloc[0]/total_shap*100:.1f}% of total SHAP)")
print(f"\n✓ Done.")
