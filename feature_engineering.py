"""
feature_engineering.py — Tính toàn bộ features cho XGBoost FOMO detection.

NGUYÊN TẮC THIẾT KẾ:
    - KHÔNG dùng bất kỳ feature nào đã được dùng trong LF của Snorkel
    - LF dùng: return_5d, rsi_14, price_above_bollinger, asset_buy_count_same_day,
               totalValue vs P90 cá nhân, days_since_last_buy + same asset
    - Feature ở đây phải capture behavioral signal từ dimension KHÁC

INPUT:
    enriched_trades_train.csv  — BUY + SELL, đã có market context
    snorkel_labels.csv         — fomo_prob per tx_id

OUTPUT:
    fomo_features.csv          — 1 row per BUY transaction, features + fomo_prob

Chạy:
    python feature_engineering.py
"""

import pandas as pd
import numpy as np
from constants import ENRICHED_TRADES_TRAIN_FILE, OUTPUT_DIR

INPUT_LABELS  = f"{OUTPUT_DIR}/snorkel_labels.csv"
OUTPUT_FILE   = f"{OUTPUT_DIR}/fomo_features.csv"

# ════════════════════════════════════════════════════════════════════════════
# LOAD & SETUP
# ════════════════════════════════════════════════════════════════════════════

print("Loading data...")
df     = pd.read_csv(ENRICHED_TRADES_TRAIN_FILE, parse_dates=["timestamp"])
labels = pd.read_csv(INPUT_LABELS)

buys = df[df["side"] == "BUY"].copy()
buys = buys.sort_values(["investor_id", "timestamp"]).reset_index(drop=True)
print(f"  BUY transactions: {len(buys):,}")

all_trades = df.sort_values(["investor_id", "timestamp"]).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 1 — INVESTOR PROFILE
# ════════════════════════════════════════════════════════════════════════════

# 1. risk_level — đã có sẵn
# 2. investment_capacity_ordinal — đã có sẵn

# 3. investor_trade_index
buys["investor_trade_index"] = buys.groupby("investor_id").cumcount()


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 2 — TRADING HABIT
# ════════════════════════════════════════════════════════════════════════════

# 4. trade_gap_days
buys["trade_gap_days"] = (
    buys.groupby("investor_id")["timestamp"]
    .diff().dt.days
)

# 5. rolling_avg_trade_gap_last_10
buys["rolling_avg_trade_gap_last_10"] = (
    buys.groupby("investor_id")["trade_gap_days"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# 6. trades_per_investor_per_day
buys["trades_per_investor_per_day"] = (
    buys.groupby(["investor_id", buys["timestamp"].dt.date]).transform("size")
)

# 7. digital_trade_flag
buys["digital_trade_flag"] = (buys["channel"] == "Internet Banking").astype(int)

# 8. consecutive_buy_streak
def compute_buy_streak(group):
    streak = []
    count = 0
    for side in group["side"]:
        if side == "BUY":
            count += 1
        else:
            count = 0
        streak.append(count)
    return streak

streaks = []
for inv_id, group in all_trades.groupby("investor_id"):
    s = compute_buy_streak(group)
    streaks.extend(list(zip(group.index, s)))

streak_series = pd.Series(dict(streaks), name="consecutive_buy_streak")
all_trades["consecutive_buy_streak"] = streak_series
buys = buys.merge(
    all_trades[["tx_id", "consecutive_buy_streak"]],
    on="tx_id", how="left"
)

# 9. rolling_buy_ratio_last_5
all_trades["is_buy"] = (all_trades["side"] == "BUY").astype(int)
all_trades["rolling_buy_ratio_last_5"] = (
    all_trades.groupby("investor_id")["is_buy"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
)
buys = buys.merge(
    all_trades[["tx_id", "rolling_buy_ratio_last_5"]],
    on="tx_id", how="left"
)

# 10. rolling_buy_ratio_last_20
all_trades["rolling_buy_ratio_last_20"] = (
    all_trades.groupby("investor_id")["is_buy"]
    .transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
)
buys = buys.merge(
    all_trades[["tx_id", "rolling_buy_ratio_last_20"]],
    on="tx_id", how="left"
)

# 11. rolling_trade_freq_5
buys["rolling_trade_freq_5"] = (
    buys.groupby("investor_id")["trade_gap_days"]
    .transform(lambda x: 1 / (x.shift(1).rolling(5, min_periods=2).mean() + 1))
)


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 3 — POSITION SIZING
# ════════════════════════════════════════════════════════════════════════════

# 12. position_size_ratio — đã có sẵn

# 13. position_size_spike_flag
p95_value = (
    buys.groupby("investor_id")["totalValue"]
    .transform(lambda x: x.shift(1).rolling(50, min_periods=5).quantile(0.95))
)
buys["position_size_spike_flag"] = (buys["totalValue"] > p95_value).astype(float)
buys.loc[p95_value.isna(), "position_size_spike_flag"] = np.nan

# 14. capital_acceleration_ratio
rolling_mean_10 = (
    buys.groupby("investor_id")["totalValue"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)
buys["capital_acceleration_ratio"] = buys["totalValue"] / rolling_mean_10.replace(0, np.nan)

# 15. rolling_avg_position_size_last_10
buys["rolling_avg_position_size_last_10"] = (
    buys.groupby("investor_id")["totalValue"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# 16. position_size_to_volatility_ratio
buys["position_size_to_volatility_ratio"] = (
    buys["totalValue"] / (buys["volatility_10d"].replace(0, np.nan) * buys["totalValue"].mean())
)


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 4 — ASSET SWITCHING
# ════════════════════════════════════════════════════════════════════════════

# 17. is_new_asset
buys["prev_asset_id"] = buys.groupby("investor_id")["asset_id"].shift(1)
buys["is_new_asset"] = (
    (buys["asset_id"] != buys["prev_asset_id"]) & buys["prev_asset_id"].notna()
).astype(float)
buys.loc[buys["prev_asset_id"].isna(), "is_new_asset"] = np.nan

# 18. asset_diversity_last_10
buys["asset_id_code"] = buys["asset_id"].astype("category").cat.codes
buys["asset_diversity_last_10"] = (
    buys.groupby("investor_id")["asset_id_code"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3)
               .apply(lambda w: len(set(w.astype(int))), raw=True))
)
buys = buys.drop(columns=["asset_id_code"])

# 19. same_day_multiple_flag
buys["same_day_multiple_flag"] = (buys["trades_per_investor_per_day"] >= 2).astype(int)


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 5 — CROWD ALIGNMENT
# ════════════════════════════════════════════════════════════════════════════

# 20. investor_alignment_with_crowd
daily_market = (
    df.groupby(df["timestamp"].dt.date)["side"]
    .apply(lambda x: (x == "BUY").sum() / len(x))
    .reset_index()
    .rename(columns={"timestamp": "date", "side": "market_buy_ratio"})
)
buys["date"] = buys["timestamp"].dt.date
buys = buys.merge(daily_market, on="date", how="left")
buys["investor_alignment_with_crowd"] = (buys["market_buy_ratio"] > 0.7).astype(int)

# 21. asset_popularity_zscore
asset_daily = (
    df[df["side"] == "BUY"]
    .groupby(["asset_id", df["timestamp"].dt.date.rename("date")])
    .size()
    .reset_index(name="daily_buy_count")
)
asset_daily["asset_popularity_zscore"] = (
    asset_daily.groupby("asset_id")["daily_buy_count"]
    .transform(lambda x: (x - x.rolling(60, min_periods=10).mean().shift(1)) /
               (x.rolling(60, min_periods=10).std().shift(1) + 1e-8))
)
buys = buys.merge(
    asset_daily[["asset_id", "date", "asset_popularity_zscore"]],
    on=["asset_id", "date"], how="left"
)


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 6 — MARKET CONTEXT SẠCH
# ════════════════════════════════════════════════════════════════════════════

# 22. return_1d — đã có sẵn
# 23. volatility_10d — đã có sẵn

# 24. volatility_regime
buys["volatility_regime"] = pd.qcut(
    buys["volatility_10d"].fillna(buys["volatility_10d"].median()),
    q=3, labels=[0, 1, 2]
).astype(float)

# 25. momentum_acceleration
if "return_3d" in buys.columns and "return_10d" in buys.columns:
    buys["momentum_acceleration"] = buys["return_3d"] - buys["return_10d"]
else:
    buys["momentum_acceleration"] = np.nan


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 7 — ASSET STATISTICAL
# ════════════════════════════════════════════════════════════════════════════

# 26. total_value_pctrank_asset
buys["total_value_pctrank_asset"] = (
    buys.groupby("asset_id")["totalValue"]
    .transform(lambda x: x.rank(pct=True))
)

# 27. volatility_10d_pctrank_asset
buys["volatility_10d_pctrank_asset"] = (
    buys.groupby("asset_id")["volatility_10d"]
    .transform(lambda x: x.rank(pct=True))
)

# 28. market_fomo_pressure_score
buys["market_fomo_pressure_score"] = (
    buys["asset_popularity_zscore"].fillna(0).clip(-3, 3) / 3 * 0.4 +
    buys["investor_alignment_with_crowd"].fillna(0) * 0.3 +
    buys["volatility_regime"].fillna(1) / 2 * 0.3
).clip(0, 1)


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 8 — ROLLING BEHAVIORAL STD
# ════════════════════════════════════════════════════════════════════════════

# 29. return_1d_rolling_std_10
buys["return_1d_rolling_std_10"] = (
    buys.groupby("investor_id")["return_1d"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).std())
)

# 30. volatility_5d_rolling_std_10
buys["volatility_5d_rolling_std_10"] = (
    buys.groupby("investor_id")["volatility_5d"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).std())
)


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 9 — NEW MARKET FEATURES
# ════════════════════════════════════════════════════════════════════════════

# 31. volatility_ratio — đã tính trong data_builder.py
# volatility_5d / volatility_10d — đã có sẵn trong enriched file

# 32. market_breadth — đã tính trong data_builder.py
# % assets đang tăng giá trong ngày — đã có sẵn trong enriched file

# 33. ema_12 — đã tính trong data_builder.py
# 34. ema_26 — đã tính trong data_builder.py

# Kiểm tra và warn nếu thiếu
for new_col in ["volatility_ratio", "market_breadth", "ema_12", "ema_26"]:
    if new_col not in buys.columns:
        print(f"  [WARNING] {new_col} không có trong enriched file")
        print(f"            → Chạy lại make_clean_data.py với data_builder.py mới")
        buys[new_col] = np.nan


# ════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    # Investor profile
    "risk_level",
    "investment_capacity_ordinal",
    "investor_trade_index",
    # Trading habit
    "trade_gap_days",
    "rolling_avg_trade_gap_last_10",
    "trades_per_investor_per_day",
    "digital_trade_flag",
    "consecutive_buy_streak",
    "rolling_buy_ratio_last_5",
    "rolling_buy_ratio_last_20",
    "rolling_trade_freq_5",
    # Position sizing
    "position_size_ratio",
    "position_size_spike_flag",
    "capital_acceleration_ratio",
    "rolling_avg_position_size_last_10",
    "position_size_to_volatility_ratio",
    # Asset switching
    "is_new_asset",
    "asset_diversity_last_10",
    "same_day_multiple_flag",
    # Crowd alignment
    "investor_alignment_with_crowd",
    "asset_popularity_zscore",
    # Market context
    "return_1d",
    "volatility_10d",
    "volatility_regime",
    "momentum_acceleration",
    # Asset statistical
    "total_value_pctrank_asset",
    "volatility_10d_pctrank_asset",
    "market_fomo_pressure_score",
    # Behavioral std
    "return_1d_rolling_std_10",
    "volatility_5d_rolling_std_10",
    # ── NEW market features ──────────────────────────────────────
    "volatility_ratio",
    "market_breadth",
    "ema_12",
    "ema_26",
]

# Join với snorkel labels
result = buys[["tx_id", "investor_id", "timestamp"] + FEATURE_COLS].merge(
    labels[["tx_id", "fomo_prob", "all_abstain"]],
    on="tx_id", how="inner"
)

# Drop all-abstain
before = len(result)
result = result[~result["all_abstain"]].drop(columns=["all_abstain"])
print(f"\n  Dropped all-abstain: {before - len(result):,} rows")
print(f"  Training set: {len(result):,} rows")

result.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved: {OUTPUT_FILE}")
print(f"  Shape: {result.shape}")
print(f"  Features: {len(FEATURE_COLS)}")

print("\nNaN summary:")
nan_pct = result[FEATURE_COLS].isna().mean() * 100
nan_pct = nan_pct[nan_pct > 0].sort_values(ascending=False)
for col, pct in nan_pct.items():
    print(f"  {col:<40} {pct:.1f}%")

print("\nfomo_prob distribution:")
print(f"  Mean  : {result['fomo_prob'].mean():.4f}")
print(f"  Median: {result['fomo_prob'].median():.4f}")
print(f"  > 0.65: {(result['fomo_prob'] > 0.65).mean()*100:.1f}%")
print(f"  < 0.35: {(result['fomo_prob'] < 0.35).mean()*100:.1f}%")
print(f"  0.35-0.65: {((result['fomo_prob'] >= 0.35) & (result['fomo_prob'] <= 0.65)).mean()*100:.1f}%")
print("\n✓ Done. Next: train XGBoost.")