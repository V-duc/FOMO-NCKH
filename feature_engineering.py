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
    fomo_features.csv          — 1 row per BUY transaction, ~28 features + fomo_prob

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

# Chỉ làm việc với BUY
buys = df[df["side"] == "BUY"].copy()
buys = buys.sort_values(["investor_id", "timestamp"]).reset_index(drop=True)
print(f"  BUY transactions: {len(buys):,}")

# Toàn bộ trades (BUY + SELL) để tính SELL history
all_trades = df.sort_values(["investor_id", "timestamp"]).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 1 — INVESTOR PROFILE (Hồ sơ nhà đầu tư)
# Mục đích: Capture đặc điểm cố định của investor — người có risk tolerance
# thấp hoặc capacity nhỏ nhưng vẫn FOMO là signal mạnh hơn.
# ════════════════════════════════════════════════════════════════════════════

# 1. risk_level
# Giải thích: Khẩu vị rủi ro được ngân hàng encode (1=Conservative → 4=Aggressive).
# Giả định: Investor conservative mà FOMO là bất thường hơn investor aggressive.
# Đã có sẵn trong enriched file, không cần tính thêm.

# 2. investment_capacity_ordinal
# Giải thích: Năng lực tài chính (1=<30k€ → 4=>300k€).
# Giả định: Investor nhỏ mà đặt lệnh lớn = over-committed = FOMO signal.
# Đã có sẵn trong enriched file.

# 3. investor_trade_index — "kinh nghiệm" của investor tại thời điểm giao dịch
# Giải thích: Đây là lệnh thứ mấy của investor này (tính cả BUY lẫn SELL).
# Giả định: Investor mới (index thấp) dễ FOMO hơn investor lâu năm.
# Lý thuyết: Barber & Odean (2001) — learning effect giảm overconfidence.
buys["investor_trade_index"] = buys.groupby("investor_id").cumcount()


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 2 — TRADING HABIT (Thói quen giao dịch)
# Mục đích: Đo nhịp và pattern giao dịch bình thường của investor.
# Deviation từ thói quen = signal bất thường.
# ════════════════════════════════════════════════════════════════════════════

# 4. trade_gap_days — khoảng nghỉ giữa hai lệnh BUY liên tiếp
# Giải thích: Số ngày kể từ lệnh BUY trước của cùng investor.
# Giả định: Khoảng nghỉ rất ngắn = giao dịch dồn dập = impulsive behavior.
# Lý thuyết: Action Bias (Glaser & Weber, 2007).
# CHÚ Ý: Khác với LF_trade_cluster (dùng days_since_last_buy == 0 same-asset).
#         Feature này dùng tất cả BUY, không filter theo asset.
buys["trade_gap_days"] = (
    buys.groupby("investor_id")["timestamp"]
    .diff().dt.days
)

# 5. rolling_avg_trade_gap_last_10 — trung bình khoảng nghỉ trong 10 lệnh gần nhất
# Giải thích: Baseline thói quen của investor — họ thường giao dịch mỗi bao nhiêu ngày.
# Giả định: Nếu trade_gap_days hiện tại << rolling average → đang giao dịch bất thường dồn dập.
buys["rolling_avg_trade_gap_last_10"] = (
    buys.groupby("investor_id")["trade_gap_days"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# 6. trades_per_investor_per_day — số lệnh BUY trong cùng ngày
# Giải thích: Investor đặt bao nhiêu lệnh BUY trong 1 ngày.
# Giả định: Nhiều lệnh trong 1 ngày = panic buying / FOMO dồn vốn.
# Lý thuyết: Impulsivity measure (Barber & Odean, 2008).
buys["trades_per_investor_per_day"] = (
    buys.groupby(["investor_id", buys["timestamp"].dt.date]).transform("size")
)

# 7. digital_trade_flag — giao dịch qua Internet Banking
# Giải thích: 1 nếu kênh là Internet Banking.
# Giả định: Internet Banking = tự quyết định, không qua tư vấn = dễ FOMO hơn.
# Lý thuyết: Online trading impulsivity (Barber & Odean, 2002 — "Online Investors").
buys["digital_trade_flag"] = (buys["channel"] == "Internet Banking").astype(int)

# 8. consecutive_buy_streak — số lệnh BUY liên tiếp đến hiện tại
# Giải thích: Đếm số lệnh BUY liên tiếp không có SELL xen giữa (trên toàn danh mục).
# Giả định: Streak BUY dài = đang trong trạng thái hưng phấn, không rebalance.
# Lý thuyết: Disposition effect ngược — FOMO investor giữ nguyên bullish bias.
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

# 9. rolling_buy_ratio_last_5 — tỷ lệ BUY trong 5 giao dịch gần nhất (BUY + SELL)
# Giải thích: Trong 5 lệnh gần nhất, bao nhiêu % là BUY.
# Giả định: Tỷ lệ BUY cao = thiên vị bullish = dễ FOMO.
# Lý thuyết: Bullish bias (Barber & Odean, 2001).
all_trades["is_buy"] = (all_trades["side"] == "BUY").astype(int)
all_trades["rolling_buy_ratio_last_5"] = (
    all_trades.groupby("investor_id")["is_buy"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
)
buys = buys.merge(
    all_trades[["tx_id", "rolling_buy_ratio_last_5"]],
    on="tx_id", how="left"
)

# 10. rolling_buy_ratio_last_20 — tỷ lệ BUY trong 20 giao dịch gần nhất
# Giải thích: Phiên bản dài hơn của rolling_buy_ratio — đo xu hướng trung hạn.
# Giả định: Nếu ratio_5 >> ratio_20 → đang tăng tốc bullish gần đây = FOMO signal.
all_trades["rolling_buy_ratio_last_20"] = (
    all_trades.groupby("investor_id")["is_buy"]
    .transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
)
buys = buys.merge(
    all_trades[["tx_id", "rolling_buy_ratio_last_20"]],
    on="tx_id", how="left"
)

# 11. rolling_trade_freq_5 — tần suất giao dịch (số lệnh / ngày) trong 5 lệnh gần nhất
# Giải thích: Đo tốc độ giao dịch — bao nhiêu lệnh per ngày trong giai đoạn gần nhất.
# Giả định: Tăng tốc giao dịch = dấu hiệu hưng phấn.
buys["rolling_trade_freq_5"] = (
    buys.groupby("investor_id")["trade_gap_days"]
    .transform(lambda x: 1 / (x.shift(1).rolling(5, min_periods=2).mean() + 1))
)


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 3 — POSITION SIZING (Quy mô vị thế)
# Mục đích: FOMO investor thường over-commit vốn — đặt lệnh lớn bất thường.
# ════════════════════════════════════════════════════════════════════════════

# 12. position_size_ratio — tỷ lệ lệnh / capacity tài chính
# Giải thích: totalValue của lệnh / investment_capacity_value (midpoint €).
# Giả định: Ratio cao = đang bet lớn so với khả năng tài chính = FOMO over-commit.
# Đã tính trong data_loader, không cần tính lại.

# 13. position_size_spike_flag — lệnh có lớn bất thường không (P95 cá nhân)
# Giải thích: 1 nếu lệnh này nằm trong top 5% lớn nhất lịch sử của chính investor đó.
# Giả định: Đặt lệnh lớn bất thường = "YOLO moment" = FOMO signal mạnh.
# CHÚ Ý: LF_value_spike dùng P90, feature này dùng P95 — stricter, ít overlap hơn.
#         Hơn nữa đây là binary flag, LF output là soft label — hai dimension khác nhau.
# Lý thuyết: Sensation seeking / lottery demand (Kumar et al., 2011).
p95_value = (
    buys.groupby("investor_id")["totalValue"]
    .transform(lambda x: x.shift(1).rolling(50, min_periods=5).quantile(0.95))
)
buys["position_size_spike_flag"] = (buys["totalValue"] > p95_value).astype(float)
buys.loc[p95_value.isna(), "position_size_spike_flag"] = np.nan

# 14. capital_acceleration_ratio — lệnh hiện tại / trung bình 10 lệnh gần nhất
# Giải thích: Lệnh này lớn hơn thói quen bao nhiêu lần.
# Giả định: Ratio > 2 = đang tăng gấp đôi bet = dấu hiệu mất kiểm soát cảm xúc.
# Lý thuyết: Capital indiscipline (Statman et al., 2006).
rolling_mean_10 = (
    buys.groupby("investor_id")["totalValue"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)
buys["capital_acceleration_ratio"] = buys["totalValue"] / rolling_mean_10.replace(0, np.nan)

# 15. rolling_avg_position_size_last_10 — trung bình position size trong 10 lệnh gần nhất
# Giải thích: Baseline position size của investor — họ thường đặt bao nhiêu.
# Giả định: Dùng làm context cho capital_acceleration_ratio.
buys["rolling_avg_position_size_last_10"] = (
    buys.groupby("investor_id")["totalValue"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# 16. position_size_to_volatility_ratio — position size / volatility thị trường
# Giải thích: Đặt lệnh lớn khi thị trường đang biến động mạnh.
# Giả định: Lệnh lớn trong high-volatility = panic / FOMO amplified by fear.
# Lý thuyết: Prospect theory — loss aversion increases risk-taking when already losing.
buys["position_size_to_volatility_ratio"] = (
    buys["totalValue"] / (buys["volatility_10d"].replace(0, np.nan) * buys["totalValue"].mean())
)


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 4 — ASSET SWITCHING & DIVERSITY (Hành vi chuyển đổi tài sản)
# Mục đích: FOMO investor thường nhảy giữa các assets theo hot trend.
# ════════════════════════════════════════════════════════════════════════════

# 17. is_new_asset — asset này có phải mới với investor không
# Giải thích: 1 nếu investor chưa từng giao dịch asset này trước đây.
# Giả định: Mua asset hoàn toàn mới = chạy theo trend mới = FOMO signal.
# Lý thuyết: Attention-driven buying (Barber & Odean, 2008).
# FIX: so sánh trực tiếp với asset lệnh trước (không dùng rolling trên string)
buys["prev_asset_id"] = buys.groupby("investor_id")["asset_id"].shift(1)
buys["is_new_asset"] = (
    (buys["asset_id"] != buys["prev_asset_id"]) & buys["prev_asset_id"].notna()
).astype(float)
buys.loc[buys["prev_asset_id"].isna(), "is_new_asset"] = np.nan

# 18. asset_diversity_last_10 — số asset unique trong 10 lệnh BUY gần nhất
# Giải thích: Investor đang giao dịch bao nhiêu asset khác nhau gần đây.
# Giả định: Diversity thấp (chỉ 1-2 asset) khi đang FOMO = all-in vào 1 trend.
#           Diversity cao = đang scatter = chasing nhiều trends cùng lúc.
# Lý thuyết: Portfolio concentration và overconfidence (Statman, 1987).
# FIX: encode asset_id thành int trước khi dùng rolling.apply
buys["asset_id_code"] = buys["asset_id"].astype("category").cat.codes
buys["asset_diversity_last_10"] = (
    buys.groupby("investor_id")["asset_id_code"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3)
               .apply(lambda w: len(set(w.astype(int))), raw=True))
)
buys = buys.drop(columns=["asset_id_code"])

# 19. same_day_multiple_flag — investor có đặt nhiều lệnh BUY trong ngày không
# Giải thích: 1 nếu investor đặt >= 2 lệnh BUY trong cùng ngày.
# Giả định: Multiple BUY cùng ngày = impulsive / averaging down / FOMO chasing.
# CHÚ Ý: Khác LF_trade_cluster (same-day same-asset). Feature này là any asset.
buys["same_day_multiple_flag"] = (buys["trades_per_investor_per_day"] >= 2).astype(int)


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 5 — CROWD ALIGNMENT (Hành vi đám đông)
# Mục đích: FOMO thường đi kèm với hành vi bầy đàn — mua khi nhiều người mua.
# ════════════════════════════════════════════════════════════════════════════

# 20. investor_alignment_with_crowd — mua khi thị trường đang bullish
# Giải thích: 1 nếu tỷ lệ BUY trên toàn thị trường > 70% trong ngày đó.
# Giả định: Mua khi đám đông cũng đang mua mạnh = herding behavior.
# CHÚ Ý: Khác LF_herding_crowd (đếm BUY per specific asset).
#         Feature này nhìn toàn thị trường, không phải per asset.
# Lý thuyết: Herding (Hirshleifer & Teoh, 2003).
daily_market = (
    df.groupby(df["timestamp"].dt.date)["side"]
    .apply(lambda x: (x == "BUY").sum() / len(x))
    .reset_index()
    .rename(columns={"timestamp": "date", "side": "market_buy_ratio"})
)
buys["date"] = buys["timestamp"].dt.date
buys = buys.merge(daily_market, on="date", how="left")
buys["investor_alignment_with_crowd"] = (buys["market_buy_ratio"] > 0.7).astype(int)

# 21. asset_popularity_zscore — độ bùng nổ giao dịch của asset so với lịch sử
# Giải thích: Z-score của số lệnh BUY asset này hôm nay vs lịch sử 60 ngày.
# Giả định: Asset đang được giao dịch bất thường nhiều = đang hot = FOMO magnet.
# CHÚ Ý: LF_herding_crowd dùng P95 threshold binary. Feature này dùng Z-score continuous.
#         Z-score cho XGBoost nhiều thông tin hơn (biết "hot đến mức nào").
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
# NHÓM 6 — MARKET CONTEXT SẠCH (không phải LF proxy)
# Mục đích: Market context giúp XGBoost hiểu environment — nhưng phải là
# dimension KHÁC với những gì LF đã dùng.
# LF đã dùng: return_5d, rsi_14, bollinger, asset_buy_count per asset.
# Features dưới đây dùng: return_1d, volatility, momentum acceleration.
# ════════════════════════════════════════════════════════════════════════════

# 22. return_1d — lợi nhuận 1 ngày của asset
# Giải thích: Giá hôm nay tăng/giảm bao nhiêu % so với hôm qua.
# Giả định: return_1d cao = "tin tức tốt hôm nay" = attention shock = FOMO trigger.
# CHÚ Ý: LF dùng return_5d (5 ngày). return_1d là dimension khác — shock ngắn hạn.
# Lý thuyết: Attention shock (Da, Engelberg & Gao, 2011).
# Đã có sẵn trong enriched file.

# 23. volatility_10d — độ biến động 10 ngày
# Giải thích: Rolling std của daily return trong 10 ngày.
# Giả định: Volatility cao = môi trường uncertainty = FOMO và fear đều tăng.
# CHÚ Ý: LF không dùng volatility trực tiếp.
# Đã có sẵn trong enriched file.

# 24. volatility_regime — chế độ biến động (low/medium/high)
# Giải thích: Phân loại volatility_10d thành 3 regime: 0=low, 1=medium, 2=high.
# Giả định: High volatility regime = FOMO và panic buying phổ biến hơn.
buys["volatility_regime"] = pd.qcut(
    buys["volatility_10d"].fillna(buys["volatility_10d"].median()),
    q=3, labels=[0, 1, 2]
).astype(float)

# 25. momentum_acceleration — return_3d trừ return_10d
# Giải thích: Đo xem đà tăng đang tăng tốc hay giảm tốc.
# Giả định: momentum_acceleration > 0 = đà tăng đang mạnh lên = FOMO environment.
# CHÚ Ý: Không phải return_5d. Đây là DERIVATIVE của momentum, không phải level.
#         Có thể tính bằng: (return_3d) - (return_10d).
#         LF không dùng feature này.
if "return_3d" in buys.columns and "return_10d" in buys.columns:
    buys["momentum_acceleration"] = buys["return_3d"] - buys["return_10d"]
else:
    # Tính từ market data nếu không có sẵn
    buys["momentum_acceleration"] = np.nan
    print("  [WARNING] return_3d hoặc return_10d không có trong enriched file")


# ════════════════════════════════════════════════════════════════════════════
# NHÓM 7 — ASSET STATISTICAL (Vị trí thống kê của asset)
# Mục đích: Normalize để XGBoost so sánh được across assets có quy mô khác nhau.
# ════════════════════════════════════════════════════════════════════════════

# 26. total_value_pctrank_asset — percentile rank của lệnh so với lịch sử asset
# Giải thích: Lệnh này nằm ở bách phân vị nào so với tất cả lệnh của asset đó.
# Giả định: Rank cao = lệnh lớn bất thường so với asset → dòng tiền lớn chảy vào.
# CHÚ Ý: LF_value_spike so sánh với P90 CỦA INVESTOR. Feature này so sánh với
#         lịch sử CỦA ASSET — hai dimension khác nhau.
buys["total_value_pctrank_asset"] = (
    buys.groupby("asset_id")["totalValue"]
    .transform(lambda x: x.rank(pct=True))
)

# 27. volatility_10d_pctrank_asset — percentile rank của volatility asset này
# Giải thích: Hôm nay asset này đang biến động ở mức nào trong lịch sử của nó.
# Giả định: Volatility rank cao = asset đang bất ổn hơn bình thường = FOMO environment.
buys["volatility_10d_pctrank_asset"] = (
    buys.groupby("asset_id")["volatility_10d"]
    .transform(lambda x: x.rank(pct=True))
)

# 28. market_fomo_pressure_score — composite score áp lực FOMO của thị trường
# Giải thích: Kết hợp asset_popularity_zscore + investor_alignment_with_crowd + volatility_regime.
# Giả định: Khi cả 3 tín hiệu đều cao → môi trường FOMO mạnh nhất.
# CHÚ Ý: Composite score dùng các features đã tính ở trên, KHÔNG dùng rsi hay return_5d.
buys["market_fomo_pressure_score"] = (
    buys["asset_popularity_zscore"].fillna(0).clip(-3, 3) / 3 * 0.4 +
    buys["investor_alignment_with_crowd"].fillna(0) * 0.3 +
    buys["volatility_regime"].fillna(1) / 2 * 0.3
).clip(0, 1)


# ════════════════════════════════════════════════════════════════════════════
# ROLLING HISTORY — ĐỘ BẤT ỔN HÀNH VI
# Mục đích: Investor hay thay đổi hành vi = không có chiến lược = dễ FOMO.
# ════════════════════════════════════════════════════════════════════════════

# 29. return_1d_rolling_std_10 — std của return_1d tại các lệnh gần nhất
# Giải thích: Investor hay mua khi return_1d biến thiên nhiều = không nhất quán.
# Giả định: Std cao = mua lung tung không theo strategy = behavioral inconsistency.
buys["return_1d_rolling_std_10"] = (
    buys.groupby("investor_id")["return_1d"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).std())
)

# 30. volatility_5d_rolling_std_10 — std của volatility asset tại các lệnh gần nhất
# Giải thích: Investor có xu hướng mua khi asset đang ở các mức volatility khác nhau.
# Giả định: Std cao = không có ngưỡng volatility nhất quán = impulsive.
buys["volatility_5d_rolling_std_10"] = (
    buys.groupby("investor_id")["volatility_5d"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).std())
)


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
    # Market context (sạch)
    "return_1d",
    "volatility_10d",
    "volatility_regime",
    "momentum_acceleration",
    # Asset statistical
    "total_value_pctrank_asset",
    "volatility_10d_pctrank_asset",
    "market_fomo_pressure_score",
    # Behavioral inconsistency
    "return_1d_rolling_std_10",
    "volatility_5d_rolling_std_10",
]

# Join với snorkel labels
result = buys[["tx_id", "investor_id", "timestamp"] + FEATURE_COLS].merge(
    labels[["tx_id", "fomo_prob", "all_abstain"]],
    on="tx_id", how="inner"
)

# Drop all-abstain theo quyết định đã chốt
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
print(f"  > 0.5 : {(result['fomo_prob'] > 0.5).mean()*100:.1f}%")
print("\n✓ Done. Next: train XGBoost với fomo_prob làm soft label.")
