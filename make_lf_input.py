"""
make_lf_input.py — Tạo file input cho LF + Snorkel.

Input:  enriched_trades_train.csv  (market context per transaction)
        market_data.csv            (daily market data per asset — unique per asset/date)
Output: lf_input.csv               (BUY only, thêm LF features)

Cột được thêm:
    tx_id                — index gốc từ enriched file, dùng để join label sau
    days_since_last_buy  — ngày kể từ lệnh BUY trước của cùng investor (NaN = lệnh đầu)
    p90_trade_value      — P90 totalValue của investor (NaN nếu < 5 BUY = sparse)
    bollinger_upper      — ma_20d + 2 * std_20d (tính từ market_data.csv)
    price_above_bollinger— 1 nếu price > bollinger_upper, else 0, NaN nếu warmup

Chạy:
    python make_lf_input.py
"""

import pandas as pd
import numpy as np
from constants import (
    TRANSACTIONS_FILE,
    CUSTOMERS_FILE,
    CLOSE_PRICES_FILE,
    ASSETS_FILE,
    ENRICHED_TRADES_TRAIN_FILE,
    ENRICHED_TRADES_VAL_FILE,
    MARKET_DATA_FILE,
    OUTPUT_DIR,
)   


SPARSE_THRESHOLD = 5

# ── Load ──────────────────────────────────────────────────────────────────
print("Loading files...")
df  = pd.read_csv(ENRICHED_TRADES_TRAIN_FILE, parse_dates=["timestamp"])
mkt = pd.read_csv(MARKET_DATA_FILE,   parse_dates=["timestamp"])
print(f"  enriched: {len(df):,} rows | market: {len(mkt):,} rows")
OUTPUT_FILE = f'{OUTPUT_DIR}/lf_input.csv'

# ── BUY only ──────────────────────────────────────────────────────────────
buys = df[df["side"] == "BUY"].copy()
buys = buys.reset_index(drop=True)
buys = buys.sort_values(["investor_id", "timestamp"]).reset_index(drop=True)
print(f"  BUY: {len(buys):,} rows, {buys['investor_id'].nunique():,} investors")

# ── days_since_last_buy ───────────────────────────────────────────────────
buys["days_since_last_buy"] = (
    buys.groupby("investor_id")["timestamp"]
    .diff().dt.days
)
print(f"\n  days_since_last_buy NaN (first buy): {buys['days_since_last_buy'].isna().sum():,}")
print(f"  == 0 (same day): {(buys['days_since_last_buy']==0).sum():,} ({(buys['days_since_last_buy']==0).mean()*100:.1f}%)")

# asset_id của lệnh BUY trước của cùng investor
buys["prev_asset_id"] = buys.groupby("investor_id")["asset_id"].shift(1)

# ── p90_trade_value per investor ──────────────────────────────────────────
buy_counts = buys.groupby("investor_id").size()
sparse_ids = buy_counts[buy_counts < SPARSE_THRESHOLD].index

p90 = buys.groupby("investor_id")["totalValue"].quantile(0.9).rename("p90_trade_value")
p90[p90.index.isin(sparse_ids)] = np.nan
buys = buys.join(p90, on="investor_id")

n_sparse_tx = buys["investor_id"].isin(sparse_ids).sum()
print(f"\n  p90_trade_value NaN (sparse investors): {n_sparse_tx:,} ({n_sparse_tx/len(buys)*100:.1f}%)")

# ── Bollinger Band từ market_data.csv ────────────────────────────────────
# market_data unique per (asset_id, timestamp) → merge an toàn
print("\n  Computing Bollinger std_20d from market_data...")
mkt = mkt.sort_values(["asset_id", "timestamp"])
mkt["std_20d"] = (
    mkt.groupby("asset_id")["market_price"]
    .transform(lambda x: x.rolling(20, min_periods=10).std())
)
mkt["bollinger_upper"] = mkt["ma_20d"] + 2 * mkt["std_20d"]

# Merge vào buys — unique per (asset_id, timestamp) nên không explode
buys = buys.merge(
    mkt[["asset_id", "timestamp", "bollinger_upper"]],
    on=["asset_id", "timestamp"],
    how="left"
)

buys["price_above_bollinger"] = np.where(
    buys["bollinger_upper"].isna(), np.nan,
    (buys["price"] > buys["bollinger_upper"]).astype(float)
)

n_above = (buys["price_above_bollinger"] == 1).sum()
n_valid = buys["price_above_bollinger"].notna().sum()
print(f"  price_above_bollinger = 1: {n_above:,} ({n_above/n_valid*100:.1f}% of valid)")
print(f"  NaN (warmup period):       {buys['price_above_bollinger'].isna().sum():,}")

# ── Output ────────────────────────────────────────────────────────────────
output_cols = [
    "tx_id", "investor_id", "asset_id", "timestamp",
    # LF features mới
    "days_since_last_buy", "prev_asset_id", "p90_trade_value", 
    "bollinger_upper", "price_above_bollinger",
    # LF features từ enriched file
    "channel", "totalValue", "return_5d", "rsi_14",
    "volatility_5d", "market_price", "ma_20d", "price_above_ma20",
]

lf_input = buys[output_cols].copy()
lf_input.to_csv(OUTPUT_FILE, index=False)

print(f"\n✓ Saved: {OUTPUT_FILE}")
print(f"  Shape: {lf_input.shape}")
print("\nNaN summary:")
for col in output_cols:
    n = lf_input[col].isna().sum()
    if n > 0:
        print(f"  {col:<30} {n:,} ({n/len(lf_input)*100:.1f}%)")
        
        
        

# ── Validation set (Professional investors) ───────────────────────────────
print("\nProcessing validation set (Professional investors)...")
df_val = pd.read_csv(ENRICHED_TRADES_VAL_FILE, parse_dates=["timestamp"])
buys_val = df_val[df_val["side"] == "BUY"].copy()
buys_val = buys_val.sort_values(["investor_id", "timestamp"]).reset_index(drop=True)
print(f"  BUY val: {len(buys_val):,} rows, {buys_val['investor_id'].nunique():,} investors")

# Tính các features giống hệt train
buys_val["days_since_last_buy"] = (
    buys_val.groupby("investor_id")["timestamp"].diff().dt.days
)
buys_val["prev_asset_id"] = buys_val.groupby("investor_id")["asset_id"].shift(1)

buy_counts_val = buys_val.groupby("investor_id").size()
sparse_ids_val = buy_counts_val[buy_counts_val < SPARSE_THRESHOLD].index
p90_val = buys_val.groupby("investor_id")["totalValue"].quantile(0.9).rename("p90_trade_value")
p90_val[p90_val.index.isin(sparse_ids_val)] = np.nan
buys_val = buys_val.join(p90_val, on="investor_id")

buys_val = buys_val.merge(
    mkt[["asset_id", "timestamp", "bollinger_upper"]],
    on=["asset_id", "timestamp"], how="left"
)
buys_val["price_above_bollinger"] = np.where(
    buys_val["bollinger_upper"].isna(), np.nan,
    (buys_val["price"] > buys_val["bollinger_upper"]).astype(float)
)

lf_input_val = buys_val[["tx_id"] + [c for c in output_cols if c != "tx_id"]].copy()
lf_input_val.to_csv(f"{OUTPUT_DIR}/lf_input_val.csv", index=False)
print(f"✓ Saved: lf_input_val.csv | Shape: {lf_input_val.shape}")