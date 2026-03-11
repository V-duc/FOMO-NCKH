"""
make_clean_data.py — Bước 1 của pipeline, chạy 1 lần duy nhất.

Việc file này làm:
    1. Load raw CSV (transactions, customers, close_prices, assets)
    2. Filter: time period Jul2020-Nov2022, Stock only, customer type
    3. Enrich trades với market context (return, volatility, MA, RSI)
    4. Lưu ra 3 file intermediate vào data/output/

Output files:
    enriched_trades_train.csv  ← Mass + Premium investors
    enriched_trades_val.csv    ← Professional investors (dùng để validate sau)
    market_data.csv            ← enriched market data (Stock, Jul2020-Nov2022)

Tại sao tách ra file riêng thay vì gộp vào make_datasets.py:
    - Bước load + enrich chậm (reindex business days, tính RSI/MA cho ~285 assets)
    - Feature engineering sẽ thay đổi nhiều lần trong quá trình phát triển
    - Tách ra → chỉ chạy bước này 1 lần, các bước sau đọc từ enriched files

Chạy:
    python make_clean_data.py
"""

import os
import data_loader
import data_builder
import pandas as pd
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


def build_enriched_trades():
    """
    Full pipeline: raw CSV → enriched trades.

    Steps:
        1. Load + filter transactions (period, Stock only, customer type)
        2. Build enriched market data (returns, volatility, MA, RSI)
        3. Merge market context vào từng trade
        4. Save ra file
    """

    # ── Step 1: Load và filter ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading and filtering raw data...")
    print("=" * 60)

    # load_trade_data trả về 2 DataFrames: train (Mass+Premium) và val (Professional)
    # Xem data_loader.py để hiểu chi tiết các filter được áp dụng
    train_trades, val_trades = data_loader.load_trade_data(
        transactions_file=TRANSACTIONS_FILE,
        customers_file=CUSTOMERS_FILE,
        assets_file=ASSETS_FILE,
    )

    print(f"\nTrain trades shape:      {train_trades.shape}")
    print(f"Validation trades shape: {val_trades.shape}")
    print(f"\nTrain columns: {list(train_trades.columns)}")

    # ── Step 2: Build enriched market data ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Building enriched market data...")
    print("(Reindexing to business days + computing returns, MA, RSI)")
    print("=" * 60)

    # Load close prices — filter Stock only, period bắt đầu từ 2020-01
    # (rộng hơn period trades để return_5d đầu period không bị NaN)
    raw_market = data_loader.load_close_prices(
        file=CLOSE_PRICES_FILE,
        assets_file=ASSETS_FILE,
    )

    # Build: reindex → returns → volatility → MA → RSI
    market = data_builder.build_market(raw_market)
    print(f"\nMarket data shape: {market.shape}")
    print(f"Columns: {list(market.columns)}")

    # ── Step 3: Enrich trades với market context ───────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Enriching trades with market context...")
    print("=" * 60)

    # build_trades: sort theo (investor_id, timestamp)
    train_trades = data_builder.build_trades(train_trades)
    val_trades   = data_builder.build_trades(val_trades)

    # Merge market data vào từng trade theo (asset_id, timestamp)
    enriched_train = data_builder.enrich_trades_with_market(train_trades, market)
    enriched_val   = data_builder.enrich_trades_with_market(val_trades, market)

    print(f"\nEnriched train shape: {enriched_train.shape}")
    print(f"Enriched val shape:   {enriched_val.shape}")

    # Sanity check: xem sample output
    print(f"\nSample enriched train (3 rows):")
    print(enriched_train.head(3).to_string())

    # ── Step 4: Save ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Saving to output files...")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    enriched_train.to_csv(ENRICHED_TRADES_TRAIN_FILE, index=False)
    print(f"✓ Train trades saved:  {ENRICHED_TRADES_TRAIN_FILE}")

    enriched_val.to_csv(ENRICHED_TRADES_VAL_FILE, index=False)
    print(f"✓ Val trades saved:    {ENRICHED_TRADES_VAL_FILE}")

    market.to_csv(MARKET_DATA_FILE, index=False)
    print(f"✓ Market data saved:   {MARKET_DATA_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Period:          {enriched_train['timestamp'].min().date()} "
          f"→ {enriched_train['timestamp'].max().date()}")
    print(f"Unique investors (train): {enriched_train['investor_id'].nunique():,}")
    print(f"Unique investors (val):   {enriched_val['investor_id'].nunique():,}")
    print(f"Unique assets:            {enriched_train['asset_id'].nunique():,}")
    print(f"BUY transactions (train): {(enriched_train['side'] == 'BUY').sum():,}")
    print(f"SELL transactions (train):{(enriched_train['side'] == 'SELL').sum():,}")

    nan_pct = enriched_train.isnull().mean() * 100
    nan_pct = nan_pct[nan_pct > 0].round(1)
    if len(nan_pct):
        print(f"\nNaN % per column (train):")
        print(nan_pct.to_string())
        print("→ NaN ở return/MA/RSI đầu series là bình thường (warmup period)")
    else:
        print("\n✓ Không có NaN trong enriched train data")


if __name__ == "__main__":
    build_enriched_trades()