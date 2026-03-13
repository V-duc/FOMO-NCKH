import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from constants import (
    ENRICHED_TRADES_TRAIN_FILE,
    OUTPUT_DIR,
)

# ── Configuration ─────────────────────────────────────────────────────────────
WINDOW          = "5D"   # Must match the window used in build_feature_table()
MIN_BUYS_WINDOW = 1      # Discard windows with 0 BUY orders (no market context)
OUTPUT_FILE     = f"{OUTPUT_DIR}/cluster_market_context.csv"

# Source column names in enriched_trades (output of make_clean_data.py)
SRC_PRICE_ABOVE_MA20    = "price_above_ma20"   # = market_price / ma_20d, raw ratio
SRC_VOLATILITY_5D       = "volatility_5d"
SRC_RSI_14              = "rsi_14"
SRC_RETURN_1D           = "return_1d"
SRC_RETURN_5D           = "return_5d"          # 5-day cumulative return -> TH_CUMULATIVE_RETURN_5D

SOURCE_COLS = [
    SRC_PRICE_ABOVE_MA20,
    SRC_VOLATILITY_5D,
    SRC_RSI_14,
    SRC_RETURN_1D,
    SRC_RETURN_5D,
]

# Raw column names (post-aggregation, pre-scaling) — used for threshold extraction
# Mapping to output thresholds:
#   raw_return_1d           → TH_RETURN_1D
#   raw_cumulative_return_5d→ TH_CUMULATIVE_RETURN_5D
#   raw_rsi_14              → TH_RSI_LEVEL
#   raw_price_above_ma20    → TH_PRICE_ABOVE_MA20 (e.g., 1.08 = price is 8% above MA20)
#   raw_volatility_5d       → auxiliary feature for clustering, no threshold extracted
RAW_COLS = [
    "raw_return_1d",
    "raw_cumulative_return_5d",
    "raw_rsi_14",
    "raw_price_above_ma20",
    "raw_volatility_5d",
]

# Final cluster column names (post-StandardScaler) — input for KMeans
CLUSTER_COLS = [
    "RETURN_1D",
    "CUMULATIVE_RETURN_5D",
    "RSI_14",
    "PRICE_ABOVE_MA20",
    "VOLATILITY_5D",
]

# Mapping raw features → threshold names (used in cluster_fomo.py for results)
RAW_TO_THRESHOLD = {
    "raw_return_1d":            "TH_RETURN_1D",
    "raw_cumulative_return_5d": "TH_CUMULATIVE_RETURN_5D",
    "raw_rsi_14":               "TH_RSI_LEVEL",
    "raw_price_above_ma20":     "TH_PRICE_ABOVE_MA20",
}


# ── Step 1: Load & validate ────────────────────────────────────────────────

def load_buy_trades(file: str) -> pd.DataFrame:
    """
    Load enriched trades, retain only BUY transactions and necessary columns.

    :param file: Path to enriched_trades_train.csv
    :return: DataFrame containing only BUY rows with complete market context
    """
    print(f"[load] Reading {file}...")
    trades = pd.read_csv(file, parse_dates=["timestamp"])

    print(f"[load] Total rows      : {len(trades):,}")
    print(f"[load] Unique investors: {trades['investor_id'].nunique():,}")
    print(f"[load] Columns         : {list(trades.columns)}")

    # Check if required columns exist
    missing = [c for c in SOURCE_COLS if c not in trades.columns]
    if missing:
        raise ValueError(
            f"Missing columns in enriched_trades: {missing}\n"
            f"→ Please run make_clean_data.py first to generate the full market context."
        )

    # Filter for BUY orders — we focus on market conditions at the time of entry
    buys = trades[trades["side"] == "BUY"].copy()
    print(f"\n[load] BUY rows        : {len(buys):,} / {len(trades):,} "
          f"({len(buys)/len(trades)*100:.1f}%)")

    # Drop rows missing market data (normal during the warmup period)
    before = len(buys)
    buys = buys.dropna(subset=SOURCE_COLS)
    dropped = before - len(buys)
    if dropped > 0:
        print(f"[load] Dropped         : {dropped:,} BUY rows missing market data "
              f"(warmup period — this is expected)")

    return buys


# ── Step 2: Aggregate by (investor_id × window) ─────────────────────────

def aggregate_by_window(buys: pd.DataFrame, window: str = WINDOW) -> pd.DataFrame:
    """
    Group BUY trades by (investor_id × time window), calculating feature means.

    Each output row represents the AVERAGE market conditions an investor faced
    across all their BUY orders within that specific window.

    Note: price_above_ma20 remains as a raw ratio (not z-scored yet) 
    because TH_PRICE_ABOVE_MA20 needs to be human-readable: 1.08 = "8% above MA20".

    :param buys: BUY transactions DataFrame
    :param window: Pandas frequency string, default "5D"
    :return: DataFrame (investor_id, window_start, n_buys_in_window, raw_* columns)
    """
    grouped = (
        buys
        .groupby(["investor_id", pd.Grouper(key="timestamp", freq=window)])
        .agg(
            raw_return_1d             = (SRC_RETURN_1D,           "mean"),
            raw_cumulative_return_5d  = (SRC_RETURN_5D,           "mean"),
            raw_rsi_14                = (SRC_RSI_14,              "mean"),
            raw_price_above_ma20      = (SRC_PRICE_ABOVE_MA20,    "mean"),
            raw_volatility_5d         = (SRC_VOLATILITY_5D,       "mean"),
            n_buys_in_window          = (SRC_RETURN_1D,           "count"),
        )
        .reset_index()
        .rename(columns={"timestamp": "window_start"})
    )

    # Drop empty windows (n_buys = 0) — caused by Grouper creating bins for all dates
    before = len(grouped)
    grouped = grouped[grouped["n_buys_in_window"] >= MIN_BUYS_WINDOW].copy()
    print(f"[aggregate] Total windows  : {before:,}")
    print(f"[aggregate] Windows kept   : {len(grouped):,} "
          f"(dropped {before - len(grouped):,} windows with no BUY activity)")
    print(f"[aggregate] Unique investors: {grouped['investor_id'].nunique():,}")

    return grouped


# ── Step 4: StandardScaler ─────────────────────────────────────────────────

def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Normalize raw features using StandardScaler."""
    scaler = StandardScaler()
    df[CLUSTER_COLS] = scaler.fit_transform(df[RAW_COLS])

    print(f"\n[scale] StandardScaler fit on {len(df):,} windows")
    header = f"{'Feature':<28} {'μ_raw':>10} {'σ_raw':>10}   {'μ_scaled':>10} {'σ_scaled':>10}"
    print(header)
    print("─" * len(header))
    for raw, col, mu, sigma in zip(RAW_COLS, CLUSTER_COLS, scaler.mean_, scaler.scale_):
        s = df[col]
        print(f"{col:<28} {mu:>10.4f} {sigma:>10.4f}   "
              f"{s.mean():>10.4f} {s.std():>10.4f}")

    return df, scaler


# ── Step 5: Quality check ──────────────────────────────────────────────────

def print_quality_report(df: pd.DataFrame) -> None:
    """Print data quality report before saving results."""
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    print(f"\nShape                 : {df.shape}")
    print(f"Unique investors      : {df['investor_id'].nunique():,}")
    windows_per_inv = df.groupby("investor_id").size()
    print(f"Windows per investor  : "
          f"min={windows_per_inv.min()}, "
          f"median={windows_per_inv.median():.0f}, "
          f"max={windows_per_inv.max()}")

    nan_counts = df[CLUSTER_COLS].isna().sum()
    if nan_counts.any():
        print(f"\n[WARNING] NaN detected in cluster features:")
        print(nan_counts[nan_counts > 0].to_string())
    else:
        print(f"\n✓ No NaN values found in the 5 cluster features")

    print(f"\nDistribution of scaled cluster features:")
    print(df[CLUSTER_COLS].describe().round(3).to_string())

    # Strong outlier warning (|z| > 4)
    print()
    for col in CLUSTER_COLS:
        n_out = (df[col].abs() > 4).sum()
        flag  = " ← consider clipping before clustering" if n_out > 0 else ""
        print(f"  {col:<28}: {n_out:>4} outliers (|z|>4) = "
              f"{n_out/len(df)*100:.1f}%{flag}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def build_cluster_data() -> pd.DataFrame:
    """
    Full pipeline execution: enriched_trades → cluster-ready feature table.

    Output columns:
        investor_id | window_start | n_buys_in_window
        raw_return_1d | raw_cumulative_return_5d | raw_rsi_14
        raw_price_above_ma20 | raw_volatility_5d
        RETURN_1D | CUMULATIVE_RETURN_5D | RSI_14 | PRICE_ABOVE_MA20 | VOLATILITY_5D

    Returns:
        pd.DataFrame ready for KMeans clustering.
    """
    print("=" * 60)
    print("STEP 1: Load & filter BUY trades")
    print("=" * 60)
    buys = load_buy_trades(ENRICHED_TRADES_TRAIN_FILE)

    print("\n" + "=" * 60)
    print(f"STEP 2: Aggregate by (investor_id × {WINDOW} window)")
    print("=" * 60)
    df = aggregate_by_window(buys, window=WINDOW)

    print("\n" + "=" * 60)
    print("STEP 3: StandardScaler normalization")
    print("=" * 60)
    df, _ = scale_features(df)

    print_quality_report(df)

    # Sort and organize output columns
    output_cols = (
        ["investor_id", "window_start", "n_buys_in_window"]
        + RAW_COLS
        + CLUSTER_COLS
    )
    df = df[output_cols].sort_values(["investor_id", "window_start"]).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 60}")
    print(f"✓ Saved → {OUTPUT_FILE}")
    print(f"  Shape : {df.shape}")
    print(f"{'=' * 60}")
    print("\nSample output (First 5 rows):")
    print(df.head(5).to_string(index=False))

    return df


if __name__ == "__main__":
    build_cluster_data()