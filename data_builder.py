import pandas as pd
import numpy as np


def build_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare trades data by converting timestamps and sorting.

    :param trades: DataFrame containing trade data
    :return: Prepared DataFrame sorted by (investor_id, timestamp)
    """
    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades = trades.sort_values(["investor_id", "timestamp"]).reset_index(drop=True)
    return trades


def _reindex_to_business_days(market: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex market data to business day calendar per asset.
    Forward fill gaps (weekends, holidays).
    """
    start = market["timestamp"].min()
    end   = market["timestamp"].max()
    full_bday_idx = pd.date_range(start=start, end=end, freq="B")

    reindexed_parts = []
    for asset_id, group in market.groupby("asset_id"):
        group = group.set_index("timestamp").reindex(full_bday_idx)
        group["asset_id"] = asset_id
        group["market_price"] = group["market_price"].ffill()
        group.index.name = "timestamp"
        reindexed_parts.append(group.reset_index())

    result = pd.concat(reindexed_parts, ignore_index=True)
    return result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)


def _add_market_returns(market: pd.DataFrame, horizons=(1, 5)) -> pd.DataFrame:
    """Compute market returns per asset."""
    for h in horizons:
        market[f"return_{h}d"] = (
            market
            .groupby("asset_id")["market_price"]
            .pct_change(h)
        )
    return market


def _add_market_volatilities(market: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    """Compute market volatility as rolling std of daily returns."""
    for w in windows:
        market[f"volatility_{w}d"] = (
            market
            .groupby("asset_id")["return_1d"]
            .rolling(w)
            .std()
            .reset_index(level=0, drop=True)
        )
    return market


def _add_moving_averages(market: pd.DataFrame, windows=(5, 20)) -> pd.DataFrame:
    """
    Compute moving averages and price-vs-MA ratio.
    Columns: ma_5d, ma_20d, price_above_ma20
    """
    for w in windows:
        market[f"ma_{w}d"] = (
            market
            .groupby("asset_id")["market_price"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
    market["price_above_ma20"] = market["market_price"] / market["ma_20d"]
    return market


def _add_rsi(market: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute RSI (Relative Strength Index) — Wilder (1978).
    Columns: rsi_14
    """
    def _compute_rsi(prices: pd.Series, period: int) -> pd.Series:
        delta    = prices.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    market[f"rsi_{period}"] = (
        market
        .groupby("asset_id")["market_price"]
        .transform(lambda x: _compute_rsi(x, period))
    )
    return market


def _add_price_distance(market: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    [NEW] Tính khoảng cách giá so với đỉnh 20 ngày.

    price_distance_high = market_price / rolling_max(20d)
        = 1.0 khi giá đang ở đỉnh
        < 1.0 khi giá đã xuống khỏi đỉnh

    Giả định: FOMO xảy ra khi giá gần đỉnh (= 0.95-1.0)
    Lý thuyết: Barber & Odean (2008) — attention-driven buying at recent highs.

    Columns: price_distance_high
    """
    rolling_max = (
        market
        .groupby("asset_id")["market_price"]
        .transform(lambda x: x.rolling(window, min_periods=5).max())
    )
    market["price_distance_high"] = (
        market["market_price"] / rolling_max.replace(0, np.nan)
    )
    return market


def _add_volatility_ratio(market: pd.DataFrame) -> pd.DataFrame:
    """
    [NEW] Tính tỷ lệ volatility ngắn hạn / dài hạn.

    volatility_ratio = volatility_5d / volatility_10d
        > 1.0: volatility đang tăng (thị trường bất ổn hơn)
        < 1.0: volatility đang giảm (thị trường ổn định hơn)

    Giả định: volatility_ratio tăng đột biến = market shock → trigger FOMO.

    Columns: volatility_ratio
    """
    market["volatility_ratio"] = (
        market["volatility_5d"] / market["volatility_10d"].replace(0, np.nan)
    )
    return market


def _add_market_breadth(market: pd.DataFrame) -> pd.DataFrame:
    """
    [NEW] Tính market breadth = % assets đang tăng giá trong ngày.

    market_breadth gần 1.0: hầu hết thị trường đang tăng → bull signal
    market_breadth gần 0.0: hầu hết thị trường đang giảm → bear signal

    CHÚ Ý: Market-level feature — tất cả assets cùng ngày có cùng giá trị.
    Lý thuyết: Herding behavior (Hirshleifer & Teoh, 2003).

    Columns: market_breadth
    """
    daily_breadth = (
        market
        .groupby("timestamp")["return_1d"]
        .apply(lambda x: (x > 0).sum() / x.notna().sum()
               if x.notna().sum() > 0 else np.nan)
        .reset_index()
        .rename(columns={"return_1d": "market_breadth"})
    )
    market = market.merge(daily_breadth, on="timestamp", how="left")
    return market


def _add_macd(market: pd.DataFrame,
              fast: int = 12,
              slow: int = 26,
              signal: int = 9) -> pd.DataFrame:
    """
    [NEW] Compute MACD (Moving Average Convergence Divergence).

    MACD      = EMA_12 - EMA_26
    macd_signal = EMA_9 của MACD
    macd_hist   = MACD - macd_signal

    macd > 0 và tăng  → bullish momentum → FOMO environment
    macd_hist > 0 đột biến → bullish crossover → trigger FOMO

    Lý thuyết: Technical momentum (Jegadeesh & Titman, 1993).

    Columns: ema_12, ema_26, macd, macd_signal, macd_hist
    """
    market[f"ema_{fast}"] = (
        market
        .groupby("asset_id")["market_price"]
        .transform(lambda x: x.ewm(span=fast, adjust=False).mean())
    )
    market[f"ema_{slow}"] = (
        market
        .groupby("asset_id")["market_price"]
        .transform(lambda x: x.ewm(span=slow, adjust=False).mean())
    )
    market["macd"] = market[f"ema_{fast}"] - market[f"ema_{slow}"]
    market["macd_signal"] = (
        market
        .groupby("asset_id")["macd"]
        .transform(lambda x: x.ewm(span=signal, adjust=False).mean())
    )
    market["macd_hist"] = market["macd"] - market["macd_signal"]
    return market


def build_market(market: pd.DataFrame) -> pd.DataFrame:
    """
    Build enriched daily market table from close_prices.

    Pipeline:
        1. Reindex về business days + forward fill
        2. Tính returns
        3. Tính volatility
        4. Tính MA + price_above_ma20
        5. Tính RSI
        6. [NEW] Tính price_distance_high
        7. [NEW] Tính volatility_ratio
        8. [NEW] Tính market_breadth
        9. [NEW] Tính MACD + EMA

    Output columns:
        asset_id | timestamp | market_price
        | return_1d | return_5d
        | volatility_5d | volatility_10d
        | ma_5d | ma_20d | price_above_ma20
        | rsi_14
        | price_distance_high
        | volatility_ratio
        | market_breadth
        | ema_12 | ema_26 | macd | macd_signal | macd_hist

    :param market: DataFrame với columns: asset_id, timestamp, market_price
    :return: Enriched market DataFrame
    """
    required_cols = {"asset_id", "timestamp", "market_price"}
    missing = required_cols - set(market.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    market = market.copy()
    market = market.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    print("[build_market] Reindexing to business days...")
    market = _reindex_to_business_days(market)

    market = _add_market_returns(market)
    market = _add_market_volatilities(market)
    market = _add_moving_averages(market)
    market = _add_rsi(market)

    # [NEW] Additional market features
    print("[build_market] Computing additional market features...")
    market = _add_price_distance(market)
    market = _add_volatility_ratio(market)
    market = _add_market_breadth(market)
    market = _add_macd(market)

    print(f"[build_market] Done. Shape: {market.shape}, "
          f"Assets: {market['asset_id'].nunique()}, "
          f"Date range: {market['timestamp'].min().date()} → "
          f"{market['timestamp'].max().date()}")
    return market


def enrich_trades_with_market(trades: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich trades DataFrame by merging with market data on (asset_id, timestamp).

    :param trades: Enriched trades DataFrame
    :param market: Enriched market DataFrame
    :return: Trades với thêm market context columns
    """
    market = market.drop_duplicates(subset=['asset_id', 'timestamp'])
    enriched = trades.merge(market, on=["asset_id", "timestamp"], how="left")

    no_match = enriched["market_price"].isna().sum()
    if no_match > 0:
        pct = no_match / len(enriched) * 100
        print(f"[WARNING] {no_match:,} trades ({pct:.1f}%) không match được market data")

    return enriched