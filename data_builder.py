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
    # Sort theo investor trước để groupby window trong feature_builder hoạt động đúng
    trades = trades.sort_values(["investor_id", "timestamp"]).reset_index(drop=True)
    return trades


def _reindex_to_business_days(market: pd.DataFrame) -> pd.DataFrame:
    """
    [NEW] Reindex market data to business day calendar per asset.

    Vấn đề với code cũ:
        pct_change(h) lấy theo row index, không theo ngày thật.
        Gap cuối tuần (Fri→Mon = 3 calendar days) bị tính như 1 ngày.
        EDA: ~20% gaps là 3 ngày → return_1d thứ Hai bị inflate.

    Fix:
        Reindex về business days (Mon-Fri) → forward fill gap.
        Forward fill = convention chuẩn trong finance: nếu không có giá ngày X
        thì dùng giá ngày gần nhất trước đó (holiday, missing data).
        Sau reindex: pct_change(1) = đúng 1 trading day, pct_change(5) = đúng 5 trading days.

    :param market: DataFrame với columns: asset_id, timestamp, market_price (sorted)
    :return: DataFrame đã reindex, có thể có thêm rows (business days không có trong data gốc)
    """
    start = market["timestamp"].min()
    end   = market["timestamp"].max()
    full_bday_idx = pd.date_range(start=start, end=end, freq="B")  # B = business days

    reindexed_parts = []
    for asset_id, group in market.groupby("asset_id"):
        group = group.set_index("timestamp").reindex(full_bday_idx)
        group["asset_id"] = asset_id
        # Forward fill: ngày không có giá → dùng giá ngày gần nhất trước đó
        group["market_price"] = group["market_price"].ffill()
        group.index.name = "timestamp"
        reindexed_parts.append(group.reset_index())

    result = pd.concat(reindexed_parts, ignore_index=True)
    return result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)


def _add_market_returns(market: pd.DataFrame, horizons=(1, 5)) -> pd.DataFrame:
    """
    Compute market returns per asset.

    [CHANGED] Sau khi reindex về business days, pct_change(h) giờ đo đúng h trading days.
    Không còn TODO về non-trading days — đã được fix ở _reindex_to_business_days.

    :param market: DataFrame đã reindex về business days
    :param horizons: Return horizons tính theo trading days
    :return: DataFrame với thêm cột return_1d, return_5d
    """
    for h in horizons:
        market[f"return_{h}d"] = (
            market
            .groupby("asset_id")["market_price"]
            .pct_change(h)
        )
    return market


def _add_market_volatilities(market: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    """
    Compute market volatility as rolling std of daily returns.

    :param market: DataFrame với cột return_1d
    :param windows: Rolling window sizes tính theo trading days
    :return: DataFrame với thêm cột volatility_5d, volatility_10d
    """
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
    [NEW] Compute moving averages and price-vs-MA ratio.

    Lý do thêm:
        MA là context quan trọng cho FOMO — investor dễ FOMO hơn khi giá
        đang trending lên (price > MA).

    Columns được thêm:
        ma_5d, ma_20d       — short và medium term moving average
        price_above_ma20    — market_price / ma_20d
                              > 1.0: giá đang cao hơn MA20 (uptrend context)
                              < 1.0: giá đang thấp hơn MA20 (downtrend context)
                              Dùng trong feature_builder: avg_price_above_ma20_at_buy

    :param market: DataFrame với cột market_price
    :param windows: MA window sizes tính theo trading days
    :return: DataFrame với thêm cột ma_5d, ma_20d, price_above_ma20
    """
    for w in windows:
        market[f"ma_{w}d"] = (
            market
            .groupby("asset_id")["market_price"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )

    # Price vs MA20 ratio — đo mức độ price đang cao hơn average bao nhiêu
    market["price_above_ma20"] = market["market_price"] / market["ma_20d"]
    return market


def _add_rsi(market: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    [NEW] Compute RSI (Relative Strength Index).

    Lý do thêm:
        RSI > 70: asset đang overbought — đây là context FOMO mạnh nhất.
        Investor mua vào khi RSI cao = chasing overbought asset = FOMO signal rõ.
        Dùng trong feature_builder: avg_rsi_at_buy per window.

    Công thức RSI chuẩn (Wilder, 1978):
        delta   = price change so với ngày hôm trước
        gain    = trung bình các ngày tăng trong `period` ngày
        loss    = trung bình các ngày giảm trong `period` ngày (absolute value)
        RS      = gain / loss
        RSI     = 100 - (100 / (1 + RS))

    :param market: DataFrame với cột return_1d
    :param period: RSI period, default 14 (Wilder standard)
    :return: DataFrame với thêm cột rsi_14
    """
    def _compute_rsi(prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)

        # Wilder smoothing (exponential moving average với alpha = 1/period)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        rs  = avg_gain / avg_loss.replace(0, np.nan)  # avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi

    market[f"rsi_{period}"] = (
        market
        .groupby("asset_id")["market_price"]
        .transform(lambda x: _compute_rsi(x, period))
    )
    return market


def build_market(market: pd.DataFrame) -> pd.DataFrame:
    """
    Build enriched daily market table from close_prices.

    [CHANGED] Pipeline bây giờ:
        1. Reindex về business days + forward fill  ← [NEW] fix weekend gap
        2. Tính returns (giờ đúng trading days)
        3. Tính volatility
        4. Tính MA + price_above_ma20               ← [NEW]
        5. Tính RSI                                 ← [NEW]

    Output columns:
        asset_id | timestamp | market_price
        | return_1d | return_5d
        | volatility_5d | volatility_10d
        | ma_5d | ma_20d | price_above_ma20
        | rsi_14

    :param market: DataFrame với columns: asset_id, timestamp, market_price
    :return: Enriched market DataFrame
    """
    required_cols = {"asset_id", "timestamp", "market_price"}
    missing = required_cols - set(market.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    market = market.copy()
    market = market.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    # [NEW] Fix weekend/holiday gap trước khi tính bất kỳ thứ gì
    print("[build_market] Reindexing to business days...")
    market = _reindex_to_business_days(market)

    market = _add_market_returns(market)
    market = _add_market_volatilities(market)

    # [NEW] Technical indicators
    market = _add_moving_averages(market)
    market = _add_rsi(market)

    print(f"[build_market] Done. Shape: {market.shape}, "
          f"Assets: {market['asset_id'].nunique()}, "
          f"Date range: {market['timestamp'].min().date()} → {market['timestamp'].max().date()}")
    return market


def enrich_trades_with_market(trades: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich trades DataFrame by merging with market data on (asset_id, timestamp).

    Sau merge: mỗi trade có thêm market context tại ngày giao dịch:
    return_1d, return_5d, volatility, MA, RSI.

    Dùng how="left" để giữ tất cả trades kể cả những ngày không có market data.
    Trades không match sẽ có NaN ở market columns — feature_builder sẽ handle.

    :param trades: Enriched trades DataFrame từ data_loader
    :param market: Enriched market DataFrame từ build_market
    :return: Trades với thêm market context columns
    """
    enriched = trades.merge(
        market,
        on=["asset_id", "timestamp"],
        how="left"
    )

    # Warn nếu có nhiều trades không match được market data
    no_match = enriched["market_price"].isna().sum()
    if no_match > 0:
        pct = no_match / len(enriched) * 100
        print(f"[WARNING] {no_match:,} trades ({pct:.1f}%) không match được market data")

    return enriched