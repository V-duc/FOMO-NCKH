import pandas as pd
import numpy as np
import data_builder


def get_buys(trades_i: pd.DataFrame) -> pd.DataFrame:
    """
    Filter trades to get only buy transactions.

    :param trades_i: DataFrame of trades (must have 'side' column)
    :return: DataFrame containing only BUY transactions
    """
    return trades_i[trades_i.side == "BUY"]


def avg_return_before_buy(buys: pd.DataFrame, horizon: str = "return_5d") -> float:
    """
    Calculate average market return before buy decisions (price chasing indicator).

    Measures whether investor tends to buy after prices have already risen,
    which is a classic FOMO behavior pattern.

    Units: Returns are stored as decimal fractions produced by
    pandas `pct_change` (e.g., 0.02 means 2%). The returned
    average is in the same decimal unit; multiply by 100 if you
    need percentage points for display.

    :param buys: DataFrame of buy transactions (enriched with market return columns)
    :param horizon: Column name for return horizon (e.g., 'return_5d')
    :return: Average return before buys, or NaN if no buys
    """
    if len(buys) == 0:
        return np.nan
    return buys[horizon].mean()


def buy_after_spike_ratio(buys: pd.DataFrame, horizon: str = "return_5d", threshold: float = 0.02) -> float:
    """
    Calculate ratio of buys that occurred after significant market spikes (momentum chasing).

    Measures whether investor tends to buy after strong positive returns,
    another FOMO behavior indicator.

    :param buys: DataFrame of buy transactions (enriched with market return columns)
    :param horizon: Column name for return horizon (e.g., 'return_5d')
    :param threshold: Minimum return to be considered a "spike" (default: 2%)
    :return: Ratio of buys after spikes (0.0 to 1.0), or 0.0 if no buys
    """
    if len(buys) == 0:
        return 0.0
    return (buys[horizon] > threshold).mean()


def avg_missed_return(trades_i: pd.DataFrame, market_i: pd.DataFrame, horizon: str = "return_5d") -> float:
    """
    Calculate average return on days when investor didn't trade (regret/FOMO indicator).

    Measures the returns investor "missed out" on by not being active on certain days.
    High missed returns may trigger FOMO in subsequent periods.

    :param trades_i: DataFrame of investor's trades
    :param market_i: DataFrame of market data for relevant assets
    :param horizon: Column name for return horizon (e.g., 'return_5d')
    :return: Average return on non-trading days, or NaN if no missed days
    """
    active_times = set(trades_i.timestamp)
    missed = market_i[~market_i.timestamp.isin(active_times)]
    return missed[horizon].mean()


def return_after_buy(buys: pd.DataFrame, market: pd.DataFrame, delta: int = 5) -> pd.Series:
    """
    Calculate actual return after buy by comparing buy price to future market price.

    This measures the outcome of a buy decision: did the investor make or lose money
    after holding for `delta` days? Used to validate whether FOMO behavior leads
    to poor investment outcomes.

    :param buys: DataFrame of buy transactions (must have: asset_id, timestamp, price)
    :param market: DataFrame of daily market prices (must have: asset_id, timestamp, market_price)
    :param delta: Number of days to hold before calculating return (default: 5)
    :return: Series of returns, NaN if future price unavailable

    Example:
        Buy AAPL on 2024-01-01 at $100
        Market price on 2024-01-06 is $105
        Return = (105 - 100) / 100 = 5%
    """
    if len(buys) == 0:
        return pd.Series(dtype=float)

    # Prepare future market prices by shifting timestamps back by delta days
    # This allows us to match buy_date with (buy_date + delta) market price
    future_market = market[["asset_id", "timestamp", "market_price"]].copy()
    future_market["timestamp"] = future_market["timestamp"] - pd.Timedelta(days=delta)

    # Merge: each buy gets the market price from `delta` days in the future
    merged = buys.merge(
        future_market,
        on=["asset_id", "timestamp"],
        how="left",
        suffixes=("", "_future")
    )

    # Calculate return: (future_price - buy_price) / buy_price
    return (merged["market_price_future"] - merged["price"]) / merged["price"]


def loss_after_buy_ratio(return_after: pd.Series) -> float:
    """
    Calculate ratio of buy transactions that resulted in losses.

    :param return_after: Series of returns after buy transactions
    :return: Ratio of negative returns (0.0 to 1.0), or NaN if no returns
    """
    if len(return_after) == 0:
        return np.nan
    return (return_after < 0).mean()


def build_fomo_features_one_window(trades_i: pd.DataFrame, market_i: pd.DataFrame) -> dict:
    """
    Build FOMO-related features for a single time window of an investor's trades.

    Extracts behavioral signals including:
    - Trading intensity (n_trades, n_buys)
    - Price/momentum chasing indicators
    - Regret chasing (missed opportunities)
    - Outcome validation (post-buy returns)

    :param trades_i: DataFrame of investor's trades in the window (enriched with market data)
    :param market_i: DataFrame of market data for assets traded in the window
    :return: Dictionary of feature values
    """
    buys = get_buys(trades_i)
    r_after = return_after_buy(buys, market_i)

    return {
        "n_trades": len(trades_i),

        # core fomo: overreaction intensity
        "n_buys": len(buys),

        # core fomo: price chasing
        "avg_return_before_buy": avg_return_before_buy(buys),

        # core fomo: momentum chasing
        "buy_after_spike_ratio": buy_after_spike_ratio(buys),

        # core fomo: regret chasing
        "avg_missed_return": avg_missed_return(trades_i, market_i),

        # outcome: for validation
        "avg_post_buy_return":
            r_after.mean() if len(r_after) else np.nan,

        # outcome: for validation
        "loss_after_buy_ratio":
            loss_after_buy_ratio(r_after)
    }


def _investor_time_windows(trades: pd.DataFrame, window: str = "5D"):
    """
    Create time-based groupings of trades by investor.

    :param trades: DataFrame of all trades
    :param window: Time window size (pandas frequency string, e.g., "5D", "1W")
    :return: GroupBy object grouped by (investor_id, time_window)
    """
    return trades.groupby([
        "investor_id",
        pd.Grouper(key="timestamp", freq=window)
    ])


def build_feature_table(trades: pd.DataFrame, market: pd.DataFrame, window: str = "5D") -> pd.DataFrame:
    """
    Aggregate FOMO features for all investors over specified time windows.

    Creates a feature table where each row represents one investor's behavior
    in a specific time window. This table is used for training/scoring FOMO models.

    :param trades: DataFrame containing enriched trade data (includes market returns)
    :param market: DataFrame containing full market data (for missed returns and future prices)
    :param window: Time window for aggregation (e.g., "5D" for 5 days, "1W" for 1 week)
    :return: DataFrame with columns: investor_id, window_start, n_trades, n_buys,
             avg_return_before_buy, buy_after_spike_ratio, avg_missed_return,
             avg_post_buy_return, loss_after_buy_ratio
    """
    rows = []
    for (inv, t), trades_i in _investor_time_windows(trades, window):
        if len(trades_i) == 0:
            continue

        market_i = market[
            market.asset_id.isin(trades_i.asset_id.unique())
        ]

        features = build_fomo_features_one_window(trades_i, market_i)
        features.update({
            "investor_id": inv,
            "window_start": trades_i["timestamp"].min(),  # Use actual first trade, not bin boundary
            "window_bin": t  # Keep bin boundary for reference if needed
        })
        rows.append(features)
    return pd.DataFrame(rows)


def build_fomo_feature_table(trade_data: pd.DataFrame, market_data: pd.DataFrame, window: str = "5D") -> pd.DataFrame:
    """
    End-to-end pipeline to build FOMO feature table from raw data.

    Steps:
    1. Prepare trades (clean, sort)
    2. Build market data (compute returns, volatility)
    3. Enrich trades with market context
    4. Aggregate features by investor and time window

    :param trade_data: Raw trade data from data loader
    :param market_data: Raw market/close price data from data loader
    :param window: Time window for feature aggregation (default: "5D")
    :return: Feature table ready for FOMO analysis or modeling
    """
    trades = data_builder.build_trades(trade_data)
    market = data_builder.build_market(market_data)
    trades = data_builder.enrich_trades_with_market(trades, market)
    return build_feature_table(
        trades,
        market,
        window=window
    )


def _weak_label(row: pd.Series, R1: float = 0.02, R2: float = 0.3, R3: float = 0.01, min_buys: int = 1) -> int:
    """
    Apply weak labeling rule to classify FOMO behavior.

    Labels a window as FOMO (1) if all conditions are met:
    - Average return before buy > R1 (price chasing)
    - Buy after spike ratio > R2 (momentum chasing)
    - Average missed return > R3 (regret/FOMO trigger)
    - Minimum number of buys >= min_buys

    :param row: Feature row from feature table
    :param R1: Threshold for avg_return_before_buy (default: 2%)
    :param R2: Threshold for buy_after_spike_ratio (default: 30%)
    :param R3: Threshold for avg_missed_return (default: 1%)
    :param min_buys: Minimum number of buys required (default: 1)
    :return: 1 if FOMO behavior detected, 0 otherwise
    """
    label = 0
    if (row["avg_return_before_buy"] > R1 and
        row["buy_after_spike_ratio"] > R2 and
        row["avg_missed_return"] > R3 and
        row["n_buys"] >= min_buys):
        label = 1
    return label


def label_feature_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Add weak FOMO labels to feature table for training.

    Applies rule-based labeling to identify potential FOMO behavior patterns.
    These labels can be used for supervised learning or validation.

    :param feature_table: DataFrame from build_feature_table()
    :return: Same DataFrame with added 'fomo_label' column (0 or 1)
    """
    feature_table["fomo_label"] = feature_table.apply(_weak_label, axis=1)
    return feature_table
