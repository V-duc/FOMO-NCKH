import pandas as pd

# investor_id | asset_id | timestamp | side | price | quantity
def build_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare traces data by converting timestamps and sorting.

    :param trades: DataFrame containing trade data
    :return: Prepared DataFrame with sorted (investor, timestamps)
    """
    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades = trades.sort_values(["investor_id", "timestamp"]).reset_index(drop=True)
    return trades


# TODO: handle non-trading days (weekends, holidays) properly in return calculations
# instead of simple pct_change which assumes continuous days
def _add_market_returns(market: pd.DataFrame, horizons=(1, 5)):
    """
    Compute market returns

    :param market: DataFrame containing market data
    :param horizons: Iterable of time horizons in days for which to compute returns
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
    Compute market volatility as rolling std of daily returns

    :param market: DataFrame containing market data
    :param windows: Rolling window sizes in days
    :return: DataFrame with added volatility columns
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


# asset_id | timestamp | market_price | return_1d | return_5d | volatility_5d | volatility_10d
def build_market(market: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily market price table from close_prices

    :param market: DataFrame containing close price data
    :return: DataFrame with columns: asset_id, timestamp, market_price
    """
    required_cols = {"asset_id", "timestamp", "market_price"}
    missing = required_cols - set(market.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    market = market.copy()
    market = market.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    market = _add_market_returns(market)
    market = _add_market_volatilities(market)
    return market


# investor_id | asset_id | timestamp | side | price | quantity |
# market_price | return_1d | return_5d | volatility_5d | volatility_10d
def enrich_trades_with_market(trades: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich trades DataFrame by merging with market data on asset_id and timestamp
    to build trade-level signals.

    :param trades: DataFrame containing trade data
    :param market: DataFrame containing market data
    :return: Enriched trades DataFrame
    """
    enriched = trades.merge(
        market,
        on=["asset_id", "timestamp"],
        how="left"
    )
    return enriched
