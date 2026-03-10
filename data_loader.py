import pandas as pd

# investor_id | asset_id | timestamp | side | price | quantity
def _load_transactions(file: str) -> pd.DataFrame:
    """
    Load and preprocess transactions data from CSV file.

    :return: Preprocessed DataFrame containing transactions data
    """
    # Load transactions
    transactions = pd.read_csv(file, parse_dates=["timestamp"])

    # Compute execution price
    transactions["price"] = (transactions["totalValue"] / transactions["units"])

    # Normalize side
    transactions["side"] = transactions["transactionType"].str.upper()

    # Validate that all sides are either BUY or SELL
    invalid_sides = transactions[~transactions["side"].isin(["BUY", "SELL"])]
    if not invalid_sides.empty:
        invalid_values = invalid_sides["side"].unique().tolist()
        raise ValueError(f"Invalid transaction types found: {invalid_values}. "
        f"Only 'BUY' and 'SELL' are allowed.")

    # Build canonical trades table
    trades = transactions[[
        "customerID",
        "ISIN",
        "timestamp",
        "side",
        "price",
        "units"
    ]].rename(columns={
        "customerID": "investor_id",
        "ISIN": "asset_id",
        "units": "quantity"
    })
    return trades


def _filter_legal_entities(file: str, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Load and preprocess legal entities data from CSV file.

    :return: DataFrame containing legal entities
    """
    customers = pd.read_csv(file, parse_dates=["timestamp"])

    # Keep only individual investors
    valid_customers = customers[
        customers["customerType"].isin(["Mass", "Premium"])
    ]["customerID"].unique()

    trades = trades[trades["investor_id"].isin(valid_customers)]
    return trades


def load_trade_data(transactions_file: str, customers_file: str) -> pd.DataFrame:
    trades = _load_transactions(transactions_file)
    trades = _filter_legal_entities(customers_file, trades)
    return trades


# asset_id | timestamp | market_price
def load_close_prices(file: str) -> pd.DataFrame:
    """
    Load and preprocess close prices data from CSV file.
    """
    close_prices = pd.read_csv(file)
    close_prices = close_prices.rename(columns={
        "ISIN": "asset_id",
        "closePrice": "market_price"
    })

    close_prices["timestamp"] = pd.to_datetime(close_prices["timestamp"])
    close_prices = close_prices[[
        "asset_id",
        "timestamp",
        "market_price"
    ]]
    return close_prices.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
