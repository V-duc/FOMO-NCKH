import pandas as pd

# ── Risk level encoding ────────────────────────────────────────────────────
# Ordinal scale: Predicted_* dùng giá trị 0.5 bước thấp hơn để phân biệt
# self-reported vs predicted mà không cần cột flag riêng.
# Not_Available → NaN, XGBoost tự handle.
_RISK_LEVEL_MAP = {
    "Conservative":           1.0,
    "Predicted_Conservative": 1.5,
    "Income":                 2.0,
    "Predicted_Income":       2.5,
    "Balanced":               3.0,
    "Predicted_Balanced":     3.5,
    "Aggressive":             4.0,
    "Predicted_Aggressive":   4.5,
    "Not_Available":          None,
}

# ── Investment capacity encoding ───────────────────────────────────────────
# Hai cột:
#   investment_capacity_ordinal (1-4): dùng làm feature trực tiếp cho model
#   investment_capacity_value (midpoint €): dùng để tính position_size_ratio
# Predicted_* dùng cùng giá trị với actual vì đây là số tiền thật (estimate
# của ngân hàng), không phải psychological label như riskLevel.
_CAPACITY_ORDINAL_MAP = {
    "CAP_LT30K":                1,
    "Predicted_CAP_LT30K":      1,
    "CAP_30K_80K":              2,
    "Predicted_CAP_30K_80K":    2,
    "CAP_80K_300K":             3,
    "Predicted_CAP_80K_300K":   3,
    "CAP_GT300K":               4,
    "Predicted_CAP_GT300K":     4,
    "Not_Available":            None,
}

_CAPACITY_VALUE_MAP = {
    "CAP_LT30K":                15_000,
    "Predicted_CAP_LT30K":      15_000,
    "CAP_30K_80K":              55_000,
    "Predicted_CAP_30K_80K":    55_000,
    "CAP_80K_300K":             190_000,
    "Predicted_CAP_80K_300K":   190_000,
    "CAP_GT300K":               400_000,
    "Predicted_CAP_GT300K":     400_000,
    "Not_Available":            None,
}

# ── Time period ────────────────────────────────────────────────────────────
# EDA findings:
#   - Jan 2018: artifact spike (39k tx), bỏ
#   - 2019: quá thưa (~1,200 tx/tháng), bỏ
#   - Mar-Jun 2020: COVID crash + recovery → false positive trong label, bỏ
#   - Jul 2020 trở đi: ổn định, data quality tốt
PERIOD_START = "2020-07-01"
PERIOD_END   = "2022-11-30"


def _load_transactions(file: str) -> pd.DataFrame:
    """
    Load and preprocess transactions data from CSV file.

    Changes from original:
    - [NEW] Filter time period: Jul 2020 – Nov 2022 (bỏ COVID period và artifact)
    - [NEW] Data quality guard: bỏ rows có totalValue <= 0 hoặc units <= 0
    - [NEW] Giữ lại cột 'channel' để tính internet_banking_ratio trong feature_builder
    - [NEW] Giữ lại cột 'totalValue' để tính position_size_ratio sau khi join customer

    :param file: Path to transactions CSV
    :return: Preprocessed trades DataFrame
    """
    transactions = pd.read_excel(file)
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])

    # [NEW] Filter time period — bỏ COVID period và data artifact đầu dataset
    before = len(transactions)
    transactions = transactions[
        (transactions["timestamp"] >= PERIOD_START) &
        (transactions["timestamp"] <= PERIOD_END)
    ]
    print(f"[filter period] {before:,} → {len(transactions):,} rows "
          f"({before - len(transactions):,} dropped)")

    # [NEW] Data quality guard — dù EDA cho thấy không có, vẫn nên có
    bad_rows = (transactions["totalValue"] <= 0) | (transactions["units"] <= 0)
    if bad_rows.sum() > 0:
        print(f"[WARNING] Dropping {bad_rows.sum()} rows với totalValue/units <= 0")
        transactions = transactions[~bad_rows]

    # Compute execution price
    transactions["price"] = transactions["totalValue"] / transactions["units"]

    # Normalize side
    transactions["side"] = transactions["transactionType"].str.upper()

    # Validate sides
    invalid_sides = transactions[~transactions["side"].isin(["BUY", "SELL"])]
    if not invalid_sides.empty:
        invalid_values = invalid_sides["side"].unique().tolist()
        raise ValueError(f"Invalid transaction types found: {invalid_values}. "
                         f"Only 'BUY' and 'SELL' are allowed.")

    # Build canonical trades table
    # [CHANGED] Thêm 'channel' và 'totalValue' so với original
    trades = transactions[[
        "customerID",
        "ISIN",
        "timestamp",
        "side",
        "price",
        "units",
        "channel",      # [NEW] để tính internet_banking_ratio
        "totalValue",   # [NEW] để tính position_size_ratio = totalValue / investment_capacity
    ]].rename(columns={
        "customerID": "investor_id",
        "ISIN":       "asset_id",
        "units":      "quantity",
    })
    return trades


def _load_customers(file: str) -> pd.DataFrame:
    """
    Load and preprocess customer information.

    Changes from original:
    - [NEW] Dedup: lấy record mới nhất per customer
      Justified by EDA: 0% customer thay đổi riskLevel hoặc investmentCapacity
      → lấy record mới nhất là đủ, không cần merge_asof
    - [NEW] Encode riskLevel thành ordinal float (xem _RISK_LEVEL_MAP)
    - [NEW] Encode investmentCapacity thành ordinal int và midpoint value
    - [NEW] Return 3 separate sets: train, validation (Professional), drop list

    :param file: Path to customer_information CSV
    :return: DataFrame với customerID + encoded features, chỉ giữ valid customers
    """
    customers = pd.read_excel(file)
    customers["timestamp"] = pd.to_datetime(customers["timestamp"])

    # [NEW] Dedup — lấy record mới nhất per customer
    # EDA: 0 customer thay đổi riskLevel/capacity → lấy latest là safe
    customers = (
        customers
        .sort_values("timestamp")
        .groupby("customerID")
        .last()
        .reset_index()
    )

    # [NEW] Encode riskLevel → ordinal float
    customers["risk_level"] = customers["riskLevel"].map(_RISK_LEVEL_MAP)

    # [NEW] Encode investmentCapacity → ordinal (1-4) và midpoint value (€)
    customers["investment_capacity_ordinal"] = customers["investmentCapacity"].map(_CAPACITY_ORDINAL_MAP)
    customers["investment_capacity_value"]   = customers["investmentCapacity"].map(_CAPACITY_VALUE_MAP)

    return customers[[
        "customerID",
        "customerType",
        "risk_level",
        "investment_capacity_ordinal",
        "investment_capacity_value",
    ]]


def _filter_by_asset_type(trades: pd.DataFrame, assets_file: str) -> pd.DataFrame:
    """
    [NEW] Filter trades to keep only Stock transactions.

    EDA findings:
    - Stock spike frequency: 13.3% vs Bond: 0.8% (17x difference)
    - buy_after_spike_ratio gần như luôn = 0 với Bond/MTF → feature vô nghĩa
    - Stock chiếm 89.1% transactions → filter an toàn, không mất nhiều data

    :param trades: Trades DataFrame
    :param assets_file: Path to asset_information CSV
    :return: Filtered trades với chỉ Stock assets
    """
    assets = pd.read_csv(assets_file)

    # Dedup assets — lấy record mới nhất per ISIN
    assets = (
        assets
        .sort_values("timestamp")
        .groupby("ISIN")
        .last()
        .reset_index()
    )

    stock_isins = assets[assets["assetCategory"] == "Stock"]["ISIN"].unique()

    before = len(trades)
    trades = trades[trades["asset_id"].isin(stock_isins)]
    print(f"[filter stock] {before:,} → {len(trades):,} rows "
          f"({before - len(trades):,} non-Stock dropped)")
    return trades


def _attach_customer_features(trades: pd.DataFrame, customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    [NEW] Filter trades by customer type and attach customer features.

    Customer type decision (EDA):
    - Train set:      Mass + Premium  → retail investor điển hình
    - Validation set: Professional    → sanity check: nếu FOMO cao → heuristics sai
    - Drop:           Legal Entity, Inactive → tổ chức / không xác định

    :return: (train_trades, validation_trades)
    """
    # Split customer sets
    train_customers = customers[customers["customerType"].isin(["Mass", "Premium"])]
    val_customers   = customers[customers["customerType"] == "Professional"]

    # Customer features để join
    customer_features = customers[[
        "customerID",
        "risk_level",
        "investment_capacity_ordinal",
        "investment_capacity_value",
    ]]

    def _join_and_compute(trades_subset: pd.DataFrame, cust_subset: pd.DataFrame) -> pd.DataFrame:
        # Filter trades theo customer set
        result = trades_subset[trades_subset["investor_id"].isin(cust_subset["customerID"])]

        # Join customer features
        result = result.merge(
            customer_features,
            left_on="investor_id",
            right_on="customerID",
            how="left"
        ).drop(columns=["customerID"])

        # [NEW] Tính position_size_ratio = totalValue / investment_capacity_value
        # Đo mức độ "bet lớn" của investor — FOMO investor thường bet lớn hơn capacity
        # NaN nếu investment_capacity_value = None (Not_Available)
        result["position_size_ratio"] = (
            result["totalValue"] / result["investment_capacity_value"]
        )

        # Bỏ investment_capacity_value sau khi tính ratio — không cần đưa vào model
        result = result.drop(columns=["investment_capacity_value"])

        return result

    train_trades = _join_and_compute(trades, train_customers)
    val_trades   = _join_and_compute(trades, val_customers)

    print(f"[customer split] train: {len(train_trades):,} rows "
          f"({train_trades['investor_id'].nunique():,} investors)")
    print(f"[customer split] validation (Professional): {len(val_trades):,} rows "
          f"({val_trades['investor_id'].nunique():,} investors)")

    return train_trades, val_trades


def load_trade_data(
    transactions_file: str,
    customers_file: str,
    assets_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load, filter, and enrich trade data.

    [CHANGED] Signature thay đổi:
    - Thêm assets_file parameter (để filter Stock)
    - Trả về tuple (train_trades, validation_trades) thay vì 1 DataFrame

    Output columns:
        investor_id | asset_id | timestamp | side | price | quantity
        | channel | totalValue | risk_level | investment_capacity_ordinal
        | position_size_ratio

    :return: (train_trades, validation_trades)
    """
    trades    = _load_transactions(transactions_file)
    trades    = _filter_by_asset_type(trades, assets_file)
    customers = _load_customers(customers_file)
    train_trades, val_trades = _attach_customer_features(trades, customers)
    return train_trades, val_trades


def load_close_prices(file: str, assets_file: str = None) -> pd.DataFrame:
    """
    Load and preprocess close prices data from CSV file.

    [NEW] Optional assets_file parameter: nếu pass vào thì filter chỉ giữ
    Stock assets và assets trong time period — giảm memory và tăng tốc độ
    tính return trong data_builder.

    :param file: Path to close_prices CSV
    :param assets_file: Optional path to asset_information CSV
    :return: DataFrame với columns: asset_id | timestamp | market_price
    """
    close_prices = pd.read_csv(file, parse_dates=["timestamp"])
    close_prices = close_prices.rename(columns={
        "ISIN":       "asset_id",
        "closePrice": "market_price",
    })

    # [NEW] Filter time period cho close prices
    # Lấy rộng hơn một chút (từ 2020-01) để tính return_5d đầu period không bị NaN
    close_prices = close_prices[
        (close_prices["timestamp"] >= "2020-01-01") &
        (close_prices["timestamp"] <= PERIOD_END)
    ]

    # [NEW] Filter chỉ Stock assets nếu assets_file được cung cấp
    if assets_file is not None:
        assets = pd.read_csv(assets_file)
        assets = assets.sort_values("timestamp").groupby("ISIN").last().reset_index()
        stock_isins = assets[assets["assetCategory"] == "Stock"]["ISIN"].unique()
        before = len(close_prices)
        close_prices = close_prices[close_prices["asset_id"].isin(stock_isins)]
        print(f"[close prices] filtered to Stock only: {before:,} → {len(close_prices):,} rows")

    close_prices = close_prices[[
        "asset_id",
        "timestamp",
        "market_price",
    ]]
    return close_prices.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)