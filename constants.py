class LabelingThresholds:
    """
    A class to hold threshold values for labeling data points in a trading model.
    """
    TH_AGV_RETURN_BEFORE_BUY = 0.02 # Threshold for avg_return_before_buy: 2%
    TH_BUY_AFTER_SPIKE_RATIO = 0.3  # Threshold for buy_after_spike_ratio: 30%
    TH_AVG_MISSED_RETURN = 0.01     # Threshold for avg_missed_return: 1%
    TH_MIN_BUYS = 1                 # Minimum number of buys required

class FOMOScoreThresholds:
    """
    A class to hold threshold values for FOMO score classification.
    """
    LOW_FOMO = 0.005    # Low FOMO threshold: 0.5%
    MEDIUM_FOMO = 0.03  # Medium FOMO threshold: 3%


INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
MODEL_DIR = "data/models"

TRANSACTIONS_FILE = f"{INPUT_DIR}/transactions.csv"
CUSTOMERS_FILE = f"{INPUT_DIR}/customer_information.csv"
CLOSE_PRICES_FILE = f"{INPUT_DIR}/close_prices.csv"

FOMO_FEATURE_FILE = f"{OUTPUT_DIR}/fomo_feature_data.csv"
FOMO_FEATURE_LABEL_FILE = f"{OUTPUT_DIR}/fomo_feature_label_data.csv"
FOMO_TRAIN_FILE = f"{OUTPUT_DIR}/fomo_train_data.csv"
FOMO_TEST_FILE = f"{OUTPUT_DIR}/fomo_test_data.csv"

FEATURE_COLS = [
    "avg_return_before_buy",
    "buy_after_spike_ratio",
    "avg_missed_return",
    "n_buys"
]
