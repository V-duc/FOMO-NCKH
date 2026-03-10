import argparse
import data_loader
import feature_builder
import pandas as pd
from sklearn.model_selection import train_test_split
from constants import (
    TRANSACTIONS_FILE,      # input
    CUSTOMERS_FILE,         # input
    CLOSE_PRICES_FILE,      # input
    FOMO_FEATURE_FILE,      # output
    FOMO_FEATURE_LABEL_FILE,# output
    FOMO_TRAIN_FILE,        # output
    FOMO_TEST_FILE,         # output
    FEATURE_COLS            # features
)


def build_fomo_feature_dataset():
    trade_data = data_loader.load_trade_data(TRANSACTIONS_FILE, CUSTOMERS_FILE)
    market_data = data_loader.load_close_prices(CLOSE_PRICES_FILE)
    # trade_data = trade_data.head(100000)  # for quick testing

    feature_table = feature_builder.build_fomo_feature_table(trade_data, market_data)
    feature_table_labeled = feature_builder.label_feature_table(feature_table)

    feature_table.to_csv(FOMO_FEATURE_FILE, index=False)
    print(f"Featured dataset without labels saved to: {FOMO_FEATURE_FILE}")

    feature_table_labeled.to_csv(FOMO_FEATURE_LABEL_FILE, index=False)
    print(f"Featured dataset with labels saved to: {FOMO_FEATURE_LABEL_FILE}")


def split_feature_dataset(feature_table):
    train_data, test_data = train_test_split(
        feature_table, 
        test_size=0.2, 
        shuffle=False
    )

    train_data.to_csv(FOMO_TRAIN_FILE, index=False)
    print(f"Train data saved to: {FOMO_TRAIN_FILE}")

    test_data.to_csv(FOMO_TEST_FILE, index=False)
    print(f"Test data saved to: {FOMO_TEST_FILE}")


def print_dataset_statistics():
    """Print statistics about the FOMO datasets."""
    print("\n" + "="*60)
    print("FOMO DATASET STATISTICS")
    print("="*60)
    
    # Load datasets
    try:
        train_data = pd.read_csv(FOMO_TRAIN_FILE)
        test_data = pd.read_csv(FOMO_TEST_FILE)
        full_data = pd.read_csv(FOMO_FEATURE_LABEL_FILE)
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found - {e}")
        print("Please run with --build flag first to create datasets.")
        return
    
    # Full dataset statistics
    print(f"\n{'FULL DATASET':-^60}")
    print(f"Total samples: {len(full_data)}")
    print(f"Unique investors: {full_data['investor_id'].nunique()}")
    
    print(f"\nLabel distribution:")
    label_counts = full_data['fomo_label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = count / len(full_data) * 100
        label_name = "FOMO" if label == 1 else "Non-FOMO"
        print(f"  {label_name} ({label}): {count:,} ({pct:.2f}%)")
    
    # Feature statistics
    print(f"\nFeature statistics:")
    for col in FEATURE_COLS:
        if col in full_data.columns:
            print(f"  {col}:")
            print(f"    Mean: {full_data[col].mean():.4f}")
            print(f"    Median: {full_data[col].median():.4f}")
            print(f"    Min: {full_data[col].min():.4f}")
            print(f"    Max: {full_data[col].max():.4f}")
    
    # Train/Test split statistics
    print(f"\n{'TRAIN/TEST SPLIT':-^60}")
    print(f"Train samples: {len(train_data)} ({len(train_data)/len(full_data)*100:.1f}%)")
    print(f"Test samples: {len(test_data)} ({len(test_data)/len(full_data)*100:.1f}%)")
    
    print(f"\nTrain label distribution:")
    train_labels = train_data['fomo_label'].value_counts().sort_index()
    for label, count in train_labels.items():
        pct = count / len(train_data) * 100
        label_name = "FOMO" if label == 1 else "Non-FOMO"
        print(f"  {label_name} ({label}): {count:,} ({pct:.2f}%)")
    
    print(f"\nTest label distribution:")
    test_labels = test_data['fomo_label'].value_counts().sort_index()
    for label, count in test_labels.items():
        pct = count / len(test_data) * 100
        label_name = "FOMO" if label == 1 else "Non-FOMO"
        print(f"  {label_name} ({label}): {count:,} ({pct:.2f}%)")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build or analyze FOMO datasets')
    parser.add_argument(
        '--build', 
        action='store_true', 
        help='Build datasets from raw data'
    )
    parser.add_argument(
        '--stats', 
        action='store_true', 
        help='Print statistics about existing datasets'
    )

    args = parser.parse_args()

    # If no arguments, default to build
    if not args.build and not args.stats:
        args.build = True

    if args.build:
        print('Building FOMO feature dataset...')
        build_fomo_feature_dataset()

        print('\nSplitting feature dataset into train and test sets...')
        feature_table_labeled = pd.read_csv(FOMO_FEATURE_LABEL_FILE)
        split_feature_dataset(feature_table_labeled)

        print('\n- Sample train data:')
        train_data = pd.read_csv(FOMO_TRAIN_FILE)
        print(train_data.head(20))
        print('\n- Sample test data:')
        test_data = pd.read_csv(FOMO_TEST_FILE)
        print(test_data.head(20))

    if args.stats:
        print_dataset_statistics()
