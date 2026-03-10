import argparse
import model_builder
import numpy as np
import pandas as pd
from datetime import datetime
import utils
from constants import (
    FOMO_TRAIN_FILE,
    FOMO_TEST_FILE,
    FEATURE_COLS,
    MODEL_DIR
)


def load_train_data() -> tuple[pd.DataFrame, pd.Series]:
    train_data = pd.read_csv(FOMO_TRAIN_FILE)

    # Extract features and labels for modeling
    X_train = train_data[FEATURE_COLS]
    y_train = train_data["fomo_label"]
    return X_train, y_train


def load_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    test_data = pd.read_csv(FOMO_TEST_FILE)

    # Extract features and labels for model testing
    X_test = test_data[FEATURE_COLS]
    y_test = test_data["fomo_label"]
    return test_data, X_test, y_test


def print_score_distribution(dataset: pd.DataFrame) -> None:
    scores = dataset["fomo_score"].values
    print(f"\nFOMO Score Distribution:")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std: {scores.std():.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    print(f"  25th percentile: {np.percentile(scores, 25):.4f}")
    print(f"  75th percentile: {np.percentile(scores, 75):.4f}")
    print(f"  95th percentile: {np.percentile(scores, 95):.4f}")
    print(f"  Unique values: {len(np.unique(scores))}")

    levels = dataset["fomo_level"].tolist()
    if levels is not None:
        print(f"\nFOMO Level Counts:")
        level_counts = pd.Series(levels).value_counts().sort_index()
        for level, count in level_counts.items():
            percentage = (count / len(levels)) * 100
            print(f"  {level}: {count} ({percentage:.1f}%)")


def make_results(classifier, test_data) -> pd.DataFrame:
    """
    Generate FOMO detection results for test data.
    
    Detects FOMO behavior in historical trading windows and reports the behavioral
    features that were observed during those periods.
    
    :param classifier: Trained FOMO detection classifier
    :param test_data: Full test DataFrame (includes investor_id, window_start, and all columns)
    :return: DataFrame with detection results including investor ID, window, FOMO score, 
             behavioral features, and key signals
    """
    def top_signals_per_row(row_idx):
        row_shap = shap_values[row_idx]
        row_features = X_test.iloc[row_idx]

        # Sort by absolute SHAP value to find most important signals
        top_idx = np.argsort(np.abs(row_shap))[::-1]
        top_features = row_features.index[top_idx][:2].tolist()
        return top_features

    X_test = test_data[FEATURE_COLS]
    shap_values = model_builder.get_shap_values(classifier, X_test)
    fomo_probabilities = classifier.predict_proba(X_test)[:, 1]
    fomo_levels = [utils.fomo_level(s) for s in fomo_probabilities]

    # Model certainty: how confident the model is in its classification
    # High certainty (near 1.0) = model is very sure about the classification (either FOMO or No FOMO)
    # Low certainty (near 0.0) = model is uncertain, score is close to decision boundary (0.5)
    # Note: High certainty can mean "definitely FOMO" OR "definitely NOT FOMO"
    model_certainty = np.abs(fomo_probabilities - 0.5) * 2

    # Build output with detection results
    output_dict = {
        "investor_id": test_data["investor_id"].values,
        "window_start": test_data["window_start"].values if "window_start" in test_data.columns else None,
        "fomo_score": fomo_probabilities.round(5),
        "fomo_level": fomo_levels,
        "model_certainty": model_certainty.round(3),
        "key_behavioral_signals": [top_signals_per_row(i) for i in range(len(X_test))],
    }

    # Add all behavioral features to the output for transparency
    for col in FEATURE_COLS:
        output_dict[col] = test_data[col].values

    output = pd.DataFrame(output_dict)
    return output


def main():
    parser = argparse.ArgumentParser(description='FOMO Detection Model Training and Evaluation')
    parser.add_argument(
        '--load-model',
        type=str,
        default=None,
        help='Path to saved model file to load (e.g., models/xgbclassifier_20260201_143022.json)'
    )
    args = parser.parse_args()

    if args.load_model:
        print(f'Loading model from: {args.load_model}...')
        classifier = model_builder.load_xgboost_model(args.load_model)
        print('Model loaded successfully')
    else:
        print('Loading train dataset...')
        X_train, y_train = load_train_data()

        print('\nBuilding classifier...')
        # classifier = model_builder.build_random_forest_classifier({
        classifier = model_builder.build_xgboost_classifier({
            "X_train": X_train,
            "y_train": y_train
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = classifier.__class__.__name__.lower()
        model_filename = f"{MODEL_DIR}/{model_type}_{timestamp}.json"
        model_builder.store_xgboost_model(classifier, model_filename)
        print(f"Model saved to: {model_filename}")

    print('\nLoading test dataset...')
    test_data, X_test, y_test = load_test_data()

    print('\nClassifier evaluation:')
    model_builder.evaluate_model(classifier, X_test, y_test)

    print('\nGenerating results...')
    results = make_results(classifier, test_data)
    print_score_distribution(results)

    # Print sample results
    print('\nSample result entries:')
    print(results.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
