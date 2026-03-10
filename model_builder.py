import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def build_xgboost_classifier(input: dict[str, any]):
    # Calculate scale_pos_weight for imbalanced data
    class_counts = input["y_train"].value_counts()
    if len(class_counts) == 2:
        scale_pos_weight = class_counts[0] / class_counts[1]
    else:
        scale_pos_weight = 1.0

    model = xgb.XGBClassifier(
        random_state=42,
        max_depth=4,
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0   # L2 regularization
    )
    model.fit(input["X_train"], input["y_train"])
    return model


def store_xgboost_model(classifier, filename):
    classifier.save_model(filename)


def load_xgboost_model(filename):
    model = xgb.XGBClassifier()
    model.load_model(filename)
    return model



def build_random_forest_classifier(input):
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True
    )
    model.fit(input["X_train"], input["y_train"])
    return model


def store_random_forest_model(classifier, filename):
    import joblib
    joblib.dump(classifier, filename)



def evaluate_model(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:,1]
    print("- Accuracy:", accuracy_score(y_test, y_pred))
    print("- ROC-AUC:", roc_auc_score(y_test, y_prob))
    return (y_pred, y_prob)


def get_shap_values(classifier, X_test):
    """
    Calculate SHAP values for model interpretability.

    For binary classification, returns SHAP values for the positive class (class 1).

    :param classifier: Trained tree-based classifier (XGBoost, RandomForest, etc.)
    :param X_test: Test features
    :return: SHAP values array of shape (n_samples, n_features)
    """
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test)
    
    # Handle binary classification: shap_values might be a list of 2 arrays
    # We want the SHAP values for the positive class (index 1)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        return shap_values[1]
    
    return shap_values


def plot_shap_summary(classifier, X_test):
    # Visualize top features
    shap_values = get_shap_values(classifier, X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
