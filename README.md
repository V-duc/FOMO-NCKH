# FOMO Investor Detection Project

## Overview

This project implements a machine learning system to detect Fear of Missing Out (FOMO) behavior in cryptocurrency investors. The system analyzes trading patterns across 5-day windows to classify investors based on behavioral signals such as price chasing (buying after market spikes), missed opportunities (returns on non-trading days), and momentum chasing. Using XGBoost with SHAP explainability, the model identifies three FOMO levels: Low (<0.5%), Medium (0.5-3%), and High (>3%), providing insights into which behavioral features contribute most to FOMO-driven trading decisions.

The project includes an interactive Streamlit dashboard that allows users to explore FOMO detection results across multiple trading windows per investor. Key features include investor filtering by FOMO level, detailed behavioral analysis with SHAP values, feature comparisons against average behavior, and visual explanations through gauge charts and bar graphs. The system is designed as a detection tool to help understand historical trading patterns and identify investors exhibiting FOMO characteristics in their past trading behavior.


## Installation

1. **Install required packages:**

```bash
pip install -r requirements.txt
```

Or create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Ensure data files are in place:**
   - Raw data: `data/input/transactions.csv`, `data/input/customer_information.csv`, `data/input/close_prices.csv`
   - Or prepare datasets using `make_datasets.py` (see Data Preparation section below)

## Data Preparation

### `make_datasets.py` - Dataset Preparation and Feature Engineering Script

This script processes raw trading data into machine learning-ready datasets for FOMO detection. It loads transaction records, customer information, and market prices, then engineers behavioral features across 5-day rolling windows for each investor. The script calculates key FOMO indicators (price chasing, missed returns, momentum buying) and applies rule-based labeling to create ground-truth labels. It outputs four datasets: unlabeled features, labeled features, and an 80/20 train-test split with comprehensive statistics reporting.

**Build datasets from raw data:**

```bash
python make_datasets.py --build
```

**View dataset statistics:**

```bash
python make_datasets.py --stats
```

**Output files:**
- `data/output/fomo_feature_data.csv` - Features without labels
- `data/output/fomo_feature_label_data.csv` - Features with FOMO labels
- `data/output/fomo_train_data.csv` - Training set (80%)
- `data/output/fomo_test_data.csv` - Test set (20%)

## Model Training

### `build_model.py` - Model Training and Evaluation Script

This script handles the complete machine learning pipeline for FOMO detection. It can either train a new XGBoost classifier from scratch using the training data or load an existing pre-trained model. The script evaluates model performance on test data, generates FOMO detection results with probability scores and confidence levels, and identifies the top behavioral signals contributing to each prediction using SHAP values. It automatically saves trained models with timestamps to the `data/models/` directory and provides detailed statistics on FOMO score distribution across Low, Medium, and High levels.

**Train a new model:**

```bash
python build_model.py
```

**Load and evaluate an existing model:**

```bash
python build_model.py --load-model data/models/xgbclassifier_20260201_143022.json
```

**Output:**
- Saves trained model to `data/models/` with timestamp
- Prints model evaluation metrics (accuracy, ROC-AUC)
- Displays FOMO score distribution statistics
- Shows sample detection results

## Run the Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The dashboard will open in your default web browser (typically at `http://localhost:8501`).

**Dashboard Features:**
- Select from available trained models
- Filter investors by FOMO level (Low/Medium/High)
- Navigate through investors using Previous/Next buttons or direct ID input
- View detailed FOMO analysis for each trading window
- Explore SHAP explanations for model predictions
- Compare individual behavior against average patterns
