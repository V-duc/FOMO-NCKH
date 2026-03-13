import pandas as pd
import jenkspy
from pathlib import Path
import numpy as np

# Path Configuration
try:
    from constants import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = "data/output"

INPUT_FILE = Path(OUTPUT_DIR) / "cluster_market_context.csv"
RESULTS_OUTPUT_FILE = Path(OUTPUT_DIR) / "thresholds_summary.csv"
# List of Group B Variables
RAW_FEATURES = ["raw_rsi_14", "raw_cumulative_return_5d", "raw_price_above_ma20", "raw_volatility_5d", "raw_return_1d"]

# =========================================================
# ASYMMETRIC CLIPPING CONFIGURATION (OPTIMIZED RATIOS)
# Definition: {variable_name: (lower_ratio, upper_ratio)}
# =========================================================
ASYMMETRIC_CONFIG = {
    "raw_return_1d": (0.05, 0.03),               # Lower 5%, Upper 3%
    "raw_cumulative_return_5d": (0.04, 0.03),    # Lower 4%, Upper 3%
    "raw_price_above_ma20": (0.05, 0.03),        # Lower 5%, Upper 3%
    "raw_rsi_14": (0.05, 0.00),                  # Lower 5%, Upper unchanged
    "raw_volatility_5d": (0.05, 0.05)            # Volatility 5% balanced
}

def calculate_gvf(data, breaks):
    """Calculate Goodness of Variance Fit (GVF) index"""
    # SDAM: Sum of squared Deviations from the Array Mean
    sdam = np.sum((data - np.mean(data))**2)
    
    # Classify data into 3 groups based on identified breaks
    classes = pd.cut(data, bins=breaks, labels=False, include_lowest=True)
    
    # SDCM: Sum of squared Deviations from Class Means
    sdcm = 0
    for i in range(len(breaks) - 1):
        class_data = data[classes == i]
        if len(class_data) > 0:
            sdcm += np.sum((class_data - np.mean(class_data))**2)
            
    return (sdam - sdcm) / sdam if sdam != 0 else 0

def extract_jenks_3_thresholds():
    if not INPUT_FILE.exists():
        print(f"❌ File not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    # ---------------------------------------------------------
    # STEP 1: PERFORM ASYMMETRIC CLIPPING
    # ---------------------------------------------------------
    for col, (low_p, high_p) in ASYMMETRIC_CONFIG.items():
        if col in df.columns:
            low_val = df[col].quantile(low_p)
            high_val = df[col].quantile(1 - high_p)
            df[col] = df[col].clip(lower=low_val, upper=high_val)

    print("="*110)
    print(f"{'🚀 PSYCHOLOGICAL STATE TRI-SECTION & GVF VERIFICATION (FINAL)':^110}")
    print("="*110)
    print(f"{'Feature':<25} | {'Bottom Threshold':<15} | {'FOMO Threshold':<15} | {'GVF Score':<12} | {'Status'}")
    print("-" * 110)
    threshold_results = []
    # ---------------------------------------------------------
    # STEP 2: EXECUTE JENKS & CALCULATE GVF
    # ---------------------------------------------------------
    # UNDERSTANDING GVF (GOODNESS OF VARIANCE FIT)
    # 1. PURPOSE: Evaluates the reliability and quality of Jenks Natural Breaks.
    # 2. SIGNIFICANCE: Measures the percentage of variance explained by the grouping.
    #    - GVF > 0.8: Indicates an excellent fit with distinct, well-separated classes.
    # 3. OBJECTIVE: Provides mathematical validation thresholds, ensuring
    #    they are data-driven rather than arbitrary. This optimizes the precision 
    #    of Labeling Functions (LFs) within the Snorkel framework.
    # 4. FORMULA: GVF = 1 - (SDCM / SDAM)
    for col in RAW_FEATURES:
        data = df[col].dropna()
        if data.empty: continue
            
        # n_classes=3 returns list: [min, break1, break2, max]
        breaks = jenkspy.jenks_breaks(data, n_classes=3)
        
        gvf_score = calculate_gvf(data.values, breaks)
        th_low = breaks[1]   # Boundary: Panic/Fear -> Normal
        th_high = breaks[2]  # Boundary: Normal -> Euphoria/FOMO
        
        # Display formatting (convert to % if variable is a return)
        fmt = lambda x: f"{x*100:.2f}%" if "return" in col else f"{x:.3f}"
        status = "⭐ EXCELLENT" if gvf_score > 0.8 else "✅ GOOD"
        
        print(f"{col:<25} | {fmt(th_low):<15} | {fmt(th_high):<15} | {gvf_score:<12.4f} | {status}")
        threshold_results.append({
            "Feature": col,
            "Bottom_Threshold": th_low,
            "FOMO_Threshold": th_high,
            "GVF_Score": round(gvf_score, 4),
        })
    results_df = pd.DataFrame(threshold_results)
    results_df.to_csv(RESULTS_OUTPUT_FILE, index=False)
    print(f"✅ Summary saved to: {RESULTS_OUTPUT_FILE}")
if __name__ == "__main__":
    extract_jenks_3_thresholds()