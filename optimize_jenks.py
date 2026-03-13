import pandas as pd
import numpy as np
import jenkspy
from pathlib import Path

# 1. Path Configuration
INPUT_FILE = Path("data/output/cluster_market_context.csv")

# 2. Asymmetric Clipping Configuration (Lower_p, Upper_p)
# Strategy: Aggressive clipping on the lower end (5%) to eliminate bottom-fishing noise,
# while maintaining light clipping on the upper end (3%) to preserve FOMO signals.
ASYMMETRIC_CONFIG = {
    "raw_return_1d": (0.05, 0.03),
    "raw_cumulative_return_5d": (0.04, 0.03),
    "raw_price_above_ma20": (0.05, 0.03), # Price can be lightly clipped at both ends
    "raw_rsi_14": (0.05, 0.00),           # RSI does not require upper bound clipping
    "raw_volatility_5d": (0.05, 0.05)     # Volatility should be balanced
}

def calculate_gvf(data, breaks):
    """Calculate the Goodness of Variance Fit (GVF) index"""
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

def run_asymmetric_check():
    if not INPUT_FILE.exists():
        print("❌ Data file not found!")
        return

    df = pd.read_csv(INPUT_FILE)
    
    print("="*115)
    print(f"{'GVF VERIFICATION WITH ASYMMETRIC CLIPPING STRATEGY (FOMO PRIORITIZED)':^115}")
    print("="*115)
    header = f"{'Feature':<25} | {'Clip (L/U)':<12} | {'GVF Score':<12} | {'Bottom Threshold':<18} | {'FOMO Threshold'}"
    print(header)
    print("-" * 115)

    for col, (low_p, high_p) in ASYMMETRIC_CONFIG.items():
        if col not in df.columns: continue
        
        raw_series = df[col].dropna()
        
        # Apply Asymmetric Clipping
        low_val = raw_series.quantile(low_p)
        high_val = raw_series.quantile(1 - high_p)
        clipped = raw_series.clip(lower=low_val, upper=high_val)
        
        # Execute 3-class Jenks Optimization
        breaks = jenkspy.jenks_breaks(clipped, n_classes=3)
        gvf_score = calculate_gvf(clipped.values, breaks)
        
        # Display Formatting
        fmt = lambda x: f"{x*100:.2f}%" if "return" in col else f"{x:.3f}"
        clip_str = f"{low_p*100:>2.0f}% / {high_p*100:>2.0f}%"
        
        print(f"{col:<25} | {clip_str:<12} | {gvf_score:<12.4f} | {fmt(breaks[1]):<18} | {fmt(breaks[2])}")

    print("-" * 115)
    print("💡 EXPLANATION: Strong Left Clipping (L) removes Bottom-fishing noise. Light Right Clipping (U) preserves FOMO extremes.")
    print("=" * 115)

if __name__ == "__main__":
    run_asymmetric_check()