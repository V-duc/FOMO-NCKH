from constants import FOMOScoreThresholds as th_fomo_score

# Rule-based score (normalized 0-1)
def rule_based_score(row):
    s = 0
    s += min(1, max(0, row["avg_return_before_buy"]/0.05))
    s += min(1, max(0, row["buy_after_spike_ratio"]/0.5))
    s += min(1, max(0, row["avg_missed_return"]/0.03))
    return min(1, s/3)


def fomo_level(score):
    # Thresholds adjusted for highly skewed distribution
    # (mean ~1.8%, most values near 0)
    if score < th_fomo_score.LOW_FOMO:
        return "Low"
    elif score < th_fomo_score.MEDIUM_FOMO:
        return "Medium"
    else:
        return "High"
