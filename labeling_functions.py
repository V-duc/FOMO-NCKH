"""
labeling_functions.py — Labeling Functions cho FOMO detection.

Quy ước nhãn:
    FOMO    =  1  — lệnh có dấu hiệu FOMO
    NORMAL  =  0  — lệnh rõ ràng KHÔNG phải FOMO (chỉ LF_rsi_extreme mới dùng)
    ABSTAIN = -1  — không đủ thông tin để kết luận

Lý do chỉ LF_rsi_extreme label NORMAL:
    RSI < 50 = oversold = mua khi thị trường yếu = rational, ngược hoàn toàn FOMO.
    Các LF khác chỉ biết khi nào là FOMO, không biết khi nào là NORMAL.
    NORMAL là "default state" — Snorkel suy ra từ sự vắng mặt của FOMO signal.

Thresholds (data-driven từ EDA + Jenks trên dense group):
    return_5d  > 0.037  — Jenks natural break (dense investors, Jul2020-Nov2022)
    rsi_14     > 75     — Wilder overbought definition (không dùng Jenks vì lý thuyết rõ)
    rsi_14     < 50     — midpoint RSI = oversold/neutral boundary
    value      > P90    — bất thường so với thói quen của chính investor đó
    bollinger  = 1      — price > MA20 + 2*std (đã tính sẵn trong lf_input.csv)
    same_day   = 0      — days_since_last_buy == 0

Input schema (1 row của lf_input.csv, access qua x.column_name):
    tx_id, investor_id, asset_id, timestamp
    days_since_last_buy    — float, NaN cho lệnh BUY đầu tiên của investor
    p90_trade_value        — float, NaN nếu investor sparse (< 5 BUY)
    price_above_bollinger  — 0.0 / 1.0 / NaN (warmup period)
    channel                — str
    totalValue             — float
    return_5d              — float, NaN đầu series
    rsi_14                 — float, NaN đầu series
    volatility_5d          — float
    market_price           — float
    ma_20d                 — float
    price_above_ma20       — float
"""

import pandas as pd
import numpy as np
from snorkel.labeling import labeling_function

# ── Label constants ───────────────────────────────────────────────────────
FOMO    =  1
NORMAL  =  0
ABSTAIN = -1

# ── Thresholds ────────────────────────────────────────────────────────────
RETURN_5D_THRESHOLD  = 0.0306   # Jenks natural break trên dense group
RSI_FOMO_THRESHOLD   = 75.0    # Wilder overbought — lý thuyết, không Jenks
RSI_NORMAL_THRESHOLD = 50.0    # RSI < 50 = oversold/neutral = rational buy


# ═══════════════════════════════════════════════════════════════════════════
# NHÓM 1 — IMPULSIVITY & ACTION BIAS
# ═══════════════════════════════════════════════════════════════════════════

@labeling_function()
def LF_trade_cluster(x):
    """
    Bắt hành vi "giao dịch chùm" — mua trong cùng ngày với lệnh mua trước.

    Lý thuyết: Action Bias (Glaser & Weber, 2007) — khi hưng phấn hoặc hoảng loạn,
    investor có nhu cầu "phải làm gì đó ngay" → đặt nhiều lệnh liên tiếp trong ngày.
    Dùng == 0 (same-day) thay vì <= 1 để đảm bảo precision.

    Vùng:
        days_since_last_buy == 0  → FOMO
        NaN (lệnh BUY đầu tiên)  → ABSTAIN
        > 0                       → ABSTAIN
    Bắt mua nhồi cùng mã trong cùng ngày — panic buying / FOMO rõ nhất.
    Đổi từ same-day bất kỳ → same-day same-asset để giảm false positive.
    """
    if pd.isna(x.days_since_last_buy):
        return ABSTAIN
    if x.days_since_last_buy == 0 and x.asset_id == x.prev_asset_id:
        return FOMO
    return ABSTAIN


# ═══════════════════════════════════════════════════════════════════════════
# NHÓM 2 — TREND CHASING & HERD BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════

@labeling_function()
def LF_return_momentum(x):
    """
    Bắt hành vi mua sau khi giá đã tăng mạnh trong 5 ngày (price chasing).

    Lý thuyết: Representativeness Bias (Sood et al., 2023) — investor nhầm đà
    tăng ngắn hạn là xu hướng dài hạn → FOMO mua vào.
    Barber & Odean (2008): retail investors là "attention-driven buyers".

    Threshold: return_5d > 0.0306 — Jenks natural break trên dense group.

    Vùng:
        return_5d > 0.0306  → FOMO
        NaN                → ABSTAIN
        <= 0.0306           → ABSTAIN
    """
    if pd.isna(x.return_5d):
        return ABSTAIN
    if x.return_5d > RETURN_5D_THRESHOLD:
        return FOMO
    return ABSTAIN


@labeling_function()
def LF_rsi_extreme(x):
    """
    Bắt mua khi overbought cực đoan (RSI > 75). Label NORMAL khi RSI < 50.

    Lý thuyết: Barber & Odean (2008) — "extreme_green_chasing": investor FOMO
    giải ngân lớn nhất tại vùng hưng phấn tột độ.
    RSI > 75: overbought mạnh theo Wilder (1978).
    RSI < 50: oversold/neutral — mua lúc này là rational, ngược FOMO.

    LF duy nhất có 3 vùng vì RSI < 50 có lý thuyết support rõ cho NORMAL.

    Vùng:
        rsi_14 > 75          → FOMO
        rsi_14 < 50          → NORMAL
        50 <= rsi_14 <= 75   → ABSTAIN
        NaN                  → ABSTAIN
    """
    if pd.isna(x.rsi_14):
        return ABSTAIN
    if x.rsi_14 > RSI_FOMO_THRESHOLD:
        return FOMO
    if x.rsi_14 < RSI_NORMAL_THRESHOLD:
        return NORMAL
    return ABSTAIN


@labeling_function()
def LF_bollinger_breakout(x):
    """
    Bắt mua khi giá vượt dải Bollinger trên (MA20 + 2*std) — statistical extreme.

    Lý thuyết: Da, Engelberg & Gao (2011) — biến động giá vượt ranh giới thống kê
    tạo "attention shock", kích hoạt dòng tiền FOMO từ retail investors.
    Hoffmann & Shefrin (2014): investor cá nhân lạm dụng tín hiệu kỹ thuật để mua đuổi.

    price_above_bollinger đã tính sẵn trong lf_input.csv:
        1.0 = price > MA20 + 2*std_20d
        0.0 = trong band
        NaN = warmup period

    Vùng:
        price_above_bollinger == 1  → FOMO
        price_above_bollinger == 0  → ABSTAIN
        NaN                         → ABSTAIN
    """
    if pd.isna(x.price_above_bollinger):
        return ABSTAIN
    if x.price_above_bollinger == 1.0:
        return FOMO
    return ABSTAIN


# ═══════════════════════════════════════════════════════════════════════════
# NHÓM 3 — CAPITAL INDISCIPLINE
# ═══════════════════════════════════════════════════════════════════════════

@labeling_function()
def LF_value_spike(x):
    """
    Bắt lệnh "YOLO" — giá trị vọt lên bất thường so với thói quen của investor.

    Lý thuyết: Kumar et al. (2011) — lệnh vượt P90 cá nhân = Sensation Seeking /
    Lottery-like demand. Khoảnh khắc investor mất kiểm soát cảm xúc hoàn toàn.
    Dùng P90 per investor (không phải population) vì "lớn" là tương đối với
    hành vi thông thường của từng người.

    Vùng:
        totalValue > p90_trade_value  → FOMO
        NaN p90 (sparse, < 5 BUY)    → ABSTAIN
        <= p90                        → ABSTAIN
    """
    if pd.isna(x.p90_trade_value):
        return ABSTAIN
    if x.totalValue > x.p90_trade_value:
        return FOMO
    return ABSTAIN


# ── Export list — dùng trong run_snorkel.py ───────────────────────────────
LFS = [
    LF_trade_cluster,
    LF_return_momentum,
    LF_rsi_extreme,
    LF_bollinger_breakout,
    LF_value_spike,
]
