"""
run_snorkel.py — Chạy Snorkel Label Model trên lf_input.csv.

Pipeline:
    1. Load lf_input.csv
    2. Apply LFs → L_train matrix
    3. Phân tích LF coverage, conflicts, overlaps
    4. Ước lượng class_balance từ union coverage của LFs
    5. Train LabelModel với class_balance data-driven
    6. Predict soft labels (probability) per transaction
    7. Handle all-abstain rows — giữ nguyên 0.5, không ép về 0
    8. Save snorkel_labels.csv

Output columns:
    tx_id         — join key về enriched_trades_train.csv
    investor_id
    timestamp
    fomo_prob     — soft label từ LabelModel [0.0, 1.0]
    lf_votes      — số LF vote FOMO (0-5), để debug
    all_abstain   — bool, True nếu tất cả LF đều ABSTAIN

Chạy:
    python run_snorkel.py
"""

import pandas as pd
import numpy as np
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

from labeling_functions import LFS, FOMO, NORMAL, ABSTAIN
from constants import OUTPUT_DIR

INPUT_FILE  = f"{OUTPUT_DIR}/lf_input.csv"
OUTPUT_FILE = f"{OUTPUT_DIR}/snorkel_labels.csv"

LABEL_MODEL_EPOCHS = 500
LABEL_MODEL_SEED   = 42
LOG_FREQ           = 100


# ── 1. Load ───────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading lf_input.csv...")
print("=" * 60)
df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
print(f"  Loaded: {len(df):,} rows, {df['investor_id'].nunique():,} investors")


# ── 2. Apply LFs → L_train ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Applying Labeling Functions...")
print("=" * 60)
applier = PandasLFApplier(lfs=LFS)
L_train = applier.apply(df)

print(f"  L_train shape: {L_train.shape}")
print(f"  Columns: {[lf.name for lf in LFS]}")


# ── 3. LF Analysis ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: LF Analysis")
print("=" * 60)
lf_analysis = LFAnalysis(L=L_train, lfs=LFS).lf_summary()
print(lf_analysis.to_string())

print("\n  Giải thích:")
print("  - Coverage  : % lệnh được LF này label (không ABSTAIN)")
print("  - Overlaps  : % lệnh được >= 2 LF label cùng lúc")
print("  - Conflicts : % lệnh bị 2 LF label khác nhau (không đáng lo)")


# ── 4. Ước lượng class_balance từ union coverage ─────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Estimating class_balance from union coverage")
print("=" * 60)

# Union coverage: số lệnh bị ít nhất 1 LF flag là FOMO
any_fomo = (L_train == FOMO).any(axis=1)
n_fomo_flagged = any_fomo.sum()
fomo_ratio = n_fomo_flagged / len(df)
normal_ratio = 1 - fomo_ratio

print(f"  Lệnh bị ít nhất 1 LF flag FOMO : {n_fomo_flagged:,} ({fomo_ratio*100:.1f}%)")
print(f"  Lệnh không bị flag FOMO nào    : {(~any_fomo).sum():,} ({normal_ratio*100:.1f}%)")
print(f"\n  → class_balance sẽ dùng: [{normal_ratio:.3f}, {fomo_ratio:.3f}]")
print(f"  Lý do: union coverage là ước lượng upper bound của FOMO ratio.")
print(f"  LabelModel sẽ học precision thực của từng LF để refine con số này.")

# Sanity check — nếu ratio quá thấp hoặc quá cao thì warn
if fomo_ratio < 0.05:
    print(f"\n  [WARNING] FOMO ratio = {fomo_ratio*100:.1f}% — rất thấp.")
    print(f"  Coverage LF có thể quá hẹp. Cân nhắc review threshold.")
elif fomo_ratio > 0.50:
    print(f"\n  [WARNING] FOMO ratio = {fomo_ratio*100:.1f}% — quá cao.")
    print(f"  LF có thể đang over-flag. Cân nhắc tăng threshold.")

class_balance = [normal_ratio, fomo_ratio]


# ── 5. Train LabelModel ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Training LabelModel...")
print("=" * 60)
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(
    L_train=L_train,
    n_epochs=LABEL_MODEL_EPOCHS,
    log_freq=LOG_FREQ,
    seed=LABEL_MODEL_SEED,
    class_balance=class_balance,
)
print("  LabelModel training complete.")


# ── 6. Predict soft labels ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Predicting soft labels...")
print("=" * 60)

# predict_proba trả về [P(NORMAL), P(FOMO)] per row
proba = label_model.predict_proba(L=L_train)
fomo_prob = proba[:, 1]   # P(FOMO)

print(f"  fomo_prob distribution:")
print(f"    Mean   : {fomo_prob.mean():.4f}")
print(f"    Median : {np.median(fomo_prob):.4f}")
print(f"    Std    : {fomo_prob.std():.4f}")
print(f"    Min    : {fomo_prob.min():.4f}")
print(f"    Max    : {fomo_prob.max():.4f}")

# Kiểm tra all-abstain rows
all_abstain = (L_train == ABSTAIN).all(axis=1)
n_all_abstain = all_abstain.sum()
print(f"\n  All-abstain rows: {n_all_abstain:,} ({n_all_abstain/len(df)*100:.1f}%)")
print(f"  → Giữ nguyên soft label ~0.5 cho all-abstain (không ép về 0)")
print(f"  → XGBoost sẽ thấy uncertainty thật sự từ những rows này")

# Verify: all-abstain rows nên có fomo_prob gần class_balance prior
abstain_probs = fomo_prob[all_abstain]
if n_all_abstain > 0:
    print(f"  All-abstain fomo_prob mean: {abstain_probs.mean():.4f} "
          f"(expected ≈ {fomo_ratio:.4f})")


# ── 7. Số lệnh vote FOMO per row (debug) ─────────────────────────────────
lf_votes = (L_train == FOMO).sum(axis=1)


# ── 8. Build output & save ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Saving snorkel_labels.csv...")
print("=" * 60)

output = pd.DataFrame({
    "tx_id"       : df["tx_id"].values,
    "investor_id" : df["investor_id"].values,
    "timestamp"   : df["timestamp"].values,
    "fomo_prob"   : fomo_prob,
    "lf_votes"    : lf_votes,
    "all_abstain" : all_abstain,
})

# Thêm từng LF vote để debug / LF analysis sau này
for i, lf in enumerate(LFS):
    output[f"lf_{lf.name}"] = L_train[:, i]

output.to_csv(OUTPUT_FILE, index=False)
print(f"  ✓ Saved: {OUTPUT_FILE}")
print(f"  Shape: {output.shape}")


# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total BUY transactions : {len(output):,}")
print(f"All-abstain            : {n_all_abstain:,} ({n_all_abstain/len(output)*100:.1f}%)")
print(f"\nfomo_prob buckets:")
buckets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
labels  = ["[0.0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0]"]
for i, label in enumerate(labels):
    mask = (fomo_prob >= buckets[i]) & (fomo_prob < buckets[i+1])
    print(f"  {label}: {mask.sum():,} ({mask.mean()*100:.1f}%)")

print(f"\nLF votes distribution (how many LFs flagged FOMO per tx):")
for v in range(len(LFS) + 1):
    n = (lf_votes == v).sum()
    print(f"  {v} LFs voted FOMO: {n:,} ({n/len(output)*100:.1f}%)")

print("\n✓ Done. Next step: join fomo_prob vào feature table → train XGBoost.")







# ── Validation set (Professional investors) ───────────────────────────────
import os
VAL_INPUT  = f"{OUTPUT_DIR}/lf_input_val.csv"
VAL_OUTPUT = f"{OUTPUT_DIR}/snorkel_labels_val.csv"

if os.path.exists(VAL_INPUT):
    print("\n" + "=" * 60)
    print("VALIDATION: Professional investors sanity check")
    print("=" * 60)
    
    df_val  = pd.read_csv(VAL_INPUT, parse_dates=["timestamp"])
    L_val   = applier.apply(df_val)
    proba_val = label_model.predict_proba(L=L_val)
    fomo_prob_val = proba_val[:, 1]

    print(f"  Total Professional BUY: {len(df_val):,}")
    print(f"  fomo_prob mean  : {fomo_prob_val.mean():.4f}  (train: {fomo_prob.mean():.4f})")
    print(f"  fomo_prob median: {np.median(fomo_prob_val):.4f}  (train: {np.median(fomo_prob):.4f})")
    print(f"  % prob > 0.5    : {(fomo_prob_val > 0.5).mean()*100:.1f}%  (train: {(fomo_prob > 0.5).mean()*100:.1f}%)")

    print("\n  LF coverage — Professional vs Train:")
    for i, lf in enumerate(LFS):
        val_fomo  = (L_val[:, i] == FOMO).mean() * 100
        train_fomo = (L_train[:, i] == FOMO).mean() * 100
        print(f"  {lf.name:<30} val: {val_fomo:.1f}%  train: {train_fomo:.1f}%")

    print("\n  → Kỳ vọng: Professional fomo_prob thấp hơn train rõ rệt")
    print("  → Nếu không → review lại LF threshold")

    # Save
    out_val = pd.DataFrame({
        "tx_id"      : df_val["tx_id"].values,
        "investor_id": df_val["investor_id"].values,
        "timestamp"  : df_val["timestamp"].values,
        "fomo_prob"  : fomo_prob_val,
        "lf_votes"   : (L_val == FOMO).sum(axis=1),
        "all_abstain": (L_val == ABSTAIN).all(axis=1),
    })
    out_val.to_csv(VAL_OUTPUT, index=False)
    print(f"\n  ✓ Saved: {VAL_OUTPUT}")