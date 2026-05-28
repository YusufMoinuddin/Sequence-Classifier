"""
data_integrity_audit.py
=======================
Checks two things Murat asked about:
  1. Whether the train/val/test split is stratified (class ratios consistent)
  2. Whether there is any sequence-level data leakage between splits

Run from the project root with the venv active:
  python data_integrity_audit.py

Expects files at:
  data/deep_enzymology_qmproxy_train.csv
  data/deep_enzymology_qmproxy_val.csv
  data/deep_enzymology_qmproxy_test.csv
"""

import pandas as pd
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
FILES = {
    "train": DATA_DIR / "deep_enzymology_qmproxy_train.csv",
    "val":   DATA_DIR / "deep_enzymology_qmproxy_val.csv",
    "test":  DATA_DIR / "deep_enzymology_qmproxy_test.csv",
}
SEQUENCE_COL = "sequence"   # ← adjust if your column is named differently
LABEL_COL    = "label"      # ← adjust if your label column has a different name

# ── Load ───────────────────────────────────────────────────────────────────
splits = {}
print("=" * 60)
print("LOADING SPLITS")
print("=" * 60)
for name, path in FILES.items():
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    splits[name] = df
    print(f"  {name:5s}: {len(df):5d} rows  |  columns: {list(df.columns)}")

# ── Auto-detect columns if defaults not found ──────────────────────────────
sample_df = splits["train"]
if SEQUENCE_COL not in sample_df.columns:
    # Try common alternatives
    for candidate in ["seq", "Sequence", "kmer", "8mer", "sequence_8mer"]:
        if candidate in sample_df.columns:
            SEQUENCE_COL = candidate
            print(f"\n  [INFO] Using '{SEQUENCE_COL}' as the sequence column.")
            break
    else:
        print(f"\n  [ERROR] Could not find a sequence column.")
        print(f"          Available columns: {list(sample_df.columns)}")
        print(f"          Edit SEQUENCE_COL at the top of this script and rerun.")
        sys.exit(1)

if LABEL_COL not in sample_df.columns:
    for candidate in ["Label", "class", "Class", "target", "y"]:
        if candidate in sample_df.columns:
            LABEL_COL = candidate
            print(f"  [INFO] Using '{LABEL_COL}' as the label column.")
            break
    else:
        print(f"\n  [ERROR] Could not find a label column.")
        print(f"          Available columns: {list(sample_df.columns)}")
        sys.exit(1)

# ── 1. STRATIFICATION CHECK ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. STRATIFICATION CHECK")
print("=" * 60)
print(f"  Label 0 = DNMT3A (majority)  |  Label 1 = DNMT3B (minority)")
print(f"  {'Split':<8} {'Total':>7} {'DNMT3A(0)':>10} {'DNMT3B(1)':>10} {'Ratio(A/B)':>12} {'%DNMT3B':>9}")
print(f"  {'-'*58}")

all_dfs = pd.concat(splits.values())
overall_counts = all_dfs[LABEL_COL].value_counts().sort_index()
overall_ratio  = overall_counts[0] / overall_counts[1]
overall_pct_b  = 100 * overall_counts[1] / len(all_dfs)

for name, df in splits.items():
    counts = df[LABEL_COL].value_counts().sort_index()
    n0 = counts.get(0, 0)
    n1 = counts.get(1, 0)
    ratio = n0 / n1 if n1 > 0 else float("inf")
    pct_b = 100 * n1 / len(df)
    print(f"  {name:<8} {len(df):>7} {n0:>10} {n1:>10} {ratio:>12.2f} {pct_b:>8.1f}%")

print(f"  {'OVERALL':<8} {len(all_dfs):>7} {overall_counts[0]:>10} {overall_counts[1]:>10} {overall_ratio:>12.2f} {overall_pct_b:>8.1f}%")

# Verdict
ratios = []
for name, df in splits.items():
    counts = df[LABEL_COL].value_counts().sort_index()
    n0 = counts.get(0, 0)
    n1 = counts.get(1, 0)
    ratios.append(n0 / n1 if n1 > 0 else float("inf"))

max_deviation = max(abs(r - overall_ratio) / overall_ratio for r in ratios)
print(f"\n  Max ratio deviation across splits: {max_deviation*100:.1f}%")
if max_deviation < 0.05:
    print("  ✅ STRATIFIED — all splits within 5% of overall class ratio.")
elif max_deviation < 0.15:
    print("  ⚠️  APPROXIMATELY STRATIFIED — deviation < 15%. Acceptable but note in paper.")
else:
    print("  ❌ NOT STRATIFIED — class ratios differ significantly between splits.")

# ── 2. LEAKAGE CHECK ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. DATA LEAKAGE CHECK (sequence overlap between splits)")
print("=" * 60)

seq_sets = {name: set(df[SEQUENCE_COL].str.upper().str.strip())
            for name, df in splits.items()}

pairs = [("train", "val"), ("train", "test"), ("val", "test")]
any_leakage = False

for s1, s2 in pairs:
    overlap = seq_sets[s1] & seq_sets[s2]
    status = "✅ CLEAN" if len(overlap) == 0 else f"❌ LEAKAGE — {len(overlap)} shared sequences"
    print(f"  {s1} ∩ {s2}: {len(overlap)} shared sequences  →  {status}")
    if overlap:
        any_leakage = True
        print(f"    First 5 overlapping sequences: {list(overlap)[:5]}")

# Check for duplicates within each split
print()
for name, df in splits.items():
    n_dupes = df[SEQUENCE_COL].duplicated().sum()
    status = "✅ no duplicates" if n_dupes == 0 else f"⚠️  {n_dupes} duplicate rows"
    print(f"  {name} internal duplicates: {status}")

# ── 3. SIZE SANITY CHECK ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. SPLIT SIZE SANITY CHECK")
print("=" * 60)
total = sum(len(df) for df in splits.values())
for name, df in splits.items():
    pct = 100 * len(df) / total
    print(f"  {name:<8}: {len(df):>5} samples  ({pct:.1f}% of total)")
print(f"  {'TOTAL':<8}: {total:>5} samples")

# ── SUMMARY ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY FOR PAPER")
print("=" * 60)
print(f"  Total sequences : {total}")
print(f"  DNMT3A (label 0): {overall_counts[0]} ({100*overall_counts[0]/total:.1f}%)")
print(f"  DNMT3B (label 1): {overall_counts[1]} ({100*overall_counts[1]/total:.1f}%)")
print(f"  Class imbalance : {overall_ratio:.2f}:1  (DNMT3A:DNMT3B)")
print(f"  Sequence column : '{SEQUENCE_COL}'")
print(f"  Label column    : '{LABEL_COL}'")
print(f"  Leakage found   : {'YES ❌' if any_leakage else 'NO ✅'}")
print("=" * 60)