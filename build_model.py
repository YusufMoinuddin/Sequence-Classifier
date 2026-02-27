# build_dataset.py
import os
import random
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np

# Folder containing all enzyme .txt files
DATA_DIR = "deep_enzymology_raw"

# List of enzyme files to include (exclude "No enzyme" ones)
# You can comment out any you don't want to use.
FILES = [
    # DNMT3A
    ("3A 0.5 CN.txt", "DNMT3A"),
    ("3A 10 µM CH.txt", "DNMT3A"),
    ("3A 20 µM CH.txt", "DNMT3A"),

    # DNMT3B
    ("3B WT 10 µM 60 min CN.txt", "DNMT3B"),
    ("3B 10 µM CH.txt", "DNMT3B"),
    ("3B 20 µM CH.txt", "DNMT3B"),
]

def file_path(fname: str) -> str:
    """Return full path to the given file in DATA_DIR."""
    p = os.path.join(DATA_DIR, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")
    return p

def cg_positions(s: str):
    """Yield indices of all CpG sites in a sequence."""
    i = s.find("CG")
    while i != -1:
        yield i
        i = s.find("CG", i + 1)

def extract_central_motif(seq: str, flank: int = 3) -> Optional[str]:
    """
    Return NNNCGNNN motif around the CpG closest to the center.

    For flank=3, this returns an 8-mer: 3 bases left + 'CG' + 3 bases right.
    """
    s = seq.strip().upper()
    if len(s) < (2 * flank + 2):
        return None
    centers = list(cg_positions(s))
    if not centers:
        return None
    mid = len(s) // 2
    cidx = min(centers, key=lambda i: abs(i - mid))
    left = cidx - flank
    right = cidx + 2 + flank
    if left < 0 or right > len(s):
        return None
    motif = s[left:cidx] + "CG" + s[cidx + 2:right]
    return motif if len(motif) == 2 * flank + 2 else None

def load_gao_reads() -> pd.DataFrame:
    """
    Load deep enzymology .txt files and extract central motifs per read.

    Returns a read-level DataFrame with columns:
      - sequence: NNNCGNNN motif
      - label: 'DNMT3A' or 'DNMT3B' indicating which enzyme file this read came from
    """
    rows: List[Tuple[str, str]] = []
    for fname, label in FILES:
        path = file_path(fname)
        count = 0
        motif_count = 0
        with open(path, "r") as f:
            for line in f:
                seq = line.strip()
                if not seq:
                    continue
                count += 1
                motif = extract_central_motif(seq, flank=3)
                if motif:
                    rows.append((motif, label))
                    motif_count += 1
        print(f"Parsed {motif_count:>7,d} motifs from {fname} ({label}), "
              f"raw reads={count:>7,d}")
    if not rows:
        raise RuntimeError("No motifs parsed—check folder or filenames.")
    df = pd.DataFrame(rows, columns=["sequence", "label"])
    return df

def motif_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate read-level motifs into motif-level counts per enzyme.

    Input:
      df with columns ['sequence', 'label'] where label ∈ {'DNMT3A', 'DNMT3B'}

    Output:
      DataFrame with columns:
        - sequence: NNNCGNNN motif
        - meth_A: count of reads for DNMT3A
        - meth_B: count of reads for DNMT3B
        - total: meth_A + meth_B
    """
    # groupby sequence + enzyme label, count reads
    counts = (
        df.groupby(["sequence", "label"])
          .size()
          .unstack("label", fill_value=0)
    )

    # Ensure both columns exist
    if "DNMT3A" not in counts.columns:
        counts["DNMT3A"] = 0
    if "DNMT3B" not in counts.columns:
        counts["DNMT3B"] = 0

    counts = counts.reset_index()
    counts = counts.rename(columns={
        "DNMT3A": "meth_A",
        "DNMT3B": "meth_B",
    })
    counts["total"] = counts["meth_A"] + counts["meth_B"]
    return counts

def write_fasta(df: pd.DataFrame, path: str, label_col: str = "label_str"):
    """Write combined dataset to FASTA format."""
    with open(path, "w") as out:
        for i, row in enumerate(df.itertuples(index=False), 1):
            seq = getattr(row, "sequence")
            lab = getattr(row, label_col)
            out.write(f">{lab}|id={i}\n{seq}\n")

def split_df(df: pd.DataFrame, seed: int = 1337, tvt=(0.8, 0.1, 0.1)):
    """
    Stratified train/val/test split on the 'label' column.

    Works whether label is string ('DNMT3A'/'DNMT3B') or numeric (0.0/1.0).
    """
    random.seed(seed)
    parts = []
    for lab, group in df.groupby("label"):
        idx = list(group.index)
        random.shuffle(idx)
        n = len(idx)
        n_train = int(n * tvt[0])
        n_val = int(n * tvt[1])
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        parts.append(("train", df.loc[train_idx]))
        parts.append(("val", df.loc[val_idx]))
        parts.append(("test", df.loc[test_idx]))

    train = pd.concat([p[1] for p in parts if p[0] == "train"]).sample(frac=1, random_state=seed)
    val   = pd.concat([p[1] for p in parts if p[0] == "val"]).sample(frac=1, random_state=seed)
    test  = pd.concat([p[1] for p in parts if p[0] == "test"]).sample(frac=1, random_state=seed)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

if __name__ == "__main__":
    print("Loading deep enzymology data from:", os.path.abspath(DATA_DIR))

    # 1. Read-level motifs
    reads_df = load_gao_reads()
    print("\nRead-level summary:")
    print("Total read-level rows:", len(reads_df))
    print(reads_df["label"].value_counts())

    # Save raw read-level motifs (optional, for QC)
    reads_df.to_csv("deep_enzymology_reads.csv", index=False)

    # 2. Motif-level counts per enzyme
    motif_df = motif_counts(reads_df)
    print("\nMotif-level summary:")
    print("Unique motifs:", len(motif_df))
    print(motif_df[["meth_A", "meth_B", "total"]].describe())

    # Optional: filter out motifs with very low total counts
    # (you can tweak this threshold or comment it out)
    min_total = 5
    motif_df = motif_df[motif_df["total"] >= min_total].reset_index(drop=True)
    print(f"\nAfter filtering motifs with total < {min_total}: {len(motif_df)} motifs remain")

    # 3. Define preference label: 1.0 if meth_B > meth_A, else 0.0
    motif_df["label"] = (motif_df["meth_B"] > motif_df["meth_A"]).astype(np.float32)

    # For convenience, also keep a string label for FASTA:
    motif_df["label_str"] = np.where(
        motif_df["label"] == 1.0,
        "DNMT3B_preferred",
        "DNMT3A_preferred",
    )

    # 4. Save combined motif-level dataset
    motif_df.to_csv("deep_enzymology_motifs.csv", index=False)
    write_fasta(motif_df, "deep_enzymology_motifs.fasta", label_col="label_str")
    print("\nWrote deep_enzymology_motifs.csv and deep_enzymology_motifs.fasta")

    # 5. Create train/val/test splits for the model
    # Keep the columns your model might care about
    model_df = motif_df[["sequence", "meth_A", "meth_B", "label"]].copy()
    train, val, test = split_df(model_df)

    train.to_csv("deep_enzymology_train.csv", index=False)
    val.to_csv("deep_enzymology_val.csv", index=False)
    test.to_csv("deep_enzymology_test.csv", index=False)
    print(f"\nSplits (motif-level, preference-labeled): "
          f"train={len(train)}, val={len(val)}, test={len(test)}")
