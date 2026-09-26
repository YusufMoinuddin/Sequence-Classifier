"""
aggregate_vqc_results.py
========================
Merges the per-task CSVs written by run_single_vqc.py (one per encoding x seed) into
a combined table, a paper-ready summary, and one confusion matrix per encoding.

Run after the Slurm array finishes:
  python aggregate_vqc_results.py
  python aggregate_vqc_results.py --results-dir results

Outputs:
  vqc_all_configs_results.csv    every run, tagged with encoding + seed
  vqc_all_configs_summary.txt    per-encoding mean +/- std, fixed-width + LaTeX
  vqc_confusion_matrices.png     one panel per encoding

CONFUSION MATRIX CONVENTION: the median-performing seed by balanced accuracy is used
(not the mean or the best), so the matrix shown is a real single run rather than an
average of counts that never occurred. The chosen seed is named in the panel title and
in the summary text.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.qml_encodings import CONFIG_DESCRIPTIONS

ENCODING_ORDER = ["config1", "config2", "config3"]
ENCODING_LABELS = {
    "config1": "Config 1 (4-pos slice)",
    "config2": "Config 2 (pair condensation)",
    "config3": "Config 3 (full 8-position)",
}

METRICS = ["balanced_acc", "roc_auc", "pr_auc",
           "recall_dnmt3b", "precision_dnmt3b", "f1_dnmt3b"]
METRIC_LABELS = ["Balanced Acc", "ROC-AUC", "PR-AUC",
                 "DNMT3B Recall", "DNMT3B Prec", "DNMT3B F1"]

OUT_CSV = "vqc_all_configs_results.csv"
OUT_TXT = "vqc_all_configs_summary.txt"
OUT_FIG = "vqc_confusion_matrices.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    files = sorted(rdir.glob("vqc_*_seed*.csv"))
    if not files:
        raise SystemExit(
            f"No per-run CSVs found in {rdir}/ (expected vqc_<encoding>_seed<N>.csv).\n"
            "Run the Slurm array first, or pass --results-dir."
        )

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values(["encoding", "seed"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"Loaded {len(files)} run files -> {len(df)} rows")
    print(f"  encodings: {sorted(df.encoding.unique())}")
    for enc in sorted(df.encoding.unique()):
        seeds = sorted(df[df.encoding == enc].seed.tolist())
        print(f"    {enc}: seeds {seeds}")
        if len(seeds) != 5:
            print(f"      WARNING: expected 5 seeds, found {len(seeds)}")
    print(f"  wrote {OUT_CSV}")

    present = [e for e in ENCODING_ORDER if e in set(df.encoding)]

    # --- summary table ---
    lines = []
    lines.append("=" * 100)
    lines.append("  VQC ENCODING COMPARISON — TEST SET")
    lines.append(f"  {len(present)} encodings x seeds; mean +/- std across seeds")
    lines.append("  Matched hyperparameters: Adam, lr=0.01, batch=32, 30 epochs, 256 shots,")
    lines.append("  2 ansatz layers, pos_weight = n_neg/n_pos, best-val-loss checkpoint,")
    lines.append("  threshold 0.5, identical train/val/test splits.")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"  {'Encoding':<30}" + "".join(f"{lab:>17}" for lab in METRIC_LABELS))
    lines.append("  " + "-" * (30 + 17 * len(METRIC_LABELS)))

    stats = {}
    for enc in present:
        sub = df[df.encoding == enc]
        stats[enc] = {m: (sub[m].mean(), sub[m].std(ddof=1) if len(sub) > 1 else 0.0)
                      for m in METRICS}
        cells = "".join(f"{stats[enc][m][0]:>9.4f}±{stats[enc][m][1]:<7.4f}" for m in METRICS)
        lines.append(f"  {ENCODING_LABELS.get(enc, enc):<30}{cells}")

    lines.append("")
    lines.append("  Qubits per encoding: config1=4, config2=4, config3=8")
    for enc in present:
        lines.append(f"    {enc}: {CONFIG_DESCRIPTIONS[enc]}")
    lines.append("")

    # --- confusion matrices, median seed by balanced accuracy ---
    lines.append("  CONFUSION MATRICES — median-performing seed by balanced accuracy")
    lines.append("  (a real single run, not an average of counts)")
    lines.append("")

    median_rows = {}
    for enc in present:
        sub = df[df.encoding == enc].sort_values("balanced_acc").reset_index(drop=True)
        row = sub.iloc[len(sub) // 2]
        median_rows[enc] = row
        lines.append(f"    {ENCODING_LABELS.get(enc, enc)} — seed {int(row.seed)} "
                     f"(balanced acc {row.balanced_acc:.4f})")
        lines.append(f"                          Pred 3A   Pred 3B")
        lines.append(f"      True 3A (DNMT3A)    {int(row.TN):>7}   {int(row.FP):>7}")
        lines.append(f"      True 3B (DNMT3B)    {int(row.FN):>7}   {int(row.TP):>7}")
        lines.append("")

    lines.append("=" * 100)
    lines.append("")
    lines.append("% ---- LaTeX ----")
    lines.append("\\begin{tabular}{l" + "c" * len(METRICS) + "}")
    lines.append("\\hline")
    lines.append("Encoding & " + " & ".join(METRIC_LABELS) + " \\\\")
    lines.append("\\hline")
    for enc in present:
        cells = " & ".join(f"${stats[enc][m][0]:.3f} \\pm {stats[enc][m][1]:.3f}$" for m in METRICS)
        lines.append(f"{ENCODING_LABELS.get(enc, enc)} & {cells} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    text = "\n".join(lines)
    Path(OUT_TXT).write_text(text + "\n")
    print("\n" + text)
    print(f"\n  wrote {OUT_TXT}")

    # --- figure ---
    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4.6))
    if len(present) == 1:
        axes = [axes]
    for ax, enc in zip(axes, present):
        r = median_rows[enc]
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]], dtype=float)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=15, fontweight="bold")
        ax.set_title(f"{ENCODING_LABELS.get(enc, enc)}\n"
                     f"median seed {int(r.seed)} | bal acc {r.balanced_acc:.3f}",
                     fontsize=10)
        ax.set_xticks([0, 1], ["Pred 3A", "Pred 3B"])
        ax.set_yticks([0, 1], ["True 3A", "True 3B"])
    fig.suptitle("VQC encoding comparison — test-set confusion matrices "
                 "(median-performing seed by balanced accuracy)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"  wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
