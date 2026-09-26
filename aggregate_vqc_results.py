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


def load_legacy_config1(path):
    """
    Load the previously published Config 1 results as a baseline row, so the
    comparison table keeps its baseline without re-running the failed encoding.

    multi_seed_results.csv was produced by multi_seed_eval.py at hyperparameters
    identical to run_single_vqc.py (2 layers / 30 epochs / batch 32 / lr 0.01 /
    256 shots, seeds 0-4), so the rows are directly comparable. It predates the
    precision/recall columns, so those are derived from TP/FP/FN here.
    """
    d = pd.read_csv(path)
    required = {"TP", "FP", "FN", "TN", "balanced_acc", "roc_auc", "pr_auc", "f1_dnmt3b"}
    missing = required - set(d.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns needed for the legacy row: {sorted(missing)}")

    d = d.copy()
    d["encoding"] = "config1"
    d["recall_dnmt3b"] = d["TP"] / (d["TP"] + d["FN"]).replace(0, np.nan)
    d["precision_dnmt3b"] = d["TP"] / (d["TP"] + d["FP"]).replace(0, np.nan)
    d["n_qubits"] = 4
    d["source"] = "prior run (not re-trained)"
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument(
        "--legacy-config1", default=None, metavar="CSV",
        help="Add the previously published Config 1 baseline from this CSV "
             "(e.g. multi_seed_results.csv) instead of re-training it.",
    )
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    files = sorted(rdir.glob("vqc_*_seed*.csv"))
    if not files:
        raise SystemExit(
            f"No per-run CSVs found in {rdir}/ (expected vqc_<encoding>_seed<N>.csv).\n"
            "Run the training commands first, or pass --results-dir."
        )

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["source"] = "this run"

    if args.legacy_config1:
        legacy = load_legacy_config1(args.legacy_config1)
        if "config1" in set(df.encoding):
            print(f"  NOTE: config1 was also trained in {rdir}/ — using the freshly "
                  f"trained rows and IGNORING {args.legacy_config1}")
        else:
            df = pd.concat([df, legacy], ignore_index=True)
            print(f"  Added Config 1 baseline from {args.legacy_config1} "
                  f"({len(legacy)} seeds, prior run, not re-trained)")

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

    def label_for(enc, sub):
        """Mark rows that came from a prior run rather than this training batch."""
        base = ENCODING_LABELS.get(enc, enc)
        if "source" in sub.columns and (sub["source"] != "this run").all():
            return base + " *"
        return base

    stats = {}
    legacy_present = False
    for enc in present:
        sub = df[df.encoding == enc]
        stats[enc] = {m: (sub[m].mean(), sub[m].std(ddof=1) if len(sub) > 1 else 0.0)
                      for m in METRICS}
        lab = label_for(enc, sub)
        if lab.endswith(" *"):
            legacy_present = True
        cells = "".join(f"{stats[enc][m][0]:>9.4f}±{stats[enc][m][1]:<7.4f}" for m in METRICS)
        lines.append(f"  {lab:<30}{cells}")

    if legacy_present:
        lines.append("")
        lines.append("  * from a prior run, not re-trained in this batch. Hyperparameters are")
        lines.append("    identical (2 layers / 30 epochs / batch 32 / lr 0.01 / 256 shots,")
        lines.append("    seeds 0-4); DNMT3B recall and precision derived from TP/FP/FN.")

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
        lines.append(f"    {label_for(enc, sub)} — seed {int(row.seed)} "
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
        lines.append(f"{label_for(enc, df[df.encoding == enc])} & {cells} \\\\")
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
        ax.set_title(f"{label_for(enc, df[df.encoding == enc])}\n"
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
