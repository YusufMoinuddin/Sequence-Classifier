"""
benchmark_classical.py
======================
Classical baselines (SVM, Random Forest, 1D CNN) for the QML vs. classical
DNMT3A/DNMT3B comparison, evaluated exactly the way the VQC is evaluated in
evaluate_metrics.py / multi_seed_eval.py.

Protocol (matched to the VQC):
  - Same frozen splits, used as-is. Never re-split, never regenerated.
  - Features are ONLY the 'sequence' column (one-hot 8-mer). All QM-proxy
    columns and the label-derived meth_A/meth_B/total/label_str/key columns
    are dropped before encoding.
  - Class weighting for all three models:
      SVM / RF -> class_weight="balanced"
      CNN      -> BCEWithLogitsLoss(pos_weight = n_neg / n_pos)
  - Fixed 0.5 decision threshold, matching evaluate_metrics.py.
  - Seeds [0, 1, 2, 3, 4], reported as mean +/- std.
  - The VQC row is READ from the existing multi_seed_results.csv. The VQC is
    never re-run here.

Run from the project root with the venv active:
  python benchmark_classical.py

Outputs:
  classical_baseline_results.csv     one row per (model, seed)
  classical_baseline_summary.txt     paper table (fixed-width + LaTeX)
  classical_confusion_matrices.png   test-set confusion matrices
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.enzyme_common import load_seq_label, flatten_onehot, full_binary_metrics
from models.svm_classifier import build_svm
from models.rf_classifier import build_rf
from models.cnn_classifier import SeqCNN

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SMOKE_TEST = False   # ← flip to True for a fast plumbing check

SEEDS          = [0] if SMOKE_TEST else [0, 1, 2, 3, 4]
SMOKE_N_TRAIN  = 300                        # subsample size, smoke only
CNN_EPOCHS     = 3 if SMOKE_TEST else 200
CNN_PATIENCE   = 2 if SMOKE_TEST else 20    # early stopping on val loss
CNN_BATCH      = 64
CNN_LR         = 1e-3
SEQ_LEN        = 8
THRESHOLD      = 0.5

TRAIN_PATH = "data/deep_enzymology_qmproxy_train.csv"
VAL_PATH   = "data/deep_enzymology_qmproxy_val.csv"
TEST_PATH  = "data/deep_enzymology_qmproxy_test.csv"

# Prefer the encoding-tagged Config 1 results if a re-run has produced them; otherwise
# fall back to the legacy untagged file, which holds the numbers currently in the paper.
VQC_RESULTS_CANDIDATES = ["multi_seed_results_config1.csv", "multi_seed_results.csv"]
VQC_RESULTS_CSV = next(
    (n for n in VQC_RESULTS_CANDIDATES if (ROOT / n).exists()),
    VQC_RESULTS_CANDIDATES[-1],
)

OUTPUT_CSV = "classical_baseline_results.csv"
OUTPUT_TXT = "classical_baseline_summary.txt"
OUTPUT_FIG = "classical_confusion_matrices.png"

# Expected (rows, n_DNMT3A, n_DNMT3B) per split. The run aborts on mismatch so
# a drifted path can never silently train on the wrong file.
EXPECTED_COUNTS = {
    TRAIN_PATH: (1652, 1394, 258),
    VAL_PATH:   (206, 174, 32),
    TEST_PATH:  (207, 174, 33),
}

MODEL_ORDER = ["SVM (RBF)", "Random Forest", "1D CNN"]

print("=" * 70)
print(f"  CLASSICAL BASELINE BENCHMARK ({'SMOKE TEST' if SMOKE_TEST else 'FULL RUN'})")
print("=" * 70)
print(f"  Seeds: {SEEDS} | CNN epochs: {CNN_EPOCHS} (patience {CNN_PATIENCE})")
print(f"  Decision threshold: {THRESHOLD}")


# ─────────────────────────────────────────────
# STEP 1: LOAD + VERIFY SPLITS  (first action)
# ─────────────────────────────────────────────
print("\n" + "-" * 70)
print("  STEP 1 — Loading splits and verifying row / class counts")
print("-" * 70)

splits = {}
for path, (exp_rows, exp_neg, exp_pos) in EXPECTED_COUNTS.items():
    x, y, n_neg, n_pos = load_seq_label(path, expected_len=SEQ_LEN)
    n_rows = len(y)
    ok = (n_rows, n_neg, n_pos) == (exp_rows, exp_neg, exp_pos)
    status = "OK" if ok else "MISMATCH"
    print(f"  [{status:8}] {path}")
    print(f"             rows={n_rows:<6} DNMT3A(0)={n_neg:<6} DNMT3B(1)={n_pos:<6}"
          f"  (expected {exp_rows} / {exp_neg} / {exp_pos})")
    if not ok:
        raise SystemExit(
            f"\nABORT: {path} does not match the confirmed dataset.\n"
            f"  got      rows={n_rows} DNMT3A={n_neg} DNMT3B={n_pos}\n"
            f"  expected rows={exp_rows} DNMT3A={exp_neg} DNMT3B={exp_pos}\n"
            "Check that the data paths have not drifted."
        )
    splits[path] = (x, y)

X_train_oh, y_train = splits[TRAIN_PATH]
X_val_oh,   y_val   = splits[VAL_PATH]
X_test_oh,  y_test  = splits[TEST_PATH]

# Leakage assertion: 4 bases x 8 positions = 32 features and nothing else.
X_train_flat = flatten_onehot(X_train_oh)
X_val_flat   = flatten_onehot(X_val_oh)
X_test_flat  = flatten_onehot(X_test_oh)
assert X_train_flat.shape[1] == 4 * SEQ_LEN, (
    f"Feature matrix has {X_train_flat.shape[1]} columns, expected {4 * SEQ_LEN}. "
    "A non-sequence column leaked into the features."
)
print(f"\n  Feature matrix: {X_train_flat.shape[1]} columns "
      f"(4 bases x {SEQ_LEN} positions) — no QM-proxy or meth_* columns present.")

pos_weight_value = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  Class weight for DNMT3B (positive class): {pos_weight_value:.4f}x")

# Majority-class reference on the test set, for context in the table.
majority_metrics = full_binary_metrics(y_test, np.zeros(len(y_test)), threshold=THRESHOLD)


# ─────────────────────────────────────────────
# STEP 2: PER-MODEL TRAINING
# ─────────────────────────────────────────────

def subsample(x, y, n, seed):
    """Stratified-ish subsample for the smoke test only."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))[:n]
    return x[idx], y[idx]


def run_sklearn(build_fn, seed):
    """Fit an sklearn estimator and return (val_metrics, test_metrics)."""
    xt, yt = (subsample(X_train_flat, y_train, SMOKE_N_TRAIN, seed)
              if SMOKE_TEST else (X_train_flat, y_train))
    clf = build_fn(random_state=seed)
    clf.fit(xt, yt)
    val_prob = clf.predict_proba(X_val_flat)[:, 1]
    test_prob = clf.predict_proba(X_test_flat)[:, 1]
    return (full_binary_metrics(y_val, val_prob, THRESHOLD),
            full_binary_metrics(y_test, test_prob, THRESHOLD),
            None)


def run_cnn(seed):
    """
    Train SeqCNN with a pos_weight-ed loss, early stopping on val loss,
    restoring the best-val-loss weights (same rule the VQC uses).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    xt, yt = (subsample(X_train_oh, y_train, SMOKE_N_TRAIN, seed)
              if SMOKE_TEST else (X_train_oh, y_train))

    # pos_weight is computed from whatever the model actually trains on.
    n_neg_t = float((yt == 0).sum())
    n_pos_t = float((yt == 1).sum())
    pw = torch.tensor([n_neg_t / n_pos_t], dtype=torch.float32)

    xt_t = torch.tensor(xt, dtype=torch.float32)
    yt_t = torch.tensor(yt, dtype=torch.float32)
    xv_t = torch.tensor(X_val_oh, dtype=torch.float32)
    yv_t = torch.tensor(y_val, dtype=torch.float32)
    xs_t = torch.tensor(X_test_oh, dtype=torch.float32)

    loader = DataLoader(TensorDataset(xt_t, yt_t), batch_size=CNN_BATCH, shuffle=True)

    model = SeqCNN(in_channels=4, hidden_channels=16, seq_len=SEQ_LEN)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=CNN_LR)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(1, CNN_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv_t), yv_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= CNN_PATIENCE:
                print(f"      early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(xv_t)).numpy()
        test_prob = torch.sigmoid(model(xs_t)).numpy()

    return (full_binary_metrics(y_val, val_prob, THRESHOLD),
            full_binary_metrics(y_test, test_prob, THRESHOLD),
            best_epoch)


RUNNERS = {
    "SVM (RBF)":     lambda seed: run_sklearn(build_svm, seed),
    "Random Forest": lambda seed: run_sklearn(build_rf, seed),
    "1D CNN":        run_cnn,
}

print("\n" + "-" * 70)
print("  STEP 2 — Training")
print("-" * 70)

rows = []
for model_name in MODEL_ORDER:
    print(f"\n  {model_name}")
    for seed in SEEDS:
        val_m, test_m, best_epoch = RUNNERS[model_name](seed)
        row = {"model": model_name, "seed": seed, "best_epoch": best_epoch}
        row.update({f"val_{k}": v for k, v in val_m.items()})
        row.update(test_m)   # test metrics unprefixed — these are the reported ones
        rows.append(row)
        print(f"    seed {seed} | bal_acc={test_m['balanced_acc']:.4f} "
              f"| roc_auc={test_m['roc_auc']:.4f} | pr_auc={test_m['pr_auc']:.4f} "
              f"| 3B recall={test_m['recall_dnmt3b']:.4f} | 3B F1={test_m['f1_dnmt3b']:.4f}")
        print(f"             CM: TN={test_m['TN']} FP={test_m['FP']} "
              f"FN={test_m['FN']} TP={test_m['TP']}")

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_CSV, index=False)
print(f"\n  Wrote per-seed results -> {OUTPUT_CSV}")


# ─────────────────────────────────────────────
# STEP 3: VQC ROW (read from file, not re-run)
# ─────────────────────────────────────────────
vqc_stats = None
vqc_path = ROOT / VQC_RESULTS_CSV
if vqc_path.exists():
    vqc = pd.read_csv(vqc_path)
    # DNMT3B recall isn't a column there; derive it per seed from TP/FN.
    vqc_recall = vqc["TP"] / (vqc["TP"] + vqc["FN"]).replace(0, np.nan)
    vqc_stats = {
        "balanced_acc": (vqc["balanced_acc"].mean(), vqc["balanced_acc"].std(ddof=1)),
        "roc_auc":      (vqc["roc_auc"].mean(), vqc["roc_auc"].std(ddof=1)),
        "pr_auc":       (vqc["pr_auc"].mean(), vqc["pr_auc"].std(ddof=1)),
        "recall_dnmt3b": (vqc_recall.mean(), vqc_recall.std(ddof=1)),
        "f1_dnmt3b":    (vqc["f1_dnmt3b"].mean(), vqc["f1_dnmt3b"].std(ddof=1)),
        "n_seeds": len(vqc),
    }
    print(f"  Read VQC reference row from {VQC_RESULTS_CSV} ({len(vqc)} seeds, not re-run)")
else:
    print(f"  NOTE: {VQC_RESULTS_CSV} not found — VQC row omitted from the table.")


# ─────────────────────────────────────────────
# STEP 4: SUMMARY TABLE
# ─────────────────────────────────────────────
METRIC_COLS = ["balanced_acc", "roc_auc", "pr_auc", "recall_dnmt3b", "f1_dnmt3b"]
METRIC_LABELS = ["Balanced Acc", "ROC-AUC", "PR-AUC", "DNMT3B Recall", "DNMT3B F1"]

table = []
for model_name in MODEL_ORDER:
    sub = results[results.model == model_name]
    stats = {m: (sub[m].mean(), sub[m].std(ddof=1) if len(sub) > 1 else 0.0)
             for m in METRIC_COLS}
    table.append((model_name, stats, len(sub)))

if vqc_stats is not None:
    table.append(("VQC (quantum)",
                  {m: vqc_stats[m] for m in METRIC_COLS},
                  vqc_stats["n_seeds"]))

table.append(("Majority baseline",
              {m: (majority_metrics[m], 0.0) for m in METRIC_COLS},
              1))

lines = []
lines.append("=" * 94)
lines.append(f"  CLASSICAL BASELINES vs VQC — TEST SET ({'SMOKE TEST' if SMOKE_TEST else 'FULL RUN'})")
lines.append(f"  Seeds: {SEEDS} | threshold={THRESHOLD} | mean +/- std over seeds")
lines.append(f"  Data: {EXPECTED_COUNTS[TRAIN_PATH][0]} train / "
             f"{EXPECTED_COUNTS[VAL_PATH][0]} val / {EXPECTED_COUNTS[TEST_PATH][0]} test"
             f" | imbalance 5.39:1 | features: sequence only")
lines.append("=" * 94)
lines.append("")
header = f"  {'Model':<20}" + "".join(f"{lab:>15}" for lab in METRIC_LABELS)
lines.append(header)
lines.append("  " + "-" * (20 + 15 * len(METRIC_LABELS)))
for model_name, stats, n in table:
    cells = "".join(f"{stats[m][0]:>8.4f}±{stats[m][1]:<6.4f}" for m in METRIC_COLS)
    lines.append(f"  {model_name:<20}{cells}")
lines.append("")
lines.append("  Positive class = DNMT3B. Class weighting: SVM/RF class_weight='balanced';")
lines.append(f"  CNN BCEWithLogitsLoss(pos_weight={pos_weight_value:.4f}); VQC pos_weight identical.")
lines.append("  NOTE: the VQC consumes only the first 4 of 8 nucleotides (N_QUBITS=4);")
lines.append("  the classical baselines see all 8 positions.")
lines.append("=" * 94)
lines.append("")

# LaTeX version for direct paste into the paper.
lines.append("% ---- LaTeX (paste into the paper) ----")
lines.append("\\begin{tabular}{lccccc}")
lines.append("\\hline")
lines.append("Model & Balanced Acc & ROC-AUC & PR-AUC & DNMT3B Recall & DNMT3B F1 \\\\")
lines.append("\\hline")
for model_name, stats, n in table:
    cells = " & ".join(f"${stats[m][0]:.3f} \\pm {stats[m][1]:.3f}$" for m in METRIC_COLS)
    lines.append(f"{model_name} & {cells} \\\\")
lines.append("\\hline")
lines.append("\\end{tabular}")

summary_text = "\n".join(lines)
with open(OUTPUT_TXT, "w") as f:
    f.write(summary_text + "\n")

print("\n" + summary_text)
print(f"\n  Wrote summary -> {OUTPUT_TXT}")


# ─────────────────────────────────────────────
# STEP 5: CONFUSION MATRIX FIGURE
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(5 * len(MODEL_ORDER), 4.4))
if len(MODEL_ORDER) == 1:
    axes = [axes]

for ax, model_name in zip(axes, MODEL_ORDER):
    sub = results[results.model == model_name]
    # Mean counts across seeds; identical to the single CM when len(SEEDS)==1.
    cm = np.array([[sub["TN"].mean(), sub["FP"].mean()],
                   [sub["FN"].mean(), sub["TP"].mean()]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.1f}" if len(sub) > 1 else f"{int(cm[i, j])}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    bal = sub["balanced_acc"].mean()
    ax.set_title(f"{model_name}\nBalanced Acc = {bal:.3f}", fontsize=11)
    ax.set_xticks([0, 1], ["Pred 3A", "Pred 3B"])
    ax.set_yticks([0, 1], ["True 3A", "True 3B"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

fig.suptitle(
    f"Classical baselines — test-set confusion matrices "
    f"({'mean over ' + str(len(SEEDS)) + ' seeds' if len(SEEDS) > 1 else 'seed ' + str(SEEDS[0])})",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(OUTPUT_FIG, dpi=150)
print(f"  Wrote figure  -> {OUTPUT_FIG}")

print("\nDone.")
