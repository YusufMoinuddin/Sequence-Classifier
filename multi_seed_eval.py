"""
multi_seed_eval.py
==================
Runs the VQC across multiple seeds and reports mean ± std for all metrics.
Matches the architecture of the original qml_classifier.py exactly:
  - Device and circuit built ONCE at module level
  - Only weights are re-seeded per run (via fresh TorchLayer each seed)

SMOKE TEST MODE: set SMOKE_TEST = True to run 2 seeds x 3 epochs quickly.
Set to False for the real run.
"""

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from src.qml_encodings import n_qubits_for, CONFIG_DESCRIPTIONS
from src.vqc_common import (
    parse_known, resolve_encoding, resolve_device, resolve_seeds,
    load_data, make_device, make_circuit, weight_shape_for, tag_path,
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SMOKE_TEST = False   # ← flip to False for real run

_args = parse_known("Multi-seed VQC evaluation")
ENCODING = resolve_encoding(_args)
DEVICE = resolve_device(_args)

SEEDS      = resolve_seeds(_args, default=([0, 1] if SMOKE_TEST else [0, 1, 2, 3, 4]))
N_QUBITS   = n_qubits_for(ENCODING)          # 4 / 4 / 8, never hardcoded
N_LAYERS   = _args.layers if _args.layers is not None else 2
N_EPOCHS   = _args.epochs if _args.epochs is not None else (3 if SMOKE_TEST else 30)
BATCH_SIZE = 32
LR         = 0.01
SHOTS      = _args.shots if _args.shots is not None else (32 if SMOKE_TEST else 256)
LIMIT_TRAIN = _args.limit_train

TRAIN_PATH = "data/deep_enzymology_qmproxy_train.csv"
VAL_PATH   = "data/deep_enzymology_qmproxy_val.csv"
TEST_PATH  = "data/deep_enzymology_qmproxy_test.csv"

# Tagged per encoding. The legacy untagged multi_seed_results.csv holds the numbers
# currently in the paper and is never overwritten.
OUTPUT_CSV = tag_path("multi_seed_results.csv", ENCODING)
OUTPUT_TXT = tag_path("multi_seed_summary.txt", ENCODING)
OUTPUT_FIG = tag_path("multi_seed_boxplots.png", ENCODING)

IS_SMOKE = SMOKE_TEST or _args.smoke
print(f"Mode: {'SMOKE TEST' if IS_SMOKE else 'FULL RUN'}")
print(f"Encoding: {ENCODING} — {CONFIG_DESCRIPTIONS[ENCODING]}")
print(f"Device: {DEVICE} | Qubits: {N_QUBITS} | Layers: {N_LAYERS}")
print(f"Seeds: {SEEDS} | Epochs: {N_EPOCHS} | Shots: {SHOTS}")
if LIMIT_TRAIN:
    print(f"Training set subsampled to {LIMIT_TRAIN} rows (smoke test)")

# ─────────────────────────────────────────────
# DATA (loaded once)
# ─────────────────────────────────────────────
print("Loading data...")
X_train, y_train = load_data(TRAIN_PATH, ENCODING, limit=LIMIT_TRAIN)
X_val,   y_val   = load_data(VAL_PATH, ENCODING)
X_test,  y_test  = load_data(TEST_PATH, ENCODING)
print(f"  X_train: {tuple(X_train.shape)} | X_val: {tuple(X_val.shape)} | X_test: {tuple(X_test.shape)}")

n_neg      = (y_train == 0).sum().item()
n_pos      = (y_train == 1).sum().item()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
print(f"  pos_weight: {pos_weight.item():.4f}")

# ─────────────────────────────────────────────
# QUANTUM DEVICE — built ONCE, shared across seeds
# ─────────────────────────────────────────────
print(f"Initializing quantum device ({N_QUBITS} qubits, {SHOTS} shots)...")
dev = make_device(DEVICE, N_QUBITS, SHOTS)
circuit_node = make_circuit(dev, n_qubits=N_QUBITS, n_layers=N_LAYERS)
weight_shape = weight_shape_for(N_LAYERS, N_QUBITS)
print("Device ready.")

# ─────────────────────────────────────────────
# PER-SEED RUN
# ─────────────────────────────────────────────
def run_seed(seed):
    print(f"\n{'='*55}\n  SEED {seed}\n{'='*55}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Fresh TorchLayer = fresh random weights from this seed
    vqc_layer = qml.qnn.TorchLayer(circuit_node, weight_shape)

    class VQCClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.vqc = vqc_layer
            self.fc  = nn.Linear(1, 1)
        def forward(self, x):
            # x already arrives at circuit width from the encoder — no slice needed.
            out = torch.stack([self.vqc(x[i]) for i in range(x.shape[0])])
            return self.fc(out.unsqueeze(1)).squeeze(1)

    model     = VQCClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=BATCH_SIZE, shuffle=True, generator=g)

    best_val_loss   = float("inf")
    best_state_dict = None
    best_epoch      = -1

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_b.size(0)
        epoch_loss /= len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
            val_preds = (torch.sigmoid(model(X_val)) >= 0.5).float()
            val_acc   = accuracy_score(y_val.numpy(), val_preds.numpy())

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_epoch      = epoch + 1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ★"
        else:
            marker = ""

        print(f"  Ep {epoch+1:02d}/{N_EPOCHS} | "
              f"Train: {epoch_loss:.4f} | Val: {val_loss:.4f} | "
              f"Acc: {val_acc:.4f}{marker}")

    torch.save({"seed": seed, "epoch": best_epoch, "encoding": ENCODING,
                "model_state_dict": best_state_dict,
                "val_loss": best_val_loss},
               f"checkpoint_{ENCODING}_seed{seed}.pt")

    model.load_state_dict(best_state_dict)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).numpy()
        preds = (probs >= 0.5).astype(float)

    y_true = y_test.numpy()
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "encoding":      ENCODING,
        "seed":          seed,
        "best_epoch":    best_epoch,
        "accuracy":      accuracy_score(y_true, preds),
        "balanced_acc":  balanced_accuracy_score(y_true, preds),
        "f1_dnmt3b":     f1_score(y_true, preds, pos_label=1, zero_division=0),
        "f1_macro":      f1_score(y_true, preds, average='macro', zero_division=0),
        "roc_auc":       roc_auc_score(y_true, probs),
        "pr_auc":        average_precision_score(y_true, probs),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
all_results = []
total_start = time.time()

for seed in SEEDS:
    all_results.append(run_seed(seed))

elapsed = time.time() - total_start
df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_CSV, index=False)

METRIC_COLS = ["accuracy", "balanced_acc", "f1_dnmt3b", "f1_macro", "roc_auc", "pr_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_acc": "Balanced Acc",
    "f1_dnmt3b": "F1 (DNMT3B)", "f1_macro": "F1 Macro",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
}

lines = []
lines.append("=" * 65)
lines.append(f"  MULTI-SEED SUMMARY ({'SMOKE TEST' if IS_SMOKE else 'FULL RUN'})")
lines.append(f"  Seeds: {SEEDS} | Epochs: {N_EPOCHS} | Shots: {SHOTS}")
lines.append(f"  Runtime: {elapsed/60:.1f} min")
lines.append("=" * 65)
lines.append(f"\n  {'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
lines.append(f"  {'-'*56}")
for col in METRIC_COLS:
    v = df[col].values
    lines.append(f"  {METRIC_LABELS[col]:<20} {v.mean():>8.4f} {v.std():>8.4f} "
                 f"{v.min():>8.4f} {v.max():>8.4f}")
lines.append("=" * 65)

summary = "\n".join(lines)
print("\n" + summary)
with open(OUTPUT_TXT, "w") as f:
    f.write(summary)

# Boxplots
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for idx, col in enumerate(METRIC_COLS):
    ax = axes.flatten()[idx]
    vals = df[col].values
    ax.boxplot(vals, patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.6),
               medianprops=dict(color='navy', linewidth=2))
    ax.scatter([1]*len(vals), vals, color='navy', zorder=5, s=40)
    for i, (v, s) in enumerate(zip(vals, SEEDS)):
        ax.annotate(f"s{s}", (1, v), xytext=(12, 0),
                    textcoords="offset points", fontsize=8, color='navy')
    ax.set_title(METRIC_LABELS[col], fontweight='bold')
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.05, 0.05, f"μ={vals.mean():.3f}\nσ={vals.std():.3f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle(f"VQC Multi-Seed ({'Smoke Test' if SMOKE_TEST else 'Full Run'})\n"
             f"{len(SEEDS)} seeds | {N_EPOCHS} epochs | {SHOTS} shots",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=150)
plt.close()
print(f"\nSaved: {OUTPUT_CSV}, {OUTPUT_TXT}, {OUTPUT_FIG}")
print(f"Total runtime: {elapsed/60:.1f} min")
