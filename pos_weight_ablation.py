"""
posweight_ablation.py
=====================
Trains VQC with and without pos_weight across multiple seeds.
Device built ONCE at module level — matches original qml_classifier.py architecture.

SMOKE_TEST = True  → 2 seeds, 3 epochs, 32 shots (fast verification)
SMOKE_TEST = False → 5 seeds, 30 epochs, 256 shots (real run)
"""

import random
import time
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
    recall_score, precision_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SMOKE_TEST = False   # ← flip to False for real run

SEEDS      = [0, 1] if SMOKE_TEST else [0, 1, 2, 3, 4]
N_QUBITS   = 4
N_LAYERS   = 2
N_EPOCHS   = 3   if SMOKE_TEST else 30
BATCH_SIZE = 32
LR         = 0.01
SHOTS      = 32  if SMOKE_TEST else 256

TRAIN_PATH = "data/deep_enzymology_qmproxy_train.csv"
VAL_PATH   = "data/deep_enzymology_qmproxy_val.csv"
TEST_PATH  = "data/deep_enzymology_qmproxy_test.csv"

OUTPUT_CSV = "posweight_ablation_results.csv"
OUTPUT_TXT = "posweight_ablation_summary.txt"
OUTPUT_FIG = "posweight_ablation_comparison.png"

print(f"Mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL RUN'}")
print(f"Seeds: {SEEDS} | Epochs: {N_EPOCHS} | Shots: {SHOTS}")

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
NUC_MAP = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

def encode_sequence(seq):
    return np.array([NUC_MAP[c] for c in seq], dtype=np.float32) * (np.pi / 3.0)

def load_data(path):
    df = pd.read_csv(path)
    X = np.stack([encode_sequence(s) for s in df['sequence']])
    y = df['label'].values.astype(np.float32)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

print("Loading data...")
X_train, y_train = load_data(TRAIN_PATH)
X_val,   y_val   = load_data(VAL_PATH)
X_test,  y_test  = load_data(TEST_PATH)

n_neg      = (y_train == 0).sum().item()
n_pos      = (y_train == 1).sum().item()
POS_WEIGHT = torch.tensor([n_neg / n_pos], dtype=torch.float32)
NO_WEIGHT  = torch.tensor([1.0], dtype=torch.float32)
print(f"  pos_weight: {POS_WEIGHT.item():.4f}")

# ─────────────────────────────────────────────
# QUANTUM DEVICE — built ONCE
# ─────────────────────────────────────────────
print(f"Initializing quantum device ({N_QUBITS} qubits, {SHOTS} shots)...")
dev = qml.device("qiskit.aer", wires=N_QUBITS, shots=SHOTS)

def quantum_circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RX(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    return qml.expval(qml.PauliZ(0))

circuit_node = qml.QNode(quantum_circuit, dev, interface="torch")
weight_shape = {"weights": (N_LAYERS, N_QUBITS, 3)}
print("Device ready.")

# ─────────────────────────────────────────────
# SINGLE RUN
# ─────────────────────────────────────────────
def run_single(seed, use_pos_weight):
    condition = "weighted" if use_pos_weight else "unweighted"
    pw        = POS_WEIGHT if use_pos_weight else NO_WEIGHT
    print(f"\n  {'─'*50}")
    print(f"  Seed {seed} | {condition} (pos_weight={pw.item():.2f})")
    print(f"  {'─'*50}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vqc_layer = qml.qnn.TorchLayer(circuit_node, weight_shape)

    class VQCClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.vqc = vqc_layer
            self.fc  = nn.Linear(1, 1)
        def forward(self, x):
            x_q = x[:, :N_QUBITS]
            out = torch.stack([self.vqc(x_q[i]) for i in range(x_q.shape[0])])
            return self.fc(out.unsqueeze(1)).squeeze(1)

    model     = VQCClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

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
            val_loss  = criterion(model(X_val), y_val).item()
            val_preds = (torch.sigmoid(model(X_val)) >= 0.5).float()
            val_acc   = accuracy_score(y_val.numpy(), val_preds.numpy())

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_epoch      = epoch + 1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ★"
        else:
            marker = ""

        print(f"    Ep {epoch+1:02d}/{N_EPOCHS} | "
              f"Loss: {epoch_loss:.4f} | Val: {val_loss:.4f} | "
              f"Acc: {val_acc:.4f}{marker}")

    ckpt = f"checkpoint_{condition}_seed{seed}.pt"
    torch.save({"seed": seed, "condition": condition, "epoch": best_epoch,
                "val_loss": best_val_loss, "model_state_dict": best_state_dict}, ckpt)

    model.load_state_dict(best_state_dict)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).numpy()
        preds = (probs >= 0.5).astype(float)

    y_true = y_test.numpy()
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "condition":    condition,
        "seed":         seed,
        "accuracy":     accuracy_score(y_true, preds),
        "balanced_acc": balanced_accuracy_score(y_true, preds),
        "precision_3b": precision_score(y_true, preds, pos_label=1, zero_division=0),
        "recall_3b":    recall_score(y_true, preds, pos_label=1, zero_division=0),
        "f1_dnmt3b":    f1_score(y_true, preds, pos_label=1, zero_division=0),
        "f1_macro":     f1_score(y_true, preds, average='macro', zero_division=0),
        "roc_auc":      roc_auc_score(y_true, probs),
        "pr_auc":       average_precision_score(y_true, probs),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
all_results = []
total_start = time.time()

for use_pw, label in [(True, "WEIGHTED"), (False, "UNWEIGHTED")]:
    print(f"\n{'#'*55}\n#  {label}\n{'#'*55}")
    for seed in SEEDS:
        all_results.append(run_single(seed, use_pw))

elapsed = time.time() - total_start
df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_CSV, index=False)

METRIC_COLS = ["accuracy", "balanced_acc", "recall_3b", "f1_dnmt3b", "roc_auc", "pr_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_acc": "Balanced Acc",
    "recall_3b": "Recall (DNMT3B)", "f1_dnmt3b": "F1 (DNMT3B)",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
}

y_true_np         = y_test.numpy()
majority_baseline = (y_true_np == 0).mean()
pr_baseline       = (y_true_np == 1).mean()

lines = []
lines.append("=" * 72)
lines.append(f"  POS_WEIGHT ABLATION ({'SMOKE TEST' if SMOKE_TEST else 'FULL RUN'})")
lines.append(f"  Seeds: {SEEDS} | Epochs: {N_EPOCHS} | Shots: {SHOTS}")
lines.append(f"  Runtime: {elapsed/60:.1f} min")
lines.append("=" * 72)
lines.append(f"\n  {'Metric':<22} {'Weighted (μ±σ)':>18} {'Unweighted (μ±σ)':>18} {'Δ':>8}")
lines.append(f"  {'-'*68}")

for col in METRIC_COLS:
    w = df[df.condition=="weighted"][col].values
    u = df[df.condition=="unweighted"][col].values
    lines.append(f"  {METRIC_LABELS[col]:<22} "
                 f"{w.mean():.3f}±{w.std():.3f}{'':>6} "
                 f"{u.mean():.3f}±{u.std():.3f}{'':>6} "
                 f"{w.mean()-u.mean():>+8.3f}")

lines.append(f"\n  Baselines: Accuracy={majority_baseline:.3f} | "
             f"BalAcc=0.500 | ROC-AUC=0.500 | PR-AUC={pr_baseline:.3f}")
lines.append("=" * 72)

summary = "\n".join(lines)
print("\n" + summary)
with open(OUTPUT_TXT, "w") as f:
    f.write(summary)

# Bar chart
PLOT_METRICS = ["accuracy", "balanced_acc", "recall_3b", "f1_dnmt3b", "roc_auc", "pr_auc"]
w_means = [df[df.condition=="weighted"][m].mean()   for m in PLOT_METRICS]
u_means = [df[df.condition=="unweighted"][m].mean() for m in PLOT_METRICS]
w_stds  = [df[df.condition=="weighted"][m].std()    for m in PLOT_METRICS]
u_stds  = [df[df.condition=="unweighted"][m].std()  for m in PLOT_METRICS]
x = np.arange(len(PLOT_METRICS))
width = 0.32

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width/2, w_means, width, yerr=w_stds, capsize=4,
       color='steelblue', alpha=0.85, label=f'Weighted ({POS_WEIGHT.item():.2f})')
ax.bar(x + width/2, u_means, width, yerr=u_stds, capsize=4,
       color='coral', alpha=0.85, label='Unweighted (1.00)')
ax.set_xticks(x)
ax.set_xticklabels([METRIC_LABELS[m] for m in PLOT_METRICS], rotation=20, ha='right')
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title(f"pos_weight Ablation ({'Smoke Test' if SMOKE_TEST else 'Full Run'})\n"
             f"Mean ± std across {len(SEEDS)} seeds")
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=150)
plt.close()
print(f"\nSaved: {OUTPUT_CSV}, {OUTPUT_TXT}, {OUTPUT_FIG}")
print(f"Total runtime: {elapsed/60:.1f} min")