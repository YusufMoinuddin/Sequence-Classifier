"""
sanity_checks.py
================
Four sanity checks: majority baseline, shuffled labels,
random features, feature ablation.
Device built ONCE at module level — matches original architecture.

SMOKE_TEST = True  → 3 epochs, 32 shots
SMOKE_TEST = False → 30 epochs, 256 shots
"""

import random
import os
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

SEED       = 0
N_QUBITS   = 4
N_LAYERS   = 2
N_EPOCHS   = 3   if SMOKE_TEST else 30
BATCH_SIZE = 32
LR         = 0.01
SHOTS      = 32  if SMOKE_TEST else 256

TRAIN_PATH      = "data/deep_enzymology_qmproxy_train.csv"
VAL_PATH        = "data/deep_enzymology_qmproxy_val.csv"
TEST_PATH       = "data/deep_enzymology_qmproxy_test.csv"
CHECKPOINT_PATH = "best_vqc_checkpoint.pt"

OUTPUT_TXT = "sanity_checks_summary.txt"
OUTPUT_FIG = "sanity_feature_ablation.png"

print(f"Mode: {'SMOKE TEST' if SMOKE_TEST else 'FULL RUN'}")
print(f"Epochs: {N_EPOCHS} | Shots: {SHOTS}")

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
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

y_true         = y_test.numpy()
majority_label = 0
pr_baseline    = (y_true == 1).mean()
maj_baseline   = (y_true == 0).mean()

# ─────────────────────────────────────────────
# QUANTUM DEVICE — built ONCE
# ─────────────────────────────────────────────
print(f"Initializing quantum device ({N_QUBITS} qubits, {SHOTS} shots)...")
dev = qml.device("lightning.gpu", wires=N_QUBITS, shots=SHOTS)

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
# HELPERS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":     accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "recall_3b":    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_dnmt3b":    f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc":      roc_auc_score(y_true, y_prob) if len(np.unique(y_prob)) > 1 else 0.5,
        "pr_auc":       average_precision_score(y_true, y_prob),
    }

def build_and_train(X_tr, y_tr, X_v, y_v, X_te, y_te, pw, seed, tag):
    """Build fresh model, train, return best-checkpoint metrics."""
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
    loader = DataLoader(TensorDataset(X_tr, y_tr),
                        batch_size=BATCH_SIZE, shuffle=True, generator=g)

    best_val_loss   = float("inf")
    best_state_dict = None

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
            val_loss  = criterion(model(X_v), y_v).item()
            val_preds = (torch.sigmoid(model(X_v)) >= 0.5).float()
            val_acc   = accuracy_score(y_v.numpy(), val_preds.numpy())

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ★"
        else:
            marker = ""

        print(f"    [{tag}] Ep {epoch+1:02d}/{N_EPOCHS} | "
              f"Loss: {epoch_loss:.4f} | Val: {val_loss:.4f} | "
              f"Acc: {val_acc:.4f}{marker}")

    model.load_state_dict(best_state_dict)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te)).numpy()
        preds = (probs >= 0.5).astype(float)

    return compute_metrics(y_te.numpy(), preds, probs), model

# ─────────────────────────────────────────────
# CHECK 1: MAJORITY BASELINE
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("CHECK 1: MAJORITY-CLASS BASELINE")
print("="*55)
maj_preds = np.zeros_like(y_true)
maj_probs = np.zeros_like(y_true, dtype=float)
maj_metrics = compute_metrics(y_true, maj_preds, maj_probs)
print(f"  Accuracy: {maj_metrics['accuracy']:.4f} | "
      f"BalAcc: {maj_metrics['balanced_acc']:.4f} | "
      f"Recall(3B): {maj_metrics['recall_3b']:.4f} | "
      f"F1(3B): {maj_metrics['f1_dnmt3b']:.4f}")

results = {"Majority Baseline": maj_metrics}

# ─────────────────────────────────────────────
# CHECK 2: SHUFFLED LABELS
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("CHECK 2: SHUFFLED-LABEL TEST")
print("="*55)
rng = np.random.default_rng(SEED)
shuffled_y = torch.tensor(rng.permutation(y_train.numpy()), dtype=torch.float32)
n_neg_sh   = (shuffled_y == 0).sum().item()
n_pos_sh   = (shuffled_y == 1).sum().item()
pw_sh      = torch.tensor([n_neg_sh / n_pos_sh], dtype=torch.float32)

shuf_metrics, _ = build_and_train(
    X_train, shuffled_y, X_val, y_val, X_test, y_test,
    pw=pw_sh, seed=SEED, tag="SHUFFLED"
)
results["Shuffled Labels"] = shuf_metrics
print(f"  Accuracy: {shuf_metrics['accuracy']:.4f} | "
      f"BalAcc: {shuf_metrics['balanced_acc']:.4f} | "
      f"Recall(3B): {shuf_metrics['recall_3b']:.4f} | "
      f"F1(3B): {shuf_metrics['f1_dnmt3b']:.4f}")

# ─────────────────────────────────────────────
# CHECK 3: RANDOM FEATURES
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("CHECK 3: RANDOM-FEATURE BASELINE")
print("="*55)
torch.manual_seed(SEED)
X_tr_rand  = torch.rand_like(X_train) * np.pi
X_val_rand = torch.rand_like(X_val)   * np.pi
X_te_rand  = torch.rand_like(X_test)  * np.pi

rand_metrics, _ = build_and_train(
    X_tr_rand, y_train, X_val_rand, y_val, X_te_rand, y_test,
    pw=pos_weight, seed=SEED, tag="RAND-FEAT"
)
results["Random Features"] = rand_metrics
print(f"  Accuracy: {rand_metrics['accuracy']:.4f} | "
      f"BalAcc: {rand_metrics['balanced_acc']:.4f} | "
      f"Recall(3B): {rand_metrics['recall_3b']:.4f} | "
      f"F1(3B): {rand_metrics['f1_dnmt3b']:.4f}")

# ─────────────────────────────────────────────
# CHECK 4: FEATURE ABLATION (inference-time)
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("CHECK 4: FEATURE ABLATION")
print("="*55)

ablation_results = {}

if not os.path.exists(CHECKPOINT_PATH):
    print(f"  [SKIP] {CHECKPOINT_PATH} not found — run qml_classifier.py first.")
else:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    vqc_abl = qml.qnn.TorchLayer(circuit_node, weight_shape)

    class VQCAbl(nn.Module):
        def __init__(self):
            super().__init__()
            self.vqc = vqc_abl
            self.fc  = nn.Linear(1, 1)
        def forward(self, x):
            x_q = x[:, :N_QUBITS]
            out = torch.stack([self.vqc(x_q[i]) for i in range(x_q.shape[0])])
            return self.fc(out.unsqueeze(1)).squeeze(1)

    model_abl = VQCAbl()
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model_abl.load_state_dict(ckpt["model_state_dict"])
    model_abl.eval()
    print(f"  Loaded checkpoint: epoch {ckpt['epoch']} | val_loss={ckpt['val_loss']:.4f}")

    with torch.no_grad():
        base_probs = torch.sigmoid(model_abl(X_test)).numpy()
        base_preds = (base_probs >= 0.5).astype(float)
    base_m = compute_metrics(y_true, base_preds, base_probs)
    ablation_results["Baseline"] = base_m
    print(f"\n  Baseline — Acc: {base_m['accuracy']:.4f} | "
          f"F1(3B): {base_m['f1_dnmt3b']:.4f} | AUC: {base_m['roc_auc']:.4f}")

    pos_means = X_train[:, :N_QUBITS].mean(dim=0)
    pos_labels = [f"Pos{i}(N{i+1})" for i in range(N_QUBITS)]

    print()
    for pos in range(N_QUBITS):
        X_masked = X_test.clone()
        X_masked[:, pos] = pos_means[pos]
        with torch.no_grad():
            abl_probs = torch.sigmoid(model_abl(X_masked)).numpy()
            abl_preds = (abl_probs >= 0.5).astype(float)
        abl_m = compute_metrics(y_true, abl_preds, abl_probs)
        ablation_results[pos_labels[pos]] = abl_m
        df1 = abl_m['f1_dnmt3b'] - base_m['f1_dnmt3b']
        da  = abl_m['roc_auc']   - base_m['roc_auc']
        print(f"  Mask {pos_labels[pos]:<12} | "
              f"F1(3B): {abl_m['f1_dnmt3b']:.4f} (Δ{df1:+.4f}) | "
              f"AUC: {abl_m['roc_auc']:.4f} (Δ{da:+.4f})")

    # Ablation figure
    pos_names  = pos_labels
    delta_f1s  = [ablation_results[p]['f1_dnmt3b'] - base_m['f1_dnmt3b'] for p in pos_names]
    delta_aucs = [ablation_results[p]['roc_auc']   - base_m['roc_auc']   for p in pos_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors1 = ['firebrick' if d < 0 else 'steelblue' for d in delta_f1s]
    colors2 = ['firebrick' if d < 0 else 'steelblue' for d in delta_aucs]
    ax1.bar(pos_names, delta_f1s, color=colors1, alpha=0.85)
    ax1.axhline(0, color='black', lw=1)
    ax1.set_title("ΔF1 (DNMT3B) when position masked", fontweight='bold')
    ax1.set_ylabel("Change vs baseline")
    ax1.grid(axis='y', alpha=0.3)
    ax2.bar(pos_names, delta_aucs, color=colors2, alpha=0.85)
    ax2.axhline(0, color='black', lw=1)
    ax2.set_title("ΔROC-AUC when position masked", fontweight='bold')
    ax2.set_ylabel("Change vs baseline")
    ax2.grid(axis='y', alpha=0.3)
    plt.suptitle("Feature Ablation — Red=position matters, Blue=no effect",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150)
    plt.close()
    print(f"\n  Saved: {OUTPUT_FIG}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
METRIC_COLS = ["accuracy", "balanced_acc", "recall_3b", "f1_dnmt3b", "roc_auc", "pr_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_acc": "Balanced Acc",
    "recall_3b": "Recall (DNMT3B)", "f1_dnmt3b": "F1 (DNMT3B)",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
}

lines = []
lines.append("=" * 70)
lines.append(f"  SANITY CHECKS SUMMARY ({'SMOKE TEST' if SMOKE_TEST else 'FULL RUN'})")
lines.append("=" * 70)
lines.append(f"\n  {'Check':<22} " +
             " ".join(f"{METRIC_LABELS[c]:>13}" for c in METRIC_COLS))
lines.append(f"  {'-'*68}")
for name, m in results.items():
    lines.append(f"  {name:<22} " +
                 " ".join(f"{m[c]:>13.4f}" for c in METRIC_COLS))
lines.append(f"\n  Baselines: Accuracy={maj_baseline:.3f} | BalAcc=0.500 | "
             f"ROC-AUC=0.500 | PR-AUC={pr_baseline:.3f}")
lines.append("=" * 70)

summary = "\n".join(lines)
print("\n" + summary)
with open(OUTPUT_TXT, "w") as f:
    f.write(summary)
print(f"Saved: {OUTPUT_TXT}")
