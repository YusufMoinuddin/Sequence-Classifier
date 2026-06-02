"""
evaluate_metrics.py
===================
Loads the best VQC checkpoint and produces the full metric suite
Murat requested:
  - Accuracy, Balanced Accuracy
  - Precision, Recall, F1-score (per-class and macro/weighted)
  - Confusion Matrix
  - ROC-AUC
  - PR-AUC

Also saves two figures:
  - metrics_confusion_matrix.png
  - metrics_roc_pr_curves.png

Run from the project root with the venv active:
  python evaluate_metrics.py

Requires: best_vqc_checkpoint.pt saved by the updated training script.
If you haven't retrained yet, set LOAD_CHECKPOINT = False to evaluate
whatever weights are currently in memory (you must import and run this
from inside qml_classifier.py in that case).
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,   # this is PR-AUC
    roc_curve,
    precision_recall_curve,
    classification_report,
)

# ─────────────────────────────────────────────
# CONFIGURATION — must match qml_classifier.py
# ─────────────────────────────────────────────
N_QUBITS        = 4
N_LAYERS        = 2
SHOTS           = 256
TEST_PATH       = "data/deep_enzymology_qmproxy_test.csv"
CHECKPOINT_PATH = "best_vqc_checkpoint.pt"

# ─────────────────────────────────────────────
# STEP 1: REBUILD MODEL (identical to training)
# ─────────────────────────────────────────────
NUC_MAP = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

def encode_sequence(seq):
    ints = np.array([NUC_MAP[c] for c in seq], dtype=np.float32)
    return ints * (np.pi / 3.0)

def load_data(path):
    df = pd.read_csv(path)
    X = np.stack([encode_sequence(seq) for seq in df['sequence']])
    y = df['label'].values.astype(np.float32)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

print("Loading test data...")
X_test, y_test = load_data(TEST_PATH)
print(f"  Test set: {X_test.shape[0]} samples")
print(f"  Label distribution: 3A={int((y_test==0).sum())} | 3B={int((y_test==1).sum())}")

# Rebuild device and circuit (must match training exactly)
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
vqc_layer    = qml.qnn.TorchLayer(circuit_node, weight_shape)

class VQCClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vqc = vqc_layer
        self.fc  = nn.Linear(1, 1)

    def forward(self, x):
        x_qubits = x[:, :N_QUBITS]
        out = torch.stack([self.vqc(x_qubits[i]) for i in range(x_qubits.shape[0])])
        out = out.unsqueeze(1)
        out = self.fc(out)
        return out.squeeze(1)

model = VQCClassifier()

# ─────────────────────────────────────────────
# STEP 2: LOAD CHECKPOINT
# ─────────────────────────────────────────────
print(f"\nLoading checkpoint from '{CHECKPOINT_PATH}'...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
print(f"  Checkpoint epoch : {checkpoint['epoch']}")
print(f"  Best val loss    : {checkpoint['val_loss']:.4f}")

# ─────────────────────────────────────────────
# STEP 3: INFERENCE
# ─────────────────────────────────────────────
print("\nRunning inference on test set (this may take a moment)...")
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_probs  = torch.sigmoid(test_logits).numpy()
    test_preds  = (test_probs >= 0.5).astype(float)

y_true = y_test.numpy()

# ─────────────────────────────────────────────
# STEP 4: COMPUTE ALL METRICS
# ─────────────────────────────────────────────

acc          = accuracy_score(y_true, test_preds)
bal_acc      = balanced_accuracy_score(y_true, test_preds)

# Per-class precision, recall, F1 (zero_division=0 avoids warnings if a
# class is never predicted)
prec_3b      = precision_score(y_true, test_preds, pos_label=1, zero_division=0)
rec_3b       = recall_score(y_true, test_preds, pos_label=1, zero_division=0)
f1_3b        = f1_score(y_true, test_preds, pos_label=1, zero_division=0)

prec_3a      = precision_score(y_true, test_preds, pos_label=0, zero_division=0)
rec_3a       = recall_score(y_true, test_preds, pos_label=0, zero_division=0)
f1_3a        = f1_score(y_true, test_preds, pos_label=0, zero_division=0)

f1_macro     = f1_score(y_true, test_preds, average='macro', zero_division=0)
f1_weighted  = f1_score(y_true, test_preds, average='weighted', zero_division=0)

cm           = confusion_matrix(y_true, test_preds)
roc_auc      = roc_auc_score(y_true, test_probs)
pr_auc       = average_precision_score(y_true, test_probs)  # PR-AUC

# Random baselines (for context in the paper)
# - Majority classifier accuracy = fraction of majority class in test set
majority_baseline_acc = (y_true == 0).mean()
# - PR-AUC random baseline = fraction of positives in test set
pr_random_baseline    = (y_true == 1).mean()

# ─────────────────────────────────────────────
# STEP 5: PRINT REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FULL METRICS REPORT — VQC Test Set Evaluation")
print("=" * 60)

print(f"\n  ── Overall ──")
print(f"  Accuracy              : {acc:.4f}   (majority baseline: {majority_baseline_acc:.4f})")
print(f"  Balanced Accuracy     : {bal_acc:.4f}   (random baseline: 0.5000)")

print(f"\n  ── Per-class ──")
print(f"  {'Metric':<22} {'DNMT3A (0)':>12} {'DNMT3B (1)':>12}")
print(f"  {'-'*48}")
print(f"  {'Precision':<22} {prec_3a:>12.4f} {prec_3b:>12.4f}")
print(f"  {'Recall':<22} {rec_3a:>12.4f} {rec_3b:>12.4f}")
print(f"  {'F1-score':<22} {f1_3a:>12.4f} {f1_3b:>12.4f}")

print(f"\n  ── Aggregate F1 ──")
print(f"  Macro F1              : {f1_macro:.4f}")
print(f"  Weighted F1           : {f1_weighted:.4f}")

print(f"\n  ── Ranking metrics ──")
print(f"  ROC-AUC               : {roc_auc:.4f}   (random baseline: 0.5000)")
print(f"  PR-AUC                : {pr_auc:.4f}   (random baseline: {pr_random_baseline:.4f})")

print(f"\n  ── Confusion Matrix ──")
print(f"  Rows = true label | Cols = predicted label")
print(f"  {'':20} Pred 3A    Pred 3B")
print(f"  True 3A (0)        {cm[0,0]:>6}     {cm[0,1]:>6}")
print(f"  True 3B (1)        {cm[1,0]:>6}     {cm[1,1]:>6}")

tn, fp, fn, tp = cm.ravel()
print(f"\n  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"  (TN=correct 3A, FP=3A called as 3B, FN=3B missed, TP=correct 3B)")

print(f"\n  ── sklearn classification_report ──")
print(classification_report(y_true, test_preds,
                             target_names=["DNMT3A (0)", "DNMT3B (1)"],
                             zero_division=0))
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 6: FIGURES
# ─────────────────────────────────────────────
print("\nGenerating figures...")

# ── Figure 1: Confusion Matrix ────────────────
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)

classes = ["DNMT3A (0)", "DNMT3B (1)"]
tick_marks = [0, 1]
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, rotation=30, ha='right')
ax.set_yticklabels(classes)

# Annotate each cell
thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]}",
                ha="center", va="center",
                color="white" if cm[i,j] > thresh else "black",
                fontsize=14, fontweight='bold')

ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
ax.set_title(f"VQC Confusion Matrix (Test Set)\nAcc={acc:.3f} | Bal.Acc={bal_acc:.3f}")
plt.tight_layout()
plt.savefig("metrics_confusion_matrix.png", dpi=150)
plt.close()
print("  Saved: metrics_confusion_matrix.png")

# ── Figure 2: ROC + PR curves side by side ────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# ROC curve
fpr, tpr, _ = roc_curve(y_true, test_probs)
ax1.plot(fpr, tpr, color='steelblue', lw=2, label=f"VQC (AUC = {roc_auc:.3f})")
ax1.plot([0, 1], [0, 1], 'k--', lw=1, label="Random (AUC = 0.500)")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve — Test Set")
ax1.legend(loc="lower right")
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.02])
ax1.grid(alpha=0.3)

# PR curve
prec_curve, rec_curve, _ = precision_recall_curve(y_true, test_probs)
ax2.plot(rec_curve, prec_curve, color='darkorange', lw=2,
         label=f"VQC (PR-AUC = {pr_auc:.3f})")
ax2.axhline(y=pr_random_baseline, color='k', linestyle='--', lw=1,
            label=f"Random (PR-AUC ≈ {pr_random_baseline:.3f})")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve — Test Set")
ax2.legend(loc="upper right")
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.02])
ax2.grid(alpha=0.3)

plt.suptitle("VQC Classifier — Test Set Curves", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("metrics_roc_pr_curves.png", dpi=150)
plt.close()
print("  Saved: metrics_roc_pr_curves.png")

print("\nDone. All metrics computed from best checkpoint weights.")