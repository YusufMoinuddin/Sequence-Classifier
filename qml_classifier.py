"""
Variational Quantum Classifier (VQC) for DNMT3A vs DNMT3B sequence classification.
Uses PennyLane for the quantum circuit and Qiskit Aer as the simulation backend.
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import time
import tracemalloc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIGURATION — tweak these as needed
# ─────────────────────────────────────────────
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.qml_encodings import n_qubits_for, CONFIG_DESCRIPTIONS
from src.vqc_common import (
    parse_known, resolve_encoding, resolve_device,
    load_data as _load_data_shared, make_device, make_circuit,
    weight_shape_for, tag_path,
)

_args = parse_known("VQC training")
ENCODING = resolve_encoding(_args)
DEVICE = resolve_device(_args)

N_QUBITS    = n_qubits_for(ENCODING)   # 4 / 4 / 8, derived from the encoding
N_LAYERS    = _args.layers if _args.layers is not None else 2
N_EPOCHS    = _args.epochs if _args.epochs is not None else 30
BATCH_SIZE  = 128     # samples per gradient update
LR          = 0.01   # learning rate for Adam optimizer
SHOTS       = _args.shots if _args.shots is not None else 256
LIMIT_TRAIN = _args.limit_train

TRAIN_PATH  = "data/deep_enzymology_qmproxy_train.csv"
VAL_PATH    = "data/deep_enzymology_qmproxy_val.csv"
TEST_PATH   = "data/deep_enzymology_qmproxy_test.csv"


# ─────────────────────────────────────────────
# STEP 1: DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

# Encoding now comes from src/qml_encodings.py, selected by --encoding.
# The nucleotide mapping and pi/3 scaling that used to live here are unchanged for
# config1 — see verify_config1_unchanged.py for the bit-for-bit proof.

def load_data(path, limit=None):
    """Load a CSV and encode it under the selected config."""
    return _load_data_shared(path, ENCODING, limit=limit)

print("Loading data...")
print(f"  Encoding: {ENCODING} — {CONFIG_DESCRIPTIONS[ENCODING]}")
print(f"  Device: {DEVICE} | Qubits: {N_QUBITS} | Layers: {N_LAYERS}")
X_train, y_train = load_data(TRAIN_PATH, limit=LIMIT_TRAIN)
X_val,   y_val   = load_data(VAL_PATH)
X_test,  y_test  = load_data(TEST_PATH)

print(f"  Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
print(f"  Label distribution (train): 3A={int((y_train==0).sum())} | 3B={int((y_train==1).sum())}")

# Compute class weight to handle the ~5.4:1 imbalance
n_neg = (y_train == 0).sum().item()   # DNMT3A count
n_pos = (y_train == 1).sum().item()   # DNMT3B count
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
print(f"  Class weight for DNMT3B (positive class): {pos_weight.item():.2f}x")

# Build DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ─────────────────────────────────────────────
# STEP 2: QUANTUM DEVICE
# ─────────────────────────────────────────────

print(f"\nInitializing quantum device ({N_QUBITS} qubits, {SHOTS} shots)...")

# Use Qiskit Aer as the simulation backend via the PennyLane-Qiskit bridge
dev = make_device(DEVICE, N_QUBITS, SHOTS)


# ─────────────────────────────────────────────
# STEP 3: QUANTUM CIRCUIT (THE VQC)
# ─────────────────────────────────────────────

# The circuit now comes from src/vqc_common.make_circuit(), which is the same gate
# sequence (RY encoding, then N_LAYERS x [RX/RY/RZ per qubit + CNOT chain + ring
# closure], measuring <Z> on qubit 0) with the loop bounds driven by N_QUBITS.
circuit_node = make_circuit(dev, n_qubits=N_QUBITS, n_layers=N_LAYERS)
weight_shape = weight_shape_for(N_LAYERS, N_QUBITS)
vqc_layer = qml.qnn.TorchLayer(circuit_node, weight_shape)


# ─────────────────────────────────────────────
# STEP 4: FULL MODEL (VQC + OUTPUT)
# ─────────────────────────────────────────────

class VQCClassifier(nn.Module):
    """
    Full quantum classifier model.
    
    Architecture:
      Input (8 angles) -> VQC (4 qubits) -> scalar output -> sigmoid -> probability
    
    The VQC outputs a value in [-1, +1]. We pass it through a linear layer
    to get a logit, which the loss function converts to a probability internally.
    """
    def __init__(self):
        super().__init__()
        self.vqc = vqc_layer
        # Small classical post-processing: map VQC scalar output to a logit
        self.fc  = nn.Linear(1, 1)

    def forward(self, x):
        # x already arrives at circuit width from the encoder — no slice needed.
        # Run each sample through the VQC
        # vqc_layer processes one sample at a time in a batch
        out = torch.stack([self.vqc(x[i]) for i in range(x.shape[0])])
        out = out.unsqueeze(1)   # shape: (batch, 1)
        out = self.fc(out)       # shape: (batch, 1)
        return out.squeeze(1)    # shape: (batch,)


model     = VQCClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # handles class imbalance


# ─────────────────────────────────────────────
# STEP 5: TRAINING LOOP  (replace existing)
# ─────────────────────────────────────────────
 
CHECKPOINT_PATH = tag_path("best_vqc_checkpoint.pt", ENCODING)
 
print(f"\nTraining VQC for {N_EPOCHS} epochs...")
print(f"  Config: {N_QUBITS} qubits | {N_LAYERS} layers | lr={LR} | batch={BATCH_SIZE}")
print(f"  Best checkpoint will be saved to: {CHECKPOINT_PATH}")
print("-" * 60)
 
train_losses   = []
val_losses     = []
val_accuracies = []
 
best_val_loss  = float("inf")
best_epoch     = -1
 
for epoch in range(N_EPOCHS):
    # ── Training ──
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)
 
    # ── Validation ──
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss   = criterion(val_logits, y_val).item()
        val_preds  = (torch.sigmoid(val_logits) >= 0.5).float()
        val_acc    = accuracy_score(y_val.numpy(), val_preds.numpy())
 
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
 
    # ── Checkpoint ──
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        best_epoch    = epoch + 1
        torch.save({
            "epoch":            epoch + 1,
            "model_state_dict": model.state_dict(),
            "val_loss":         val_loss,
            "val_acc":          val_acc,
            "config": {
                "n_qubits": N_QUBITS,
                "n_layers": N_LAYERS,
                "shots":    SHOTS,
                "lr":       LR,
                "batch":    BATCH_SIZE,
            }
        }, CHECKPOINT_PATH)
 
    marker = " ★ (new best)" if is_best else ""
    print(f"  Epoch {epoch+1:02d}/{N_EPOCHS} | "
          f"Train Loss: {epoch_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}{marker}")
 
print("-" * 60)
print(f"Training complete. Best checkpoint: epoch {best_epoch} "
      f"(val loss = {best_val_loss:.4f})")
print(f"Saved to: {CHECKPOINT_PATH}")
 
# ─────────────────────────────────────────────
# STEP 6: ML METRICS ON TEST SET
# ─────────────────────────────────────────────

print("\nEvaluating on test set...")
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_probs  = torch.sigmoid(test_logits).numpy()
    test_preds  = (test_probs >= 0.5).astype(float)

test_acc = accuracy_score(y_test.numpy(), test_preds)
test_auc = roc_auc_score(y_test.numpy(), test_probs)

print(f"  Test Accuracy : {test_acc:.4f}")
print(f"  Test AUC      : {test_auc:.4f}")

# Prediction breakdown — shows if model is just guessing one class
unique, counts = np.unique(test_preds, return_counts=True)
print(f"  Prediction breakdown: {dict(zip(unique, counts))}")


# ─────────────────────────────────────────────
# STEP 7: HARDWARE METRICS
# ─────────────────────────────────────────────

print("\nCollecting hardware metrics from Qiskit Aer...")

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Build a representative Qiskit circuit matching our VQC structure
# (N_QUBITS qubits, N_LAYERS layers of rotations + CNOTs)
def build_benchmark_circuit(n_qubits, n_layers):
    qc = QuantumCircuit(n_qubits)
    # Encoding layer
    for i in range(n_qubits):
        qc.ry(np.pi / 4, i)  # representative angle
    # Ansatz layers
    for _ in range(n_layers):
        for i in range(n_qubits):
            qc.rx(0.5, i)
            qc.ry(0.5, i)
            qc.rz(0.5, i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(n_qubits - 1, 0)
    qc.measure_all()
    return qc

sim = AerSimulator()
benchmark_qc = build_benchmark_circuit(N_QUBITS, N_LAYERS)

# Gate and qubit counts
gate_counts = dict(benchmark_qc.count_ops())
n_qubits_used = benchmark_qc.num_qubits
total_gates = sum(v for k, v in gate_counts.items() if k != 'measure')

# Latency: time a single circuit execution
N_BENCH_RUNS = 3
latencies = []
for _ in range(N_BENCH_RUNS):
    start = time.perf_counter()
    sim.run(benchmark_qc, shots=SHOTS).result()
    end = time.perf_counter()
    latencies.append(end - start)

avg_latency   = np.mean(latencies)
throughput    = 1.0 / avg_latency   # circuits per second

# Memory: peak memory during one circuit execution
tracemalloc.start()
sim.run(benchmark_qc, shots=SHOTS).result()
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_mem_mb = peak_mem / (1024 ** 2)

print(f"\n  ── Quantum Hardware Metrics ──")
print(f"  Qubits used      : {n_qubits_used}")
print(f"  Total gates      : {total_gates}")
print(f"  Gate breakdown   : {gate_counts}")
print(f"  Avg latency      : {avg_latency*1000:.2f} ms per circuit")
print(f"  Throughput       : {throughput:.2f} circuits/sec")
print(f"  Peak memory      : {peak_mem_mb:.2f} MB")


# ─────────────────────────────────────────────
# STEP 8: SAVE FIGURES
# ─────────────────────────────────────────────

print("\nGenerating figures...")

# Figure 1: Training loss curve
plt.figure(figsize=(8, 4))
plt.plot(range(1, N_EPOCHS + 1), train_losses, marker='o', label='Train Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VQC Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig("vqc_loss_curve.png", dpi=150)
plt.close()
print("  Saved: vqc_loss_curve.png")

# Figure 2: ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test.numpy(), test_probs)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("VQC ROC Curve (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig("vqc_roc_curve.png", dpi=150)
plt.close()
print("  Saved: vqc_roc_curve.png")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  RESULTS SUMMARY")
print("=" * 55)
print(f"  Model        : VQC ({N_QUBITS} qubits, {N_LAYERS} layers)")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  Test AUC     : {test_auc:.4f}")
print(f"  Qubits       : {n_qubits_used}")
print(f"  Gates        : {total_gates}")
print(f"  Latency      : {avg_latency*1000:.2f} ms")
print(f"  Throughput   : {throughput:.2f} circuits/sec")
print(f"  Peak Memory  : {peak_mem_mb:.2f} MB")
print("=" * 55)
