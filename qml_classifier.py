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
N_QUBITS    = 4      # number of qubits; 4 is a good starting point for 8-mer sequences
N_LAYERS    = 2      # number of ansatz repetition layers (more = more expressive, slower)
N_EPOCHS    = 30     # number of training epochs
BATCH_SIZE  = 128     # samples per gradient update
LR          = 0.01   # learning rate for Adam optimizer
SHOTS       = 256    # number of circuit measurement shots (higher = less noise, slower)

TRAIN_PATH  = "data/deep_enzymology_qmproxy_train.csv"
VAL_PATH    = "data/deep_enzymology_qmproxy_val.csv"
TEST_PATH   = "data/deep_enzymology_qmproxy_test.csv"


# ─────────────────────────────────────────────
# STEP 1: DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

# Mapping from nucleotide character to integer
NUC_MAP = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

def encode_sequence(seq):
    """
    Convert an 8-mer DNA string into a numpy array of angles in [0, pi].
    Each nucleotide maps to an integer (0-3), then scaled to an angle.
    These angles will be used as rotation angles in the quantum circuit.
    
    Example: 'ATGC' -> [0, 1, 2, 3] -> [0.0, 1.047, 2.094, 3.141]
    """
    ints = np.array([NUC_MAP[c] for c in seq], dtype=np.float32)
    # Scale from [0, 3] to [0, pi] so values work as rotation angles
    angles = ints * (np.pi / 3.0)
    return angles

def load_data(path):
    """Load a CSV, encode sequences, return feature tensor and label tensor."""
    df = pd.read_csv(path)
    
    # Encode each sequence into 8 angles
    X = np.stack([encode_sequence(seq) for seq in df['sequence']])
    y = df['label'].values.astype(np.float32)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor

print("Loading data...")
X_train, y_train = load_data(TRAIN_PATH)
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
#dev = qml.device("default.qubit", wires=N_QUBITS, shots=SHOTS)
dev = qml.device("lightning.gpu", wires=N_QUBITS, shots=SHOTS)


# ─────────────────────────────────────────────
# STEP 3: QUANTUM CIRCUIT (THE VQC)
# ─────────────────────────────────────────────

def quantum_circuit(inputs, weights):
    """
    The Variational Quantum Circuit (VQC).
    
    inputs:  tensor of shape (N_QUBITS,) — the encoded DNA features (angles)
             NOTE: we only use the first N_QUBITS values of the 8-length input.
             With N_QUBITS=4, we use the first 4 nucleotide angles.
    weights: tensor of shape (N_LAYERS, N_QUBITS, 3) — the trainable parameters
             Each qubit in each layer gets 3 rotation angles (RX, RY, RZ).
    
    Returns: expectation value of PauliZ on qubit 0 — a scalar in [-1, +1].
             This is the model's raw output score.
    """
    # --- Encoding layer ---
    # Embed the input DNA features into the qubit states using angle encoding.
    # RY(angle) rotates qubit i by the angle corresponding to nucleotide i.
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # --- Ansatz layers (repeated N_LAYERS times) ---
    for layer in range(N_LAYERS):
        # Trainable rotation gates on each qubit
        for i in range(N_QUBITS):
            qml.RX(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)

        # Entangling gates: chain of CNOTs connecting adjacent qubits
        # This creates correlations between features (nucleotide positions)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        # Also connect last qubit back to first (ring topology)
        qml.CNOT(wires=[N_QUBITS - 1, 0])

    # --- Measurement ---
    # Measure qubit 0's expectation value under PauliZ
    # Output is a scalar in [-1, +1]
    return qml.expval(qml.PauliZ(0))


# Wrap the circuit as a PennyLane QNode (executable quantum function)
circuit_node = qml.QNode(quantum_circuit, dev, interface="torch")

# Define the shape of the trainable weights tensor
weight_shape = {"weights": (N_LAYERS, N_QUBITS, 3)}

# Wrap into a PyTorch-compatible layer using TorchLayer
# This makes the VQC behave exactly like an nn.Linear layer
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
        # Only use first N_QUBITS features (first 4 nucleotide angles)
        x_qubits = x[:, :N_QUBITS]
        
        # Run each sample through the VQC
        # vqc_layer processes one sample at a time in a batch
        out = torch.stack([self.vqc(x_qubits[i]) for i in range(x_qubits.shape[0])])
        out = out.unsqueeze(1)   # shape: (batch, 1)
        out = self.fc(out)       # shape: (batch, 1)
        return out.squeeze(1)    # shape: (batch,)


model     = VQCClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # handles class imbalance


# ─────────────────────────────────────────────
# STEP 5: TRAINING LOOP  (replace existing)
# ─────────────────────────────────────────────
 
CHECKPOINT_PATH = "best_vqc_checkpoint.pt"
 
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
