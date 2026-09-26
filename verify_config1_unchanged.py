"""
verify_config1_unchanged.py
===========================
Proves Config 1 still behaves exactly as it did before the encoding flag was wired in.

Config 1 is the reproducibility baseline for the paper, so "probably the same" is not
good enough. Three assertions, narrowest to broadest:

  1. ENCODING     — encode_sequence(s, "config1") equals the legacy inline
                    `ints * pi/3` then `[:4]`, exactly, on every training sequence.
  2. SHAPES       — n_qubits and weight_shape are unchanged, so the TorchLayer draws
                    the same parameters in the same order from the torch RNG.
  3. FORWARD PASS — the legacy-style model (8-wide input, x[:, :4] slice, locally
                    defined circuit) and the new-style model (4-wide input, no slice,
                    shared circuit) produce identical logits.

Assertion 3 runs on default.qubit with shots=None. Analytic mode is required: shot-based
sampling is stochastic, so it could never prove equivalence.

This script trains nothing. Run from the project root:
  python verify_config1_unchanged.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from src.qml_encodings import encode_sequence, n_qubits_for
from src.vqc_common import make_circuit, weight_shape_for, make_vqc_classifier, load_data

TRAIN_PATH = "data/deep_enzymology_qmproxy_train.csv"
N_LAYERS = 2          # the published Config 1 setting
SEED = 0

results = {}


def banner(t):
    print("\n" + "=" * 74)
    print(f"  {t}")
    print("=" * 74)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENCODING
# ─────────────────────────────────────────────────────────────────────────────
banner("1. ENCODING — config1 matches the legacy inline encoding exactly")


def legacy_encode_then_slice(seq):
    """Verbatim qml_classifier.py:53-56 followed by the x[:, :N_QUBITS] slice."""
    NUC_MAP = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    ints = np.array([NUC_MAP[c] for c in seq], dtype=np.float32)
    angles = ints * (np.pi / 3.0)
    return angles[:4]


seqs = pd.read_csv(TRAIN_PATH, usecols=["sequence"])["sequence"].astype(str).str.upper().tolist()
mismatches = [
    s for s in seqs
    if not np.array_equal(encode_sequence(s, "config1"), legacy_encode_then_slice(s))
]
results["1. encoding identical on all training sequences"] = not mismatches
print(f"  sequences checked      : {len(seqs)}")
print(f"  exact mismatches       : {len(mismatches)}")
print(f"  [{'PASS' if not mismatches else 'FAIL'}] encoder reproduces legacy encode+slice bit-for-bit")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHAPES
# ─────────────────────────────────────────────────────────────────────────────
banner("2. SHAPES — qubit count and weight_shape unchanged")

nq = n_qubits_for("config1")
ws = weight_shape_for(N_LAYERS, nq)["weights"]
legacy_ws = (N_LAYERS, 4, 3)          # what the original hardcoded
n_params = int(np.prod(ws))

print(f"  n_qubits (config1)     : {nq}          (legacy: 4)")
print(f"  weight_shape           : {ws}   (legacy: {legacy_ws})")
print(f"  trainable VQC params   : {n_params}         (legacy: {int(np.prod(legacy_ws))})")

shapes_ok = (nq == 4) and (tuple(ws) == legacy_ws)
results["2. n_qubits and weight_shape unchanged"] = shapes_ok
print(f"  [{'PASS' if shapes_ok else 'FAIL'}] same parameter count and RNG draw order")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────
banner("3. FORWARD PASS — legacy model vs new model, analytic device")

print("  device: default.qubit, shots=None (analytic -> deterministic)")

BATCH = 16
batch_seqs = seqs[:BATCH]

# New pathway: encoder produces the circuit-width input directly.
# Built from batch_seqs explicitly — load_data(limit=...) permutes before subsampling,
# so using it here would feed the two models different sequences.
X_new = torch.tensor(
    np.stack([encode_sequence(s, "config1") for s in batch_seqs]),
    dtype=torch.float32,
)

# Legacy pathway: encode all 8, feed 8-wide input, slice inside forward().
X_legacy = torch.tensor(
    np.stack([
        np.array([{'A': 0, 'T': 1, 'G': 2, 'C': 3}[c] for c in s], dtype=np.float32) * (np.pi / 3.0)
        for s in batch_seqs
    ]),
    dtype=torch.float32,
)

dev_legacy = qml.device("default.qubit", wires=4, shots=None)
dev_new = qml.device("default.qubit", wires=4, shots=None)


def legacy_circuit(inputs, weights):
    """Verbatim copy of the original quantum_circuit with N_QUBITS=4."""
    N_QUBITS = 4
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


class LegacyVQC(nn.Module):
    """Verbatim structure of the original VQCClassifier, including the input slice."""

    def __init__(self, layer):
        super().__init__()
        self.vqc = layer
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x_q = x[:, :4]                      # the original slice
        out = torch.stack([self.vqc(x_q[i]) for i in range(x_q.shape[0])])
        return self.fc(out.unsqueeze(1)).squeeze(1)


# Same seed before each construction so both draw identical initial weights.
torch.manual_seed(SEED)
np.random.seed(SEED)
legacy_node = qml.QNode(legacy_circuit, dev_legacy, interface="torch")
legacy_model = LegacyVQC(qml.qnn.TorchLayer(legacy_node, {"weights": legacy_ws}))

torch.manual_seed(SEED)
np.random.seed(SEED)
new_node = make_circuit(dev_new, n_qubits=nq, n_layers=N_LAYERS)
new_model = make_vqc_classifier(new_node, n_layers=N_LAYERS, n_qubits=nq)

# Weights must match before the outputs can mean anything.
lw = dict(legacy_model.named_parameters())
nw = dict(new_model.named_parameters())
weights_match = all(
    torch.equal(lw[k], nw[k]) for k in lw.keys() & nw.keys()
) and (lw.keys() == nw.keys())
print(f"  initial weights identical : {weights_match}  ({sorted(lw.keys())})")

legacy_model.eval()
new_model.eval()
with torch.no_grad():
    out_legacy = legacy_model(X_legacy)
    out_new = new_model(X_new)

exact = torch.equal(out_legacy, out_new)
max_abs = (out_legacy - out_new).abs().max().item()

print(f"\n  batch size                : {BATCH}")
print(f"  legacy input width        : {tuple(X_legacy.shape)}  (8-wide, sliced in forward)")
print(f"  new input width           : {tuple(X_new.shape)}  (4-wide, no slice)")
print(f"  max |logit difference|    : {max_abs:.3e}")
print(f"  exactly equal             : {exact}")
print("\n  first 5 logits:")
for i in range(min(5, BATCH)):
    print(f"    {batch_seqs[i]}   legacy={out_legacy[i].item():+.12f}   new={out_new[i].item():+.12f}")

forward_ok = bool(exact and weights_match)
results["3. forward-pass logits identical"] = forward_ok
print(f"\n  [{'PASS' if forward_ok else 'FAIL'}] legacy and new models produce identical logits")


# ─────────────────────────────────────────────────────────────────────────────
banner("SUMMARY")
for name, ok in results.items():
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")

all_ok = all(results.values())
print()
if all_ok:
    print("  Config 1 is unchanged. The refactor is safe for the reproducibility baseline.")
else:
    print("  CONFIG 1 HAS CHANGED — do not trust any Config 1 results until this is fixed.")

sys.exit(0 if all_ok else 1)
