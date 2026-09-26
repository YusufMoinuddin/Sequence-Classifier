# src/vqc_common.py
"""
Shared VQC machinery: encoding selection, device selection, circuit construction.

Before this module, five scripts (qml_classifier.py, multi_seed_eval.py,
evaluate_metrics.py, sanity_checks.py, pos_weight_ablation.py) each carried their own
verbatim copy of NUC_MAP, encode_sequence, the device init, the circuit definition and
weight_shape. That duplication is exactly how a set of supposedly-matched runs drifts
apart, so it lives here once.

Encoding is selected with --encoding / VQC_ENCODING, device with --device / VQC_DEVICE.
N_QUBITS is derived from the encoding (4 / 4 / 8), never hardcoded.

CONFIG 1 EQUIVALENCE
--------------------
Config 1 is the reproducibility baseline and must stay bit-for-bit identical to the
original code. Two things preserve that:

  * load_data() returns X already at the circuit's width, because
    encode_config1_slice4() does encode-all-8-then-take-first-4 — the exact operation
    the original performed as encode_sequence() followed by x[:, :N_QUBITS]. The slice
    therefore disappears from forward() without changing any number.
  * weight_shape stays (N_LAYERS, 4, 3) for config1, so the TorchLayer draws the same
    number of parameters in the same order from the torch RNG.

verify_config1_unchanged.py asserts both, plus exact forward-pass equivalence on an
analytic (shots=None) device.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn

from src.qml_encodings import (
    ENCODINGS,
    encode_sequence,
    n_qubits_for,
    positions_feeding_qubit,
    CONFIG_DESCRIPTIONS,
)

DEFAULT_ENCODING = "config1"          # preserves current behaviour when unspecified
# lightning.qubit (CPU) is the default: it is always present wherever pennylane-lightning
# is installed, and it measured 3-4x faster than default.qubit on these 4-8 qubit
# circuits. lightning.gpu is an optional install and would crash on import if it were
# the default. Override with --device / VQC_DEVICE.
DEFAULT_DEVICE = "lightning.qubit"

VALID_ENCODINGS = tuple(sorted(ENCODINGS))


# ─────────────────────────────────────────────────────────────────────────────
# Flag / environment resolution
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser(description: str = "") -> argparse.ArgumentParser:
    """Standard flags shared by every VQC pipeline script."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--encoding", choices=VALID_ENCODINGS, default=None,
                   help=f"Encoding config (default: $VQC_ENCODING or {DEFAULT_ENCODING})")
    p.add_argument("--device", default=None,
                   help=f"PennyLane device (default: $VQC_DEVICE or {DEFAULT_DEVICE})")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seeds, e.g. 0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    p.add_argument("--shots", type=int, default=None, help="Override shot count")
    p.add_argument("--layers", type=int, default=None, help="Override ansatz layers")
    p.add_argument("--limit-train", type=int, default=None,
                   help="Subsample the training set to N rows (smoke tests)")
    p.add_argument("--smoke", action="store_true",
                   help="Fast plumbing check: 1 layer, 2 epochs, 1 shot, 100 train rows")
    return p


def parse_known(description: str = ""):
    """
    Parse the shared flags, tolerating extra script-specific argv entries.
    Applies --smoke defaults for any override the user did not set explicitly.
    """
    args, _unknown = build_arg_parser(description).parse_known_args()

    if args.smoke:
        if args.layers is None:
            args.layers = 1
        if args.epochs is None:
            args.epochs = 2
        if args.shots is None:
            args.shots = 1
        if args.limit_train is None:
            args.limit_train = 100
        if args.seeds is None:
            args.seeds = "0"
        if args.device is None:
            # lightning.gpu is not available off-cluster; smoke tests must still run.
            args.device = "default.qubit"

    return args


def resolve_encoding(args=None, default: str = DEFAULT_ENCODING) -> str:
    """CLI flag > VQC_ENCODING env var > default."""
    if args is not None and getattr(args, "encoding", None):
        value = args.encoding
    else:
        value = os.environ.get("VQC_ENCODING", default)
    if value not in ENCODINGS:
        raise SystemExit(
            f"Unknown encoding {value!r}. Expected one of {list(VALID_ENCODINGS)}."
        )
    return value


def resolve_device(args=None, default: str = DEFAULT_DEVICE) -> str:
    """CLI flag > VQC_DEVICE env var > default."""
    if args is not None and getattr(args, "device", None):
        return args.device
    return os.environ.get("VQC_DEVICE", default)


def resolve_seeds(args=None, default=(0, 1, 2, 3, 4)):
    if args is not None and getattr(args, "seeds", None):
        return [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    env = os.environ.get("VQC_SEEDS")
    if env:
        return [int(s) for s in env.split(",") if s.strip() != ""]
    return list(default)


def describe_run(encoding: str, device: str, n_qubits: int, **extra) -> str:
    bits = [f"encoding={encoding} ({CONFIG_DESCRIPTIONS[encoding]})",
            f"device={device}", f"n_qubits={n_qubits}"]
    bits += [f"{k}={v}" for k, v in extra.items()]
    return " | ".join(bits)


def tag_path(name: str, encoding: str) -> str:
    """
    'multi_seed_results.csv' -> 'multi_seed_results_config2.csv'

    Every config writes to its own tagged file. The legacy untagged files hold the
    numbers currently in the paper and are never overwritten.
    """
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        return f"{stem}_{encoding}.{ext}"
    return f"{name}_{encoding}"


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str, encoding: str, limit: int = None, seed: int = 0):
    """
    Load a split and encode it under the chosen config.

    Returns (X, y) as torch float32 tensors. X has exactly n_qubits_for(encoding)
    columns, so the circuit consumes it directly — no slicing in forward().

    `limit` subsamples deterministically for smoke tests only.
    """
    df = pd.read_csv(path)
    seqs = df["sequence"].astype(str).str.upper().tolist()
    y = df["label"].to_numpy().astype(np.float32)

    if limit is not None and limit < len(seqs):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(seqs))[:limit]
        seqs = [seqs[i] for i in idx]
        y = y[idx]

    X = np.stack([encode_sequence(s, encoding) for s in seqs]).astype(np.float32)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def pos_weight_from(y: torch.Tensor) -> torch.Tensor:
    """pos_weight = n_neg / n_pos, identical to qml_classifier.py:82."""
    n_neg = float((y == 0).sum().item())
    n_pos = float((y == 1).sum().item())
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Circuit
# ─────────────────────────────────────────────────────────────────────────────

def make_device(device_name: str, n_qubits: int, shots):
    try:
        return qml.device(device_name, wires=n_qubits, shots=shots)
    except Exception as exc:
        raise SystemExit(
            f"Could not create PennyLane device {device_name!r} with {n_qubits} wires.\n"
            f"  {exc}\n"
            "Use --device lightning.qubit (CPU) or --device default.qubit if the GPU "
            "plugin is unavailable here."
        )


def make_circuit(dev, n_qubits: int, n_layers: int):
    """
    The VQC. Gate sequence is copied verbatim from qml_classifier.py:118-142:
    RY angle encoding, then n_layers of (RX,RY,RZ per qubit + CNOT chain + ring
    closure), measuring <Z> on qubit 0. Only the loop bounds are parameterised.
    """

    def quantum_circuit(inputs, weights):
        # --- Encoding layer ---
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        # --- Ansatz layers ---
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

        # --- Measurement ---
        return qml.expval(qml.PauliZ(0))

    return qml.QNode(quantum_circuit, dev, interface="torch")


def weight_shape_for(n_layers: int, n_qubits: int) -> dict:
    return {"weights": (n_layers, n_qubits, 3)}


class VQCClassifier(nn.Module):
    """
    VQC -> scalar in [-1,1] -> Linear(1,1) -> logit.

    Identical to the class in qml_classifier.py:160-186 except that the input arrives
    pre-sized by the encoder, so the x[:, :N_QUBITS] slice is gone.
    """

    def __init__(self, vqc_layer):
        super().__init__()
        self.vqc = vqc_layer
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        out = torch.stack([self.vqc(x[i]) for i in range(x.shape[0])])
        return self.fc(out.unsqueeze(1)).squeeze(1)


def make_vqc_classifier(circuit_node, n_layers: int, n_qubits: int) -> VQCClassifier:
    """
    Build a fresh model. Call AFTER seeding — the TorchLayer draws its initial weights
    from the torch RNG at construction time.
    """
    vqc_layer = qml.qnn.TorchLayer(circuit_node, weight_shape_for(n_layers, n_qubits))
    return VQCClassifier(vqc_layer)


def qubit_label(encoding: str, qubit: int) -> str:
    """'q2<-5+6' — used for ablation labels, correct for paired configs."""
    positions = positions_feeding_qubit(encoding, qubit)
    return f"q{qubit}<-" + "+".join(f"N{p}" for p in positions)
