"""
run_single_vqc.py
=================
Runs ONE (encoding, seed) pair and writes a single-row CSV.

Designed for a Slurm job array: 3 configs x 5 seeds = 15 independent tasks that can
run in parallel. aggregate_vqc_results.py merges the rows afterward.

Hyperparameters are FIXED here, not exposed as flags, so all 15 runs are matched by
construction. They are the settings that produced the published Config 1 numbers in
multi_seed_eval.py. The only thing that varies with the encoding is N_QUBITS, which is
forced by the encoding itself (4 / 4 / 8).

Usage:
  python run_single_vqc.py --encoding config2 --seed 3
  python run_single_vqc.py --encoding config1 --seed 0 --device lightning.qubit
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score

from src.qml_encodings import n_qubits_for, CONFIG_DESCRIPTIONS, ENCODINGS
from src.vqc_common import (
    load_data, make_device, make_circuit, weight_shape_for,
    DEFAULT_DEVICE, pos_weight_from,
)
from src.enzyme_common import full_binary_metrics

# ─────────────────────────────────────────────
# FIXED HYPERPARAMETERS — identical for all 15 runs
# ─────────────────────────────────────────────
N_LAYERS = 2
N_EPOCHS = 30
BATCH_SIZE = 32
LR = 0.01
SHOTS = 256
THRESHOLD = 0.5

TRAIN_PATH = "data/deep_enzymology_qmproxy_train.csv"
VAL_PATH = "data/deep_enzymology_qmproxy_val.csv"
TEST_PATH = "data/deep_enzymology_qmproxy_test.csv"

RESULTS_DIR = ROOT / "results"


def main():
    ap = argparse.ArgumentParser(description="Single VQC run (one encoding, one seed)")
    ap.add_argument("--encoding", required=True, choices=sorted(ENCODINGS))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    # Smoke knobs — used only for pipeline testing, never for the reported runs.
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--limit-train", type=int, default=None)
    args = ap.parse_args()

    encoding, seed = args.encoding, args.seed
    n_qubits = n_qubits_for(encoding)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  VQC RUN — encoding={encoding}  seed={seed}")
    print("=" * 70)
    print(f"  {CONFIG_DESCRIPTIONS[encoding]}")
    print(f"  device={args.device} | qubits={n_qubits} | layers={args.layers}")
    print(f"  epochs={args.epochs} | shots={args.shots} | batch={BATCH_SIZE} | lr={LR}")

    # --- data ---
    X_train, y_train = load_data(TRAIN_PATH, encoding, limit=args.limit_train)
    X_val, y_val = load_data(VAL_PATH, encoding)
    X_test, y_test = load_data(TEST_PATH, encoding)
    pos_weight = pos_weight_from(y_train)
    print(f"  train={tuple(X_train.shape)} val={tuple(X_val.shape)} test={tuple(X_test.shape)}")
    print(f"  pos_weight={pos_weight.item():.4f}")

    # --- device + circuit ---
    dev = make_device(args.device, n_qubits, args.shots)
    circuit_node = make_circuit(dev, n_qubits=n_qubits, n_layers=args.layers)
    weight_shape = weight_shape_for(args.layers, n_qubits)

    # --- seed, then build (TorchLayer draws its weights at construction) ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    vqc_layer = qml.qnn.TorchLayer(circuit_node, weight_shape)

    class VQCClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.vqc = vqc_layer
            self.fc = nn.Linear(1, 1)

        def forward(self, x):
            out = torch.stack([self.vqc(x[i]) for i in range(x.shape[0])])
            return self.fc(out.unsqueeze(1)).squeeze(1)

    model = VQCClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=BATCH_SIZE, shuffle=True, generator=g)

    best_val_loss, best_state, best_epoch = float("inf"), None, -1
    t0 = time.time()

    for epoch in range(args.epochs):
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
            val_acc = accuracy_score(
                y_val.numpy(), (torch.sigmoid(model(X_val)) >= THRESHOLD).float().numpy()
            )

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            mark = " *"
        else:
            mark = ""
        print(f"  Ep {epoch+1:02d}/{args.epochs} | train={epoch_loss:.4f} "
              f"| val={val_loss:.4f} | acc={val_acc:.4f}{mark}", flush=True)

    elapsed_min = (time.time() - t0) / 60.0

    # --- best checkpoint -> test metrics ---
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).numpy()

    m = full_binary_metrics(y_test.numpy(), probs, threshold=THRESHOLD)

    row = {
        "encoding": encoding,
        "seed": seed,
        "n_qubits": n_qubits,
        "n_layers": args.layers,
        "epochs": args.epochs,
        "shots": args.shots,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "pos_weight": float(pos_weight.item()),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "runtime_min": elapsed_min,
        **m,
    }

    # Everything the later Braket hardware evaluation needs to rebuild this model
    # without re-deriving anything: the encoding, the circuit geometry, and the
    # decision threshold. n_qubits/n_layers are technically recoverable from the
    # encoding and the vqc.weights shape, but a checkpoint opened months later by a
    # different script should not have to infer them.
    ckpt = outdir / f"checkpoint_{encoding}_seed{seed}.pt"
    torch.save({"encoding": encoding, "seed": seed, "epoch": best_epoch,
                "val_loss": best_val_loss, "model_state_dict": best_state,
                "n_qubits": n_qubits, "n_layers": args.layers,
                "shots": args.shots, "threshold": THRESHOLD,
                "batch_size": BATCH_SIZE, "lr": LR,
                "pos_weight": float(pos_weight.item())}, ckpt)

    out_csv = outdir / f"vqc_{encoding}_seed{seed}.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)

    print("\n  RESULT")
    for k in ["balanced_acc", "roc_auc", "pr_auc",
              "recall_dnmt3b", "precision_dnmt3b", "f1_dnmt3b"]:
        print(f"    {k:<20} {row[k]:.4f}")
    print(f"    CM: TN={row['TN']} FP={row['FP']} FN={row['FN']} TP={row['TP']}")
    print(f"\n  runtime: {elapsed_min:.1f} min")
    print(f"  wrote {out_csv}")
    print(f"  wrote {ckpt}")


if __name__ == "__main__":
    main()
