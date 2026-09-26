"""
benchmark_devices.py
====================
Times a real training step on lightning.qubit vs lightning.gpu so you can decide,
ON THE INSTANCE, which device to use before committing to ~35 CPU-hours of runs.

Benchmarks Config 3 (8 qubits, 48 VQC params) because it is the worst case: it needs
2 x 48 = 96 circuit evaluations per sample for the parameter-shift backward pass, on
top of larger circuits. If GPU does not help here, it will not help Config 1 or 2.

Skips lightning.gpu gracefully if the plugin is not installed (the default under the
CPU-only requirements).

  python benchmark_devices.py
  python benchmark_devices.py --samples 64      # longer, more stable timing

PRIOR EXPECTATION: these circuits are tiny (8 qubits = 256 amplitudes). GPU
kernel-launch overhead per circuit is expected to dominate, making lightning.gpu
SLOWER than lightning.qubit. This script exists to confirm or refute that with
real numbers on real hardware rather than guessing.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

from src.qml_encodings import n_qubits_for
from src.vqc_common import (
    load_data, make_circuit, weight_shape_for, VQCClassifier, pos_weight_from,
)

TRAIN_PATH = "data/deep_enzymology_qmproxy_train.csv"
TRAIN_ROWS = 1652          # full training set size, for extrapolation
EPOCHS_PER_RUN = 30        # the real training budget


def time_device(device_name, encoding, n_samples, batch_size, shots, layers):
    """Run one timed pass. Returns seconds, or None if the device is unavailable."""
    n_qubits = n_qubits_for(encoding)
    try:
        dev = qml.device(device_name, wires=n_qubits, shots=shots)
    except Exception as exc:
        print(f"  {device_name:<18} UNAVAILABLE — {str(exc)[:70]}")
        return None

    X, y = load_data(TRAIN_PATH, encoding, limit=n_samples)
    pw = pos_weight_from(y)

    torch.manual_seed(0)
    node = make_circuit(dev, n_qubits=n_qubits, n_layers=layers)
    model = VQCClassifier(qml.qnn.TorchLayer(node, weight_shape_for(layers, n_qubits)))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    # One warm-up batch so JIT/allocation costs don't land in the measurement.
    xb, yb = next(iter(loader))
    opt.zero_grad()
    crit(model(xb), yb).backward()
    opt.step()

    t0 = time.time()
    for xb, yb in loader:
        opt.zero_grad()
        crit(model(xb), yb).backward()
        opt.step()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", default="config3", help="worst case by default")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()

    n_qubits = n_qubits_for(args.encoding)
    n_params = args.layers * n_qubits * 3

    print("=" * 74)
    print("  DEVICE BENCHMARK")
    print("=" * 74)
    print(f"  encoding    : {args.encoding} ({n_qubits} qubits, {n_params} VQC params)")
    print(f"  samples     : {args.samples} | batch {args.batch_size} | shots {args.shots}")
    print(f"  backward    : parameter-shift -> {2*n_params} circuit evals/sample")
    print(f"  os.cpu_count: {os.cpu_count()}")
    print(f"  torch threads: {torch.get_num_threads()}")
    print()

    results = {}
    for dev_name in ["lightning.qubit", "lightning.gpu"]:
        print(f"  timing {dev_name} ...", flush=True)
        el = time_device(dev_name, args.encoding, args.samples,
                         args.batch_size, args.shots, args.layers)
        if el is not None:
            results[dev_name] = el
            per_epoch_min = el * (TRAIN_ROWS / args.samples) / 60
            print(f"  {dev_name:<18} {el:>7.1f} s / {args.samples} samples"
                  f"  ->  {per_epoch_min:>6.1f} min/epoch"
                  f"  ->  {per_epoch_min*EPOCHS_PER_RUN/60:>5.1f} hr per 30-epoch run")

    print("\n" + "=" * 74)
    print("  RECOMMENDATION")
    print("=" * 74)

    if "lightning.gpu" not in results:
        print("  lightning.gpu is not installed — use lightning.qubit.")
        print("  To test GPU anyway:")
        print("    pip install pennylane-lightning-gpu==0.38.0 custatevec-cu12")
        print("    python benchmark_devices.py")
        best = "lightning.qubit"
    else:
        cpu, gpu = results["lightning.qubit"], results["lightning.gpu"]
        if gpu < cpu:
            print(f"  lightning.gpu is {cpu/gpu:.2f}x FASTER — use --device lightning.gpu")
            print("  (This contradicts the prior expectation; trust the measurement.)")
            best = "lightning.gpu"
        else:
            print(f"  lightning.qubit is {gpu/cpu:.2f}x faster — use --device lightning.qubit")
            print("  (As expected: these circuits are too small to amortise GPU overhead.)")
            best = "lightning.qubit"

    cpus = os.cpu_count() or 4
    print(f"\n  Parallelism: with {cpus} vCPUs and OMP_NUM_THREADS=1, run up to ~{cpus}")
    print(f"  processes concurrently. All 15 runs fit at once if you have >=15 vCPUs.")
    if best == "lightning.gpu":
        print("  NOTE: GPU runs contend for one card — do NOT launch 15 GPU processes.")
        print("        Prefer CPU + parallelism over GPU + serialisation.")

    print(f"\n  Use: --device {best}")


if __name__ == "__main__":
    main()
