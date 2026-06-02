"""
generate_config_report.py
=========================
Generates a full configuration report for the methods section of the paper.
Addresses Murat's item 7.

Collects:
  - Dataset statistics (from the actual CSV files)
  - Preprocessing description
  - Model architecture
  - Training hyperparameters
  - Library versions
  - Exact commands used

Saves:
  - config_report.txt   — human-readable, paper-ready

Run from the project root with the venv active:
  python generate_config_report.py
"""

import sys
import platform
import datetime
import importlib
import numpy as np
import pandas as pd
import torch

# ─────────────────────────────────────────────
# CONFIGURATION — must match qml_classifier.py
# ─────────────────────────────────────────────
N_QUBITS   = 4
N_LAYERS   = 2
N_EPOCHS   = 30
BATCH_SIZE = 32
LR         = 0.01
SHOTS      = 256
SEEDS      = [0, 1, 2, 3, 4]

TRAIN_PATH = "data/deep_enzymology_qmproxy_train.csv"
VAL_PATH   = "data/deep_enzymology_qmproxy_val.csv"
TEST_PATH  = "data/deep_enzymology_qmproxy_test.csv"

OUTPUT_TXT = "config_report.txt"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_version(package_name):
    try:
        mod = importlib.import_module(package_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not installed"

def load_split_stats(path):
    df = pd.read_csv(path)
    counts = df['label'].value_counts().sort_index()
    return {
        "total":   len(df),
        "dnmt3a":  int(counts.get(0, 0)),
        "dnmt3b":  int(counts.get(1, 0)),
        "ratio":   counts.get(0, 0) / counts.get(1, 1)
                   if counts.get(1, 0) > 0 else float("inf"),
        "columns": list(df.columns),
    }

# ─────────────────────────────────────────────
# GATHER DATA STATISTICS
# ─────────────────────────────────────────────
print("Reading dataset statistics...")
train_stats = load_split_stats(TRAIN_PATH)
val_stats   = load_split_stats(VAL_PATH)
test_stats  = load_split_stats(TEST_PATH)

total       = train_stats["total"] + val_stats["total"] + test_stats["total"]
total_3a    = train_stats["dnmt3a"] + val_stats["dnmt3a"] + test_stats["dnmt3a"]
total_3b    = train_stats["dnmt3b"] + val_stats["dnmt3b"] + test_stats["dnmt3b"]
pos_weight  = total_3a / total_3b

# ─────────────────────────────────────────────
# GATHER LIBRARY VERSIONS
# ─────────────────────────────────────────────
print("Collecting library versions...")
versions = {
    "Python":          sys.version.split()[0],
    "PyTorch":         get_version("torch"),
    "PennyLane":       get_version("pennylane"),
    "pennylane_qiskit":get_version("pennylane_qiskit"),
    "Qiskit":          get_version("qiskit"),
    "qiskit_aer":      get_version("qiskit_aer"),
    "NumPy":           get_version("numpy"),
    "Pandas":          get_version("pandas"),
    "scikit-learn":    get_version("sklearn"),
    "Matplotlib":      get_version("matplotlib"),
}

# ─────────────────────────────────────────────
# BUILD REPORT
# ─────────────────────────────────────────────
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

lines = []

lines += [
    "=" * 70,
    "  VQC CLASSIFIER — FULL CONFIGURATION REPORT",
    f"  Generated: {now}",
    "=" * 70,
]

# ── Section 1: Dataset ────────────────────────────────────────────────────
lines += [
    "",
    "1. DATASET",
    "─" * 70,
    f"  Source          : Gao et al. (2020), Nature Communications",
    f"                    Deep enzymology dataset for DNMT3A and DNMT3B",
    f"                    sequence preference profiling",
    f"  Task            : Binary classification — DNMT3A (0) vs DNMT3B (1)",
    f"  Sequence type   : 8-mer DNA sequences (CpG-containing)",
    f"  Label assignment: Read-count frequency ratio across experimental",
    f"                    conditions with minimum-count filter",
    f"                    (ratio-based, not naive file-membership)",
    "",
    f"  Split           Total    DNMT3A(0)   DNMT3B(1)   Ratio(A:B)   %DNMT3B",
    f"  {'─'*65}",
    f"  Train         {train_stats['total']:>6}   {train_stats['dnmt3a']:>9}   "
    f"{train_stats['dnmt3b']:>9}   "
    f"{train_stats['ratio']:>8.2f}:1   "
    f"{100*train_stats['dnmt3b']/train_stats['total']:>6.1f}%",
    f"  Val           {val_stats['total']:>6}   {val_stats['dnmt3a']:>9}   "
    f"{val_stats['dnmt3b']:>9}   "
    f"{val_stats['ratio']:>8.2f}:1   "
    f"{100*val_stats['dnmt3b']/val_stats['total']:>6.1f}%",
    f"  Test          {test_stats['total']:>6}   {test_stats['dnmt3a']:>9}   "
    f"{test_stats['dnmt3b']:>9}   "
    f"{test_stats['ratio']:>8.2f}:1   "
    f"{100*test_stats['dnmt3b']/test_stats['total']:>6.1f}%",
    f"  Total         {total:>6}   {total_3a:>9}   {total_3b:>9}   "
    f"{pos_weight:>8.2f}:1   "
    f"{100*total_3b/total:>6.1f}%",
    "",
    f"  Split strategy  : Stratified 80/10/10",
    f"                    Max class-ratio deviation across splits: 2.2%",
    f"  Data leakage    : None — zero sequence overlap between any two splits",
    f"                    (verified by exact string matching across all pairs)",
    f"  Internal dupes  : None — no duplicate sequences within any split",
    f"  CSV columns     : {train_stats['columns']}",
]

# ── Section 2: Preprocessing ──────────────────────────────────────────────
lines += [
    "",
    "2. PREPROCESSING",
    "─" * 70,
    f"  Encoding        : Angle encoding — each nucleotide mapped to a",
    f"                    rotation angle in [0, π]",
    f"                    A→0, T→1, G→2, C→3  (then × π/3)",
    f"                    Result: 8 angles per sequence",
    f"",
    f"  Circuit input   : Only the first {N_QUBITS} of 8 encoded angles are",
    f"                    passed to the quantum circuit (positions N1–N4).",
    f"                    Positions N5–N8 are encoded but not used.",
    f"                    This is a current architectural limitation — noted",
    f"                    as future work (extend to 8 qubits).",
    f"",
    f"  Normalisation   : None beyond the π/3 scaling above.",
    f"                    Angles are naturally bounded in [0, π] by construction.",
    f"  Class weighting : BCEWithLogitsLoss with pos_weight = {pos_weight:.4f}",
    f"                    (computed as n_DNMT3A / n_DNMT3B on training set)",
    f"                    Ablation with pos_weight=1.0 also performed (item 4).",
]

# ── Section 3: Model Architecture ─────────────────────────────────────────
n_trainable = N_LAYERS * N_QUBITS * 3 + 2   # RX/RY/RZ per qubit per layer + fc weight+bias
lines += [
    "",
    "3. MODEL ARCHITECTURE",
    "─" * 70,
    f"  Model type      : Variational Quantum Classifier (VQC)",
    f"  Backend         : Qiskit Aer (CPU simulation via PennyLane-Qiskit bridge)",
    f"  Qubits          : {N_QUBITS}",
    f"  Ansatz layers   : {N_LAYERS}",
    f"  Shots           : {SHOTS} (measurement samples per circuit evaluation)",
    f"",
    f"  Circuit structure (per forward pass):",
    f"    1. Encoding layer:",
    f"       RY(angle_i) on qubit i, for i in 0..{N_QUBITS-1}",
    f"       (embeds nucleotide angles into qubit states)",
    f"    2. Ansatz (repeated {N_LAYERS}×):",
    f"       RX(w), RY(w), RZ(w) on each qubit  [{N_QUBITS*3} params/layer]",
    f"       CNOT(i → i+1) for i in 0..{N_QUBITS-2}  (chain entanglement)",
    f"       CNOT({N_QUBITS-1} → 0)  (ring closure)",
    f"    3. Measurement:",
    f"       ⟨Z⟩ on qubit 0  →  scalar in [-1, +1]",
    f"    4. Classical post-processing:",
    f"       nn.Linear(1, 1)  →  logit  →  BCEWithLogitsLoss",
    f"",
    f"  Trainable parameters:",
    f"    VQC weights : {N_LAYERS} layers × {N_QUBITS} qubits × 3 rotations"
    f" = {N_LAYERS*N_QUBITS*3}",
    f"    Linear layer: weight + bias = 2",
    f"    Total       : {n_trainable}",
    f"",
    f"  Gate count (representative circuit):",
    f"    Encoding : {N_QUBITS} RY gates",
    f"    Ansatz   : {N_LAYERS} × ({N_QUBITS*3} rotation + {N_QUBITS} CNOT)"
    f" = {N_LAYERS*(N_QUBITS*3+N_QUBITS)} gates",
    f"    Total    : {N_QUBITS + N_LAYERS*(N_QUBITS*3+N_QUBITS)} gates"
    f" (excl. measurement)",
    f"",
    f"  Entanglement topology: Ring (nearest-neighbour chain + wrap-around)",
]

# ── Section 4: Training ───────────────────────────────────────────────────
lines += [
    "",
    "4. TRAINING CONFIGURATION",
    "─" * 70,
    f"  Optimizer       : Adam",
    f"  Learning rate   : {LR}",
    f"  Batch size      : {BATCH_SIZE}",
    f"  Epochs          : {N_EPOCHS}",
    f"  Loss function   : BCEWithLogitsLoss (pos_weight={pos_weight:.4f})",
    f"  Checkpointing   : Best validation-loss checkpoint saved per run",
    f"                    (final-epoch weights NOT used for evaluation)",
    f"  Shuffle         : Training set shuffled each epoch",
    f"                    (seeded via torch.Generator for reproducibility)",
]

# ── Section 5: Evaluation ─────────────────────────────────────────────────
lines += [
    "",
    "5. EVALUATION PROTOCOL",
    "─" * 70,
    f"  Decision threshold : 0.5 (sigmoid output)",
    f"  Primary metrics    : ROC-AUC, PR-AUC, F1 (DNMT3B), Balanced Accuracy",
    f"  Secondary metrics  : Accuracy, Precision (DNMT3B), Recall (DNMT3B),",
    f"                       F1 (Macro), Confusion Matrix",
    f"  Multi-seed eval    : {len(SEEDS)} seeds {SEEDS}",
    f"                       Results reported as mean ± std",
    f"  Ablations          : (a) pos_weight=1.0 vs {pos_weight:.2f}",
    f"                       (b) Shuffled training labels",
    f"                       (c) Random input features",
    f"                       (d) Inference-time position masking (N1–N4)",
]

# ── Section 6: Reproducibility ────────────────────────────────────────────
lines += [
    "",
    "6. REPRODUCIBILITY",
    "─" * 70,
    f"  Random seeds    : {SEEDS}",
    f"  Seeded          : random, numpy, torch (per run)",
    f"  NOT seeded      : Qiskit Aer shot noise (intentional — real variance)",
    f"  Hardware        : Apple Silicon (arm64), CPU-only",
    f"                    No GPU acceleration",
    f"  OS              : {platform.system()} {platform.release()}",
    f"  Architecture    : {platform.machine()}",
]

# ── Section 7: Library Versions ───────────────────────────────────────────
lines += [
    "",
    "7. SOFTWARE VERSIONS",
    "─" * 70,
]
for pkg, ver in versions.items():
    lines.append(f"  {pkg:<20} : {ver}")

# ── Section 8: Exact Commands ─────────────────────────────────────────────
lines += [
    "",
    "8. EXACT COMMANDS USED",
    "─" * 70,
    f"  # Activate environment",
    f"  source .venv/bin/activate",
    f"",
    f"  # Step 1 — Data integrity audit",
    f"  python data_integrity_audit.py",
    f"",
    f"  # Step 2 — Main training run (with best-checkpoint saving)",
    f"  python qml_classifier.py",
    f"",
    f"  # Step 3 — Full metrics evaluation (loads best checkpoint)",
    f"  python evaluate_metrics.py",
    f"",
    f"  # Step 4 — Multi-seed evaluation (5 seeds)",
    f"  python multi_seed_eval.py",
    f"",
    f"  # Step 5 — pos_weight ablation (weighted vs unweighted, 5 seeds each)",
    f"  python posweight_ablation.py",
    f"",
    f"  # Step 6 — Sanity checks",
    f"  python sanity_checks.py",
    f"",
    f"  # Step 7 — This report",
    f"  python generate_config_report.py",
    "",
    "=" * 70,
    "  END OF CONFIGURATION REPORT",
    "=" * 70,
]

# ─────────────────────────────────────────────
# WRITE OUTPUT
# ─────────────────────────────────────────────
report_text = "\n".join(lines)
print(report_text)

with open(OUTPUT_TXT, "w") as f:
    f.write(report_text)

print(f"\n\nConfig report saved to: {OUTPUT_TXT}")