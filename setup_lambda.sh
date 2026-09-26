#!/usr/bin/env bash
# =============================================================================
# setup_lambda.sh — prepare a Lambda Labs On-Demand instance for the VQC runs.
#
# The instance bills per minute and cannot be paused, so this script is
# deliberately fail-fast and verifies everything before you spend time on it.
#
# Usage, from the instance (user "ubuntu"):
#
#   git clone -b lambda-runs https://github.com/YusufMoinuddin/Sequence-Classifier.git
#   cd Sequence-Classifier
#   bash setup_lambda.sh
#
# Takes a few minutes. Every check must print OK before you launch real runs.
# =============================================================================

set -euo pipefail

BRANCH="lambda-runs"
VENV=".venv"

say()  { printf "\n\033[1m== %s ==\033[0m\n" "$1"; }
ok()   { printf "  OK    %s\n" "$1"; }
bad()  { printf "  FAIL  %s\n" "$1"; exit 1; }

# ---------------------------------------------------------------------------
say "0. Where am I"
# ---------------------------------------------------------------------------
echo "  host    : $(hostname)"
echo "  pwd     : $(pwd)"
echo "  branch  : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'NOT A GIT REPO')"
echo "  commit  : $(git rev-parse --short HEAD 2>/dev/null || echo '?')"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo "  WARNING: expected branch '$BRANCH', got '$CURRENT_BRANCH'."
    echo "           The corrected encoding work only exists on '$BRANCH'."
    echo "           Run: git checkout $BRANCH"
    exit 1
fi
ok "on branch $BRANCH"

# ---------------------------------------------------------------------------
say "1. The corrected code is actually present"
# ---------------------------------------------------------------------------
# If this clone were the old code, src/qml_encodings.py would not exist and you
# would be paying to retrain the original slice bug.
for f in src/qml_encodings.py src/vqc_common.py run_single_vqc.py \
         src/enzyme_common.py src/qm_features.py aggregate_vqc_results.py; do
    [ -f "$f" ] || bad "missing $f — this clone is NOT the corrected pipeline"
done
ok "all required source files present"

for f in data/deep_enzymology_qmproxy_train.csv \
         data/deep_enzymology_qmproxy_val.csv \
         data/deep_enzymology_qmproxy_test.csv; do
    [ -f "$f" ] || bad "missing $f"
done
ok "all three data files present"

# ---------------------------------------------------------------------------
say "2. Python environment"
# ---------------------------------------------------------------------------
PY=$(command -v python3.11 || command -v python3.10 || command -v python3)
echo "  interpreter: $PY ($("$PY" --version 2>&1))"

"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' \
    || bad "need Python 3.10+"
ok "python version"

if [ ! -d "$VENV" ]; then
    "$PY" -m venv "$VENV"
    ok "created $VENV"
else
    ok "$VENV already exists (reusing)"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip
ok "pip upgraded"

say "3. Installing pinned requirements (a few minutes)"
pip install -r requirements-lambda.txt
ok "requirements installed"

# ---------------------------------------------------------------------------
say "4. Verify imports and device"
# ---------------------------------------------------------------------------
python - <<'PYEOF'
import sys
fails = []

import pennylane as qml, torch, numpy as np, pandas as pd, sklearn, matplotlib
print(f"  pennylane    {qml.version()}")
print(f"  torch        {torch.__version__}")
print(f"  numpy        {np.__version__}")
print(f"  pandas       {pd.__version__}")
print(f"  scikit-learn {sklearn.__version__}")
print(f"  matplotlib   {matplotlib.__version__}")

if int(np.__version__.split(".")[0]) >= 2:
    fails.append("numpy is 2.x — must be <2 for this torch/pennylane build")

# lightning.qubit is the workhorse; it must exist.
try:
    qml.device("lightning.qubit", wires=8, shots=256)
    print("  OK    lightning.qubit (8 wires) available")
except Exception as e:
    fails.append(f"lightning.qubit unavailable: {e}")

# lightning.gpu is optional.
try:
    qml.device("lightning.gpu", wires=8, shots=256)
    print("  NOTE  lightning.gpu IS available — run benchmark_devices.py to see if it helps")
except Exception:
    print("  NOTE  lightning.gpu not installed (expected; CPU-only requirements)")

print(f"\n  torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print("  (The VQC model itself runs on CPU; only a GPU *simulator* would use the card.)")

import os
print(f"\n  os.cpu_count(): {os.cpu_count()}  <- this is what determines parallelism")

if fails:
    print("\nFAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
PYEOF
ok "imports and devices verified"

# ---------------------------------------------------------------------------
say "5. Verify the data and the encoding fix"
# ---------------------------------------------------------------------------
python - <<'PYEOF'
import sys
sys.path.append(".")
import pandas as pd
from src.qml_encodings import n_qubits_for, positions_feeding_qubit

EXPECTED = {
    "data/deep_enzymology_qmproxy_train.csv": (1652, 1394, 258),
    "data/deep_enzymology_qmproxy_val.csv":   (206,  174,  32),
    "data/deep_enzymology_qmproxy_test.csv":  (207,  174,  33),
}
bad = False
for path, (rows, neg, pos) in EXPECTED.items():
    df = pd.read_csv(path)
    n, a, b = len(df), int((df.label == 0).sum()), int((df.label == 1).sum())
    status = "OK  " if (n, a, b) == (rows, neg, pos) else "FAIL"
    if status == "FAIL":
        bad = True
    print(f"  {status}  {path}: rows={n} 3A={a} 3B={b}  (expected {rows}/{neg}/{pos})")

print()
for cfg in ["config1", "config2", "config3"]:
    nq = n_qubits_for(cfg)
    reach = sorted({p for q in range(nq) for p in positions_feeding_qubit(cfg, q)})
    print(f"  {cfg}: {nq} qubits | positions reaching circuit: {reach}")

# Config 2 and 3 must see all 8 positions — that is the whole point of the fix.
for cfg in ["config2", "config3"]:
    nq = n_qubits_for(cfg)
    reach = {p for q in range(nq) for p in positions_feeding_qubit(cfg, q)}
    if reach != set(range(1, 9)):
        print(f"  FAIL  {cfg} does not reach all 8 positions: {sorted(reach)}")
        bad = True

sys.exit(1 if bad else 0)
PYEOF
ok "data counts and encoding reach verified"

# ---------------------------------------------------------------------------
say "SETUP COMPLETE"
# ---------------------------------------------------------------------------
mkdir -p results logs
cat <<'EOF'

  Created: results/  logs/

  NEXT, in order:
    1. source .venv/bin/activate
    2. bash smoke_lambda.sh          # ~2 min, proves the pipeline runs
    3. python benchmark_devices.py   # decide lightning.qubit vs lightning.gpu
    4. see lambda_commands.md        # then launch the real runs in tmux

  Do NOT start 30-epoch runs until steps 2 and 3 pass.
EOF
