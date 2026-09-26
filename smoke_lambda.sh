#!/usr/bin/env bash
# =============================================================================
# smoke_lambda.sh — fast end-to-end plumbing check for the configs being trained.
#
# Runs identically on your Mac (before launching) and on the Lambda instance
# (right after setup_lambda.sh, before any 30-epoch run).
#
#   bash smoke_lambda.sh
#
# ~2 minutes. 1 epoch, 32 shots, 40 training rows per config.
#
# The METRICS FROM THIS ARE MEANINGLESS — 1 epoch on 40 rows at 32 shots is
# noise. The only thing being tested is that every config runs end to end,
# writes a checkpoint and a CSV, and that the aggregator can merge them.
#
# Writes to a scratch dir and deletes it, so smoke output can never be mistaken
# for real results.
# =============================================================================

set -euo pipefail

SMOKE_DIR="${TMPDIR:-/tmp}/vqc_smoke_$$"
DEVICE="${VQC_DEVICE:-lightning.qubit}"

# On Lambda inside an activated venv this is "python"; on a Mac it is usually
# "python3" or the repo venv. Override with PYTHON=... if needed.
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
[ -n "$PYTHON" ] || { echo "No python interpreter found. Set PYTHON=..."; exit 1; }
echo "interpreter: $PYTHON"

cleanup() { rm -rf "$SMOKE_DIR"; }
trap cleanup EXIT

echo "======================================================================"
echo "  SMOKE TEST — config2 + config3"
echo "  device : $DEVICE"
echo "  scratch: $SMOKE_DIR  (deleted on exit)"
echo "======================================================================"

mkdir -p "$SMOKE_DIR"

for CFG in config2 config3; do
    echo ""
    echo "--- $CFG ---"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" run_single_vqc.py \
        --encoding "$CFG" \
        --seed 0 \
        --device "$DEVICE" \
        --epochs 1 \
        --shots 32 \
        --limit-train 40 \
        --outdir "$SMOKE_DIR" \
      | grep -E "VQC RUN|qubits=|train=\(|balanced_acc|CM:|wrote" \
      || { echo "  FAILED: $CFG"; exit 1; }
done

echo ""
echo "--- aggregation ---"
"$PYTHON" aggregate_vqc_results.py --results-dir "$SMOKE_DIR" 2>&1 \
    | grep -E "Loaded|encodings:|WARNING|wrote" \
    || { echo "  FAILED: aggregation"; exit 1; }

echo ""
echo "--- checkpoint reloadability (needed for Braket later) ---"
"$PYTHON" - "$SMOKE_DIR" <<'PYEOF'
import sys, torch
from pathlib import Path
d = Path(sys.argv[1])
need = {"encoding", "seed", "n_qubits", "n_layers", "shots", "threshold",
        "model_state_dict"}
bad = False
for ck in sorted(d.glob("checkpoint_*.pt")):
    c = torch.load(ck, map_location="cpu", weights_only=False)
    missing = need - set(c)
    tag = "OK  " if not missing else "FAIL"
    if missing:
        bad = True
    print(f"  {tag} {ck.name}: encoding={c.get('encoding')} "
          f"n_qubits={c.get('n_qubits')} n_layers={c.get('n_layers')} "
          f"shots={c.get('shots')}" + (f"  MISSING {sorted(missing)}" if missing else ""))
sys.exit(1 if bad else 0)
PYEOF

# The aggregator writes its summary to the repo root; remove the smoke version so
# it cannot be confused with real output later.
rm -f vqc_all_configs_results.csv vqc_all_configs_summary.txt vqc_confusion_matrices.png

echo ""
echo "======================================================================"
echo "  SMOKE TEST PASSED — config2 + config3 ran, aggregated, checkpoints load"
echo "  (metrics above are noise by design; scratch dir removed)"
echo "======================================================================"
