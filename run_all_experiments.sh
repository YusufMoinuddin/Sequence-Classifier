#!/bin/bash
# run_all_experiments.sh
# ======================
# Chains all VQC experiments sequentially.
# All Python scripts read SMOKE_TEST internally.
# Logs start time of each step so you can estimate remaining runtime.
#
# Usage:
#   source .venv/bin/activate
#   caffeinate -i bash run_all_experiments.sh

LOG="run_all_experiments.log"
exec > >(tee -a "$LOG") 2>&1

echo "========================================================"
echo "  VQC EXPERIMENT SUITE — STARTED"
echo "  $(date)"
echo "========================================================"

run_step() {
    local step_num="$1"
    local step_name="$2"
    local script="$3"
    local start_ts=$(date +%s)

    echo ""
    echo "--------------------------------------------------------"
    echo "  STEP $step_num: $step_name"
    echo "  Started: $(date)"
    echo "--------------------------------------------------------"

    python "$script"
    EXIT_CODE=$?

    local end_ts=$(date +%s)
    local elapsed=$(( end_ts - start_ts ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "  [ERROR] Step $step_num failed (exit code $EXIT_CODE) after ${mins}m ${secs}s"
        echo "  Stopping. Check $LOG for details."
        exit $EXIT_CODE
    fi

    echo ""
    echo "  Step $step_num complete in ${mins}m ${secs}s — $(date)"
}

run_step 1 "Main training run"                            "qml_classifier.py"
run_step 2 "Full metrics evaluation"                      "evaluate_metrics.py"
run_step 3 "Multi-seed evaluation"                        "multi_seed_eval.py"
run_step 4 "pos_weight ablation"                          "posweight_ablation.py"
run_step 5 "Sanity checks"                                "sanity_checks.py"
run_step 6 "Configuration report"                         "generate_config_report.py"

echo ""
echo "========================================================"
echo "  ALL STEPS COMPLETE — $(date)"
echo "========================================================"