# Lambda run commands

Replaces the old Slurm job array. **No orchestrator** — one explicit command per run, so
a failure is visible in its own log instead of being swallowed by a wrapper.

Every command assumes: `cd ~/Sequence-Classifier && source .venv/bin/activate`.

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` is on every command and is **not optional**. Torch
defaults to multiple threads per process; 15 concurrent runs each grabbing 6 threads
would oversubscribe the box and make everything slower. One thread per process, many
processes, is the right shape for this workload.

---

## 0. Before anything

```bash
bash smoke_lambda.sh          # ~2 min, must pass
python benchmark_devices.py   # decides lightning.qubit vs lightning.gpu
```

Set `DEV` to whatever the benchmark recommends. Default:

```bash
export DEV=lightning.qubit
```

---

## 1. Expected cost

Measured locally on CPU (`lightning.qubit`, 256 shots, 2 layers, 30 epochs):

| Config | qubits | hr / run | x5 seeds |
|---|---|---|---|
| config1 | 4 | ~0.5 | ~2.5 |
| config2 | 4 | ~0.5 | ~2.5 |
| config3 | 8 | ~1.7 | ~8.5 |

**Main experiment ≈ 13.5 CPU-hours.** Run in parallel and wall time ≈ the slowest single
run (~1.7 h for config3), provided you have enough vCPUs. Lambda instance speed will
differ from the Mac — trust `benchmark_devices.py` on the box.

Config 3 costs ~3.4x Config 1 because parameter-shift needs `2 x n_params` circuit
evaluations per sample: 96 for Config 3 vs 48 for Configs 1/2, on larger circuits.

---

## 2. The 15 main runs

Start the long ones (config3) first so they are not the tail of the job.

```bash
mkdir -p results logs
```

### config3 — 8 qubits, slowest, launch first

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config3 --seed 0 --device $DEV --outdir results 2>&1 | tee logs/config3_seed0.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config3 --seed 1 --device $DEV --outdir results 2>&1 | tee logs/config3_seed1.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config3 --seed 2 --device $DEV --outdir results 2>&1 | tee logs/config3_seed2.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config3 --seed 3 --device $DEV --outdir results 2>&1 | tee logs/config3_seed3.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config3 --seed 4 --device $DEV --outdir results 2>&1 | tee logs/config3_seed4.log
```

### config1 — baseline

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config1 --seed 0 --device $DEV --outdir results 2>&1 | tee logs/config1_seed0.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config1 --seed 1 --device $DEV --outdir results 2>&1 | tee logs/config1_seed1.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config1 --seed 2 --device $DEV --outdir results 2>&1 | tee logs/config1_seed2.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config1 --seed 3 --device $DEV --outdir results 2>&1 | tee logs/config1_seed3.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config1 --seed 4 --device $DEV --outdir results 2>&1 | tee logs/config1_seed4.log
```

### config2 — pair condensation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config2 --seed 0 --device $DEV --outdir results 2>&1 | tee logs/config2_seed0.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config2 --seed 1 --device $DEV --outdir results 2>&1 | tee logs/config2_seed1.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config2 --seed 2 --device $DEV --outdir results 2>&1 | tee logs/config2_seed2.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config2 --seed 3 --device $DEV --outdir results 2>&1 | tee logs/config2_seed3.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py --encoding config2 --seed 4 --device $DEV --outdir results 2>&1 | tee logs/config2_seed4.log
```

---

## 3. Running them in tmux

One window per run, so each failure is visible and isolated.

```bash
tmux new -s vqc          # detach with Ctrl-b then d; reattach: tmux attach -t vqc
```

Create a window per run and send the command into it:

```bash
for CFG in config3 config1 config2; do
  for SEED in 0 1 2 3 4; do
    tmux new-window -t vqc -n "${CFG}_s${SEED}"
    tmux send-keys -t "vqc:${CFG}_s${SEED}" \
      "cd ~/Sequence-Classifier && source .venv/bin/activate && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_single_vqc.py \
--encoding ${CFG} --seed ${SEED} --device \$DEV --outdir results 2>&1 \
| tee logs/${CFG}_seed${SEED}.log" C-m
  done
done
```

**Only launch as many at once as you have vCPUs.** Check with `nproc`. If you have fewer
than 15, run config3 (5 runs) first, then config1 and config2 together.

Useful tmux: `tmux list-windows -t vqc`, `tmux attach -t vqc`,
`Ctrl-b w` to pick a window, `Ctrl-b d` to detach.

---

## 4. Monitoring

```bash
tail -f logs/*.log                                   # everything at once
tail -f logs/config3_seed0.log                       # one run
grep -c "^  Ep " logs/config3_seed0.log              # epochs done (of 30)
for f in logs/*.log; do printf "%-28s %s/30\n" "$(basename $f)" "$(grep -c '^  Ep ' $f)"; done
ls results/*.csv | wc -l                             # completed runs (want 15)
nproc; uptime                                        # load check
```

Find runs that died without producing output:

```bash
grep -L "wrote results/" logs/*.log
```

Empty output means every run completed.

---

## 5. Aggregate (after all 15 finish)

```bash
python aggregate_vqc_results.py --results-dir results
```

Writes `vqc_all_configs_results.csv`, `vqc_all_configs_summary.txt`,
`vqc_confusion_matrices.png`. It warns if any config has fewer than 5 seeds — do not
ignore that warning.

---

## 6. Optional extras (in scope, but expensive)

Per config, `sanity_checks.py` is 2 training runs and `pos_weight_ablation.py` is
**6** (3 seeds x 2 conditions). Across 3 configs that is 24 more 30-epoch runs — nearly
double the main experiment. Run only if you still want them after the main 15 land.

```bash
for CFG in config1 config2 config3; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python sanity_checks.py --encoding $CFG --device $DEV 2>&1 | tee logs/sanity_${CFG}.log
done

for CFG in config1 config2 config3; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python pos_weight_ablation.py --encoding $CFG --device $DEV 2>&1 | tee logs/posweight_${CFG}.log
done
```

Full metrics + ROC/PR and confusion-matrix figures from a saved checkpoint (inference
only, seconds):

```bash
for CFG in config1 config2 config3; do
  cp results/checkpoint_${CFG}_seed0.pt best_vqc_checkpoint_${CFG}.pt
  python evaluate_metrics.py --encoding $CFG --device $DEV 2>&1 | tee logs/evalmetrics_${CFG}.log
done
```

**Do not run** `multi_seed_eval.py` (exact duplicate of the 15 runs above, same
hyperparameters — double cost, same numbers) or `qml_classifier.py` (uses
`BATCH_SIZE=128`, unseeded, not comparable).

---

## 7. Retrieval — run these on your MAC, before terminating

```bash
mkdir -p ~/lambda_harvest
scp -r ubuntu@<INSTANCE_IP>:~/Sequence-Classifier/results ~/lambda_harvest/
scp -r ubuntu@<INSTANCE_IP>:~/Sequence-Classifier/logs    ~/lambda_harvest/
scp ubuntu@<INSTANCE_IP>:~/Sequence-Classifier/vqc_all_configs_*.{csv,txt} ~/lambda_harvest/
scp ubuntu@<INSTANCE_IP>:~/Sequence-Classifier/vqc_confusion_matrices.png  ~/lambda_harvest/
```

Backup copy via git, **from the instance**:

```bash
git add -f results/*.csv logs/*.log vqc_all_configs_*.csv vqc_all_configs_*.txt
git commit -m "Lambda run results: 3 configs x 5 seeds"
git push origin lambda-runs
```

`-f` is required because `.gitignore` excludes some of these. **`.gitignore` contains
`*.pt`, so checkpoints are NOT captured by git.** scp is the only thing that brings back
the Braket inputs — do not skip it.

---

## 8. Pre-termination checklist

Run on your **Mac** against `~/lambda_harvest`. Do not terminate until all pass.

```bash
cd ~/lambda_harvest
ls results/checkpoint_*.pt | wc -l     # expect 15
ls results/vqc_*_seed*.csv  | wc -l    # expect 15
ls logs/*.log               | wc -l    # expect 15 (+ extras if run)
```

Verify the checkpoints actually load and carry the Braket fields:

```bash
python3 - <<'EOF'
import torch, glob
need = {"encoding","seed","n_qubits","n_layers","shots","threshold","model_state_dict"}
files = sorted(glob.glob("results/checkpoint_*.pt"))
print(f"{len(files)} checkpoints (expect 15)")
for f in files:
    c = torch.load(f, map_location="cpu", weights_only=False)
    missing = need - set(c)
    print(("OK   " if not missing else "FAIL "), f.split("/")[-1],
          c.get("encoding"), "q=%s" % c.get("n_qubits"),
          ("" if not missing else f"MISSING {sorted(missing)}"))
EOF
```

Confirm all 15 rows merged and no config is short of seeds:

```bash
python3 -c "
import pandas as pd
d = pd.read_csv('vqc_all_configs_results.csv')
print(d.groupby('encoding').seed.count())
print('total rows:', len(d), '(expect 15)')
"
```

Final gates before `terminate`:

- [ ] 15 `.pt`, 15 `.csv`, 15 `.log` present on the Mac
- [ ] every checkpoint loads and has all 7 required fields
- [ ] `vqc_all_configs_results.csv` has 15 rows, 5 per config
- [ ] summary + confusion-matrix figure copied
- [ ] any extras you ran (sanity / ablation / evaluate_metrics) also copied
- [ ] `git push origin lambda-runs` succeeded (backup of CSVs/logs)

Terminating deletes local disk permanently. There is no pause.
