"""
verify_encodings.py
===================
Encoding verification ONLY — this script trains nothing.

Requested by Murat and Prof. Gowher before any full training runs, to establish on
real data which nucleotide positions actually reach the quantum circuit under each
of the three encoding configurations.

  CONFIG 1  current 4-position slice   (existing behaviour, diagnostic baseline)
  CONFIG 2  pair condensation, base-4, adjacent pairs (1,2)(3,4)(5,6)(7,8)
  CONFIG 3  full 8-position encoding

Checks performed:
  (a) all 8 positions measurably influence the Config 2 / Config 3 representations
  (b) downstream +1/+2/+3 (positions 6,7,8) are genuinely present in the circuit input
  (c) changing a single downstream nucleotide changes the Config 2/3 circuit input
  (d) the nucleotide ordering in code matches what the manuscript claims

Run from the project root with the venv active:
  python verify_encodings.py

IMPORTANT: qml_classifier.py is a flat script that starts training on import, so it is
never imported here. Config 1 is instead checked against a verbatim transcription of
its encoding lines, plus a source-text check that the constants still match.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd

from src.qml_encodings import (
    NUC_MAP,
    CONFIG2_PAIRS,
    encode_sequence,
    n_qubits_for,
    positions_feeding_qubit,
    CONFIG_DESCRIPTIONS,
)

TRAIN_PATH = "data/deep_enzymology_qmproxy_train.csv"
QML_SOURCE = "qml_classifier.py"
CONFIGS = ["config1", "config2", "config3"]

# What the manuscript (Section III-B) claims the ordinal mapping is.
MANUSCRIPT_NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# Motif layout: NNNCGNNN
POSITION_ROLES = {
    1: "upstream -3", 2: "upstream -2", 3: "upstream -1",
    4: "CpG (C)", 5: "CpG (G)",
    6: "downstream +1", 7: "downstream +2", 8: "downstream +3",
}
DOWNSTREAM_POSITIONS = [6, 7, 8]

results = {}   # check name -> bool


def banner(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def fmt(vec):
    return "[" + " ".join(f"{v:6.3f}" for v in vec) + "]"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Config 1 baseline integrity
# ─────────────────────────────────────────────────────────────────────────────
banner("STEP 0 — Config 1 reproduces the original encoding exactly")


def original_encode_then_slice(seq):
    """
    Verbatim transcription of the ORIGINAL qml_classifier.py encoding
    (ints * pi/3) followed by its x[:, :N_QUBITS] slice, with N_QUBITS = 4.

    This is the historical reference and is deliberately kept literal here — the
    pipeline has since moved this logic into src/qml_encodings.py, so this function
    is what proves the move did not change anything.
    """
    _NUC_MAP = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    ints = np.array([_NUC_MAP[c] for c in seq], dtype=np.float32)
    angles = ints * (np.pi / 3.0)
    return angles[:4]


# The encoding now lives in src/qml_encodings.py; qml_classifier.py imports it rather
# than defining NUC_MAP inline, so the source check targets the module that owns it.
enc_src = (ROOT / "src" / "qml_encodings.py").read_text()
src_map = re.search(r"^NUC_MAP\s*=\s*(\{[^}]*\})", enc_src, re.M)
src_scale = re.search(r"SINGLE_SCALE\s*=\s*np\.pi\s*/\s*\(N_STATES\s*-\s*1\)", enc_src)

print(f"  src/qml_encodings.py NUC_MAP : {src_map.group(1) if src_map else 'NOT FOUND'}")
print(f"  imported NUC_MAP             : {NUC_MAP}")
print(f"  pi/(n_states-1) scaling      : {bool(src_scale)}")
print(f"  n_qubits_for('config1')      : {n_qubits_for('config1')}")

source_ok = (
    src_map is not None
    and eval(src_map.group(1)) == NUC_MAP
    and src_scale is not None
    and n_qubits_for("config1") == 4
)

df = pd.read_csv(TRAIN_PATH, usecols=["sequence"])
all_seqs = df["sequence"].astype(str).str.upper().tolist()
print(f"\n  Loaded {len(all_seqs)} training sequences from {TRAIN_PATH}")

numeric_ok = all(
    np.allclose(encode_sequence(s, "config1"), original_encode_then_slice(s), atol=0, rtol=0)
    for s in all_seqs
)
print(f"  Config 1 matches original on all {len(all_seqs)} sequences (exact): {numeric_ok}")

results["Config 1 baseline integrity"] = bool(source_ok and numeric_ok)
print(f"\n  [{'PASS' if results['Config 1 baseline integrity'] else 'FAIL'}] "
      "Config 1 pathway is unchanged and faithfully reproduced")


# ─────────────────────────────────────────────────────────────────────────────
# Example sequences — real training rows
# ─────────────────────────────────────────────────────────────────────────────
banner("EXAMPLE SEQUENCES (all real rows from the training split)")

PAIRS = [
    ("ATGCGGAC", "ATGCGTAC", 6),
    ("GAACGGAC", "GAACGGGC", 7),
    ("GAGCGGTA", "GAGCGGTC", 8),
]

seq_set = set(all_seqs)
examples = []
for a, b, pos in PAIRS:
    examples.extend([a, b])
    assert a in seq_set and b in seq_set, f"{a}/{b} not in training data"
    diffs = [i + 1 for i in range(8) if a[i] != b[i]]
    print(f"  {a}  vs  {b}   differ only at position {diffs}  (expected [{pos}])")

print(f"\n  {len(examples)} example sequences, all confirmed present in {TRAIN_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side encodings
# ─────────────────────────────────────────────────────────────────────────────
banner("SIDE-BY-SIDE ENCODINGS")

print("  Motif layout: N N N C G N N N")
for p, role in POSITION_ROLES.items():
    print(f"    position {p} = {role}")

print("\n  Qubit -> source positions:")
for cfg in CONFIGS:
    mapping = ", ".join(
        f"q{q}<-{'+'.join(str(p) for p in positions_feeding_qubit(cfg, q))}"
        for q in range(n_qubits_for(cfg))
    )
    print(f"    {cfg} ({n_qubits_for(cfg)} qubits): {mapping}")

for s in examples:
    print(f"\n  sequence: {' '.join(s)}")
    print(f"    ints        : {[NUC_MAP[c] for c in s]}")
    for cfg in CONFIGS:
        print(f"    {cfg:<8} : {fmt(encode_sequence(s, cfg))}   ({CONFIG_DESCRIPTIONS[cfg]})")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK (a) — do all 8 positions influence the representation?
# ─────────────────────────────────────────────────────────────────────────────
banner("CHECK (a) — all 8 positions measurably influence the representation")

print("  Method: for every position, substitute each of the other three bases and")
print("  test whether the encoded vector changes. Averaged over the example sequences.")
print("  This probes the ENCODING FUNCTION, so the constant CpG positions are exercised")
print("  too (they never vary in the dataset, but the function must still respond).\n")

print(f"  {'position':<10} {'role':<16} {'config1':>10} {'config2':>10} {'config3':>10}")
print("  " + "-" * 60)

influence = {cfg: {} for cfg in CONFIGS}
for pos in range(1, 9):
    row = {}
    for cfg in CONFIGS:
        changed = False
        for s in examples:
            base = encode_sequence(s, cfg)
            for b in "ACGT":
                if b == s[pos - 1]:
                    continue
                mutant = s[:pos - 1] + b + s[pos:]
                if not np.array_equal(base, encode_sequence(mutant, cfg)):
                    changed = True
                    break
            if changed:
                break
        row[cfg] = changed
        influence[cfg][pos] = changed
    print(f"  {pos:<10} {POSITION_ROLES[pos]:<16} "
          + "".join(f"{'YES' if row[c] else 'no':>10}" for c in CONFIGS))

a_ok = all(influence["config2"][p] for p in range(1, 9)) and \
       all(influence["config3"][p] for p in range(1, 9))
results["(a) all 8 positions influence Config 2 and Config 3"] = a_ok

c1_live = [p for p in range(1, 9) if influence["config1"][p]]
print(f"\n  Config 1 responds to positions {c1_live} only — positions 5-8 are sliced away.")
print("  Note: position 4 is constant (C) in every dataset sequence, so although the")
print("  Config 1 function responds to it, it carries zero information in practice —")
print("  Config 1's 4 qubits encode only 3 varying positions.")
print(f"\n  [{'PASS' if a_ok else 'FAIL'}] all 8 positions influence Config 2 and Config 3")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK (b) — are downstream +1/+2/+3 present in the circuit input?
# ─────────────────────────────────────────────────────────────────────────────
banner("CHECK (b) — downstream +1/+2/+3 present in the circuit input")

b_ok = True
for cfg in CONFIGS:
    reachable = set()
    for q in range(n_qubits_for(cfg)):
        reachable.update(positions_feeding_qubit(cfg, q))
    present = {p: (p in reachable) for p in DOWNSTREAM_POSITIONS}
    where = {
        p: [q for q in range(n_qubits_for(cfg)) if p in positions_feeding_qubit(cfg, q)]
        for p in DOWNSTREAM_POSITIONS
    }
    ok = all(present.values())
    if cfg != "config1":
        b_ok = b_ok and ok
    print(f"  {cfg}:")
    for p in DOWNSTREAM_POSITIONS:
        loc = f"qubit {where[p][0]}" if where[p] else "ABSENT"
        print(f"    position {p} ({POSITION_ROLES[p]:<14}) -> {loc}")
    print(f"    -> {'all present' if ok else 'MISSING downstream positions'}\n")

results["(b) downstream +1/+2/+3 present in Config 2 and Config 3"] = b_ok
print(f"  [{'PASS' if b_ok else 'FAIL'}] downstream positions present in Config 2 and Config 3")
print("  (Config 1 is expected to be missing them — that is the defect under investigation.)")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK (c) — does a single downstream change alter the circuit input?
# ─────────────────────────────────────────────────────────────────────────────
banner("CHECK (c) — a single downstream nucleotide change alters the circuit input")

c_ok = True
for a, b, pos in PAIRS:
    print(f"\n  {a} -> {b}   (only position {pos}, {POSITION_ROLES[pos]}, changes: "
          f"{a[pos-1]} -> {b[pos-1]})")
    for cfg in CONFIGS:
        va, vb = encode_sequence(a, cfg), encode_sequence(b, cfg)
        diff_q = [i for i in range(len(va)) if not np.isclose(va[i], vb[i])]
        changed = len(diff_q) > 0
        if cfg != "config1":
            c_ok = c_ok and changed
        tag = "CHANGED" if changed else "IDENTICAL"
        print(f"    {cfg:<8} {fmt(va)} -> {fmt(vb)}  {tag}"
              + (f" at qubit {diff_q}" if changed else " (downstream base invisible)"))

results["(c) downstream change alters Config 2 and Config 3 input"] = c_ok
print(f"\n  [{'PASS' if c_ok else 'FAIL'}] downstream change alters Config 2 and Config 3")
print("  Config 1 is IDENTICAL in every case — it cannot distinguish these sequences at all.")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK (d) — code ordering vs manuscript claim
# ─────────────────────────────────────────────────────────────────────────────
banner("CHECK (d) — nucleotide ordering: code vs manuscript")

print(f"  Code (qml_classifier.py:43 and src/qml_encodings.py): {NUC_MAP}")
print(f"  Manuscript Section III-B claims                     : {MANUSCRIPT_NUC_MAP}")

mismatched = sorted(k for k in NUC_MAP if NUC_MAP[k] != MANUSCRIPT_NUC_MAP[k])
d_ok = not mismatched
results["(d) code ordering matches manuscript"] = d_ok

print()
print(f"  {'base':<6} {'code':>6} {'manuscript':>12}   {'angle (code)':>14} {'angle (manuscript)':>20}")
print("  " + "-" * 64)
for base in "ACGT":
    flag = "  <-- MISMATCH" if base in mismatched else ""
    print(f"  {base:<6} {NUC_MAP[base]:>6} {MANUSCRIPT_NUC_MAP[base]:>12}   "
          f"{NUC_MAP[base]*np.pi/3:>14.4f} {MANUSCRIPT_NUC_MAP[base]*np.pi/3:>20.4f}{flag}")

if mismatched:
    print(f"\n  [MISMATCH] bases {mismatched} disagree between code and manuscript.")
    print("  The manuscript's A->0, C->1, G->2, T->3 is exactly the index order of")
    print("  BASES = \"ACGT\" at src/enzyme_common.py:9, which the CLASSICAL one-hot")
    print("  models use. The manuscript appears to describe the classical ordering")
    print("  rather than the QML path's. Not reconciled here — reported only.")
print(f"\n  [{'PASS' if d_ok else 'FAIL'}] code ordering matches manuscript")


# ─────────────────────────────────────────────────────────────────────────────
# Config 2 bijectivity (supports the lossless claim)
# ─────────────────────────────────────────────────────────────────────────────
banner("SUPPLEMENTARY — Config 2 pair condensation is lossless")

pair_angles = {}
for b1 in "ATGC":
    for b2 in "ATGC":
        v = NUC_MAP[b1] * 4 + NUC_MAP[b2]
        pair_angles[b1 + b2] = v * (np.pi / 15)
uniq = len(set(np.round(list(pair_angles.values()), 12)))
bijective = uniq == 16
results["Config 2 condensation is bijective (no collisions)"] = bijective

print(f"  16 possible pairs -> {uniq} distinct angles")
print("  " + "  ".join(f"{k}:{v:.3f}" for k, v in list(pair_angles.items())[:8]))
print("  " + "  ".join(f"{k}:{v:.3f}" for k, v in list(pair_angles.items())[8:]))
print(f"\n  [{'PASS' if bijective else 'FAIL'}] no two pairs collide — condensation is information-preserving")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
banner("SUMMARY")

for name, ok in results.items():
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")

expected_fail = "(d) code ordering matches manuscript"
hard = {k: v for k, v in results.items() if k != expected_fail}

print()
if all(hard.values()):
    print("  All encoding checks PASSED.")
else:
    print("  One or more encoding checks FAILED — see above.")
if not results[expected_fail]:
    print("  Check (d) FAILED as expected: the manuscript's stated nucleotide ordering")
    print("  does not match the code. This needs a decision before the paper is submitted.")

print("\n  No training was run. Encoding verification only, as requested.")
