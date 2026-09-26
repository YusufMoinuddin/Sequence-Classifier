# src/qml_encodings.py
"""
Nucleotide angle-encoding schemes for the VQC.

Three configurations, all selectable through the ENCODINGS registry so the eventual
training runs can switch with a flag. This module is additive: qml_classifier.py is
NOT modified, and Config 1 here is a faithful re-implementation of what it already
does, kept so the three schemes can be compared side by side.

  CONFIG 1 — "config1" : current behaviour. One angle per nucleotide (pi/3 scaling),
                         then a plain slice of the first 4 positions. Positions 5-8
                         are computed and discarded.
                         Mirrors qml_classifier.py:53-56 + qml_classifier.py:178.
                         4 qubits.

  CONFIG 2 — "config2" : nucleotide-pair condensation. Adjacent positions are paired
                         (1,2)(3,4)(5,6)(7,8) and each pair is condensed into a single
                         angle by reading it as a two-digit base-4 number:
                             combined = v1 * 4 + v2      # 0 .. 15
                             angle    = combined * pi/15 # 0 .. pi
                         Bijective: all 16 pairs map to 16 distinct angles, so no
                         information is lost. All 8 positions reach the circuit.
                         4 qubits.

  CONFIG 3 — "config3" : one angle per nucleotide for all 8 positions, no slicing.
                         Same pi/3 per-nucleotide scaling as Config 1
                         (4 states -> pi/(4-1)), so the scaling convention is
                         unchanged; only the qubit count differs.
                         8 qubits.

The motif is NNNCGNNN: positions 4 and 5 are the constant CpG site across the whole
dataset, positions 1-3 are the upstream flank (-3,-2,-1) and positions 6-8 are the
downstream flank (+1,+2,+3).

NOTE on ordinal mapping: NUC_MAP below is the mapping the code actually uses and is
copied from qml_classifier.py:43. The manuscript (Section III-B) instead claims
A->0, C->1, G->2, T->3, which swaps T and C. That mismatch is reported by
verify_encodings.py rather than silently reconciled here.
"""

import numpy as np

# Mapping from nucleotide character to integer — identical to qml_classifier.py:43
NUC_MAP = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

SEQ_LEN = 8
N_STATES = 4                      # A/T/G/C
SINGLE_SCALE = np.pi / (N_STATES - 1)          # pi/3  -> single nucleotide, 0..pi
PAIR_SCALE = np.pi / (N_STATES ** 2 - 1)       # pi/15 -> condensed pair,   0..pi

# Adjacent pairing used by Config 2, as 0-indexed position pairs.
# (1,2)(3,4)(5,6)(7,8) in 1-indexed terms.
CONFIG2_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))


def _to_ints(seq: str) -> np.ndarray:
    """Validate an 8-mer and map it to integers via NUC_MAP."""
    seq = (seq or "").upper().strip()
    if len(seq) != SEQ_LEN:
        raise ValueError(f"Expected {SEQ_LEN}-mer, got len={len(seq)}: {seq!r}")
    bad = [c for c in seq if c not in NUC_MAP]
    if bad:
        raise ValueError(f"Unknown nucleotide(s) {bad} in {seq!r}; expected A/T/G/C.")
    return np.array([NUC_MAP[c] for c in seq], dtype=np.float32)


def encode_config1_slice4(seq: str) -> np.ndarray:
    """
    CONFIG 1 — current behaviour, unchanged.

    All 8 nucleotides are encoded at pi/3, then only the first 4 are kept. This
    reproduces qml_classifier.py's encode_sequence() followed by the x[:, :N_QUBITS]
    slice at qml_classifier.py:178.

    Returns (4,) float32 — positions 1,2,3,4.
    """
    angles = _to_ints(seq) * SINGLE_SCALE   # all 8, exactly as qml_classifier.py does
    return angles[:4].astype(np.float32)    # positions 5-8 discarded here


def encode_config2_pairs(seq: str) -> np.ndarray:
    """
    CONFIG 2 — nucleotide-pair condensation.

    Adjacent positions (1,2)(3,4)(5,6)(7,8) are each condensed into one angle by
    treating the pair as a two-digit base-4 number:

        combined = v1 * 4 + v2        # 0 .. 15
        angle    = combined * pi/15   # 0 .. pi

    Bijective, so all 16 possible pairs give 16 distinct angles and nothing collides.
    Every one of the 8 positions influences the result.

    Returns (4,) float32 — one angle per pair.
    """
    ints = _to_ints(seq)
    angles = np.empty(len(CONFIG2_PAIRS), dtype=np.float32)
    for k, (i, j) in enumerate(CONFIG2_PAIRS):
        combined = ints[i] * N_STATES + ints[j]     # base-4 two-digit value, 0..15
        angles[k] = combined * PAIR_SCALE
    return angles


def encode_config3_full8(seq: str) -> np.ndarray:
    """
    CONFIG 3 — one angle per nucleotide, all 8 positions, nothing discarded.

    Identical per-nucleotide scaling to Config 1 (pi/3); the only difference is that
    all 8 angles are kept and the circuit needs 8 qubits.

    Returns (8,) float32.
    """
    return (_to_ints(seq) * SINGLE_SCALE).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Registry — the seam for selecting a config by flag in the eventual training runs.
# ─────────────────────────────────────────────────────────────────────────────

ENCODINGS = {
    "config1": encode_config1_slice4,
    "config2": encode_config2_pairs,
    "config3": encode_config3_full8,
}

N_QUBITS_FOR = {
    "config1": 4,
    "config2": 4,
    "config3": 8,
}

CONFIG_DESCRIPTIONS = {
    "config1": "current 4-position slice (positions 1-4; 5-8 discarded)",
    "config2": "pair condensation, base-4, adjacent pairs (1,2)(3,4)(5,6)(7,8)",
    "config3": "full 8-position, one angle per nucleotide",
}


def encode_sequence(seq: str, config: str = "config1") -> np.ndarray:
    """Encode an 8-mer under the named config. Default preserves current behaviour."""
    if config not in ENCODINGS:
        raise ValueError(f"Unknown config {config!r}; expected one of {sorted(ENCODINGS)}.")
    return ENCODINGS[config](seq)


def n_qubits_for(config: str) -> int:
    """Number of qubits the named config requires."""
    if config not in N_QUBITS_FOR:
        raise ValueError(f"Unknown config {config!r}; expected one of {sorted(N_QUBITS_FOR)}.")
    return N_QUBITS_FOR[config]


def positions_feeding_qubit(config: str, qubit: int):
    """
    Which 1-indexed sequence positions feed a given qubit under a config.
    Used by the verification script to prove downstream positions are present.
    """
    if config == "config1":
        return (qubit + 1,) if qubit < 4 else ()
    if config == "config2":
        i, j = CONFIG2_PAIRS[qubit]
        return (i + 1, j + 1)
    if config == "config3":
        return (qubit + 1,)
    raise ValueError(f"Unknown config {config!r}.")
