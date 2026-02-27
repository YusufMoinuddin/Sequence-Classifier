# src/qm_features.py
import numpy as np
import pandas as pd
from typing import List

# RAW QM columns that exist in your non-scaled CSV + your split files
QM_COLUMNS: List[str] = [
    "nuclear repulsion energy",
    "scf total energy",
    "maximum gradient",
    "rms gradient",
]

def extract_qm_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extract QM feature matrix (N, 4) as float32.
    Coerces non-numeric values to NaN.
    """
    missing = [c for c in QM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing QM columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    mat = df[QM_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return mat