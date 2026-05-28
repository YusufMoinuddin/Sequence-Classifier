# src/qm_features.py
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import StandardScaler

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


def fit_qm_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """
    Fit StandardScaler on TRAIN split only using finite QM rows.
    """
    train_mat = extract_qm_matrix(train_df)
    finite_mask = np.isfinite(train_mat).all(axis=1)
    if not finite_mask.any():
        raise ValueError("No finite QM rows in train split; cannot fit StandardScaler.")
    scaler = StandardScaler()
    scaler.fit(train_mat[finite_mask])
    return scaler


def apply_qm_scaler(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """
    Transform QM features with an already-fitted scaler.
    Keeps NaN/Inf rows untouched so downstream filtering can remove them.
    """
    mat = extract_qm_matrix(df)
    finite_mask = np.isfinite(mat).all(axis=1)
    scaled = mat.copy()
    if finite_mask.any():
        scaled[finite_mask] = scaler.transform(mat[finite_mask]).astype(np.float32)
    return scaled.astype(np.float32)


def qm_variance_report(df: pd.DataFrame) -> Dict[str, float]:
    """
    Return per-feature variance to detect near-constant QM features.
    """
    mat = extract_qm_matrix(df)
    finite_mask = np.isfinite(mat).all(axis=1)
    if not finite_mask.any():
        raise ValueError("No finite QM rows available for variance report.")
    finite_mat = mat[finite_mask]
    variances = np.var(finite_mat, axis=0, dtype=np.float64)
    return {col: float(var) for col, var in zip(QM_COLUMNS, variances)}