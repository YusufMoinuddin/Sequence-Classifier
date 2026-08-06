# src/enzyme_common.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.metrics import roc_auc_score

BASES = "ACGT"

from src.qm_features import QM_COLUMNS, fit_qm_scaler, apply_qm_scaler


def one_hot_seq(seq: str, expected_len: int = 8) -> np.ndarray:
    """One-hot encode an L-mer. Returns array (4, L)."""
    seq = (seq or "").upper().strip()
    if len(seq) != expected_len:
        raise ValueError(f"Expected {expected_len}-mer, got len={len(seq)}: {seq!r}")
    arr = np.zeros((4, expected_len), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in BASES:
            arr[BASES.index(b), i] = 1.0
        else:
            arr[:, i] = 0.25
    return arr


class EnzDataset(Dataset):
    """
    If use_qm=False: returns (x_onehot, y)
    If use_qm=True : returns (x_onehot, qm_feats, y)

    QM handling:
      - If qm_scaled is provided: uses pre-scaled values
      - Else: leaves raw qm values as-is
    """
    def __init__(
        self,
        df: pd.DataFrame,
        expected_len: int = 8,
        use_qm: bool = False,
        qm_scaled: Optional[np.ndarray] = None,
    ):
        # --- sequence column detection ---
        if "sequence" in df.columns:
            seq_col = "sequence"
        elif "motif" in df.columns:
            seq_col = "motif"
        elif "seq" in df.columns:
            seq_col = "seq"
        else:
            raise ValueError("DataFrame must have a 'sequence', 'motif', or 'seq' column.")

        self.use_qm = use_qm
        self.expected_len = expected_len

        seqs = df[seq_col].astype(str).tolist()
        x = np.stack([one_hot_seq(s, expected_len=expected_len) for s in seqs])  # (N,4,L)

        # --- labels ---
        if "label" in df.columns:
            if df["label"].dtype == object:
                lab_map = {"DNMT3A": 0.0, "DNMT3B": 1.0}
                y = df["label"].map(lab_map).astype(np.float32).values
            else:
                y = df["label"].astype(np.float32).values
        elif "label_str" in df.columns:
            lab_map = {"DNMT3A": 0.0, "DNMT3B": 1.0}
            y = df["label_str"].map(lab_map).astype(np.float32).values
        elif {"meth_A", "meth_B"}.issubset(df.columns):
            diff = df["meth_B"].astype(float) - df["meth_A"].astype(float)
            y = (diff > 0).astype(np.float32).values
        else:
            raise ValueError("Need 'label'/'label_str' or meth_A/meth_B to derive labels.")

        self.seqs = seqs
        self.x = x
        self.y = y
        self.qm = None

        if self.use_qm:
            missing = [c for c in QM_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(
                    f"use_qm=True but missing QM columns: {missing}\n"
                    f"Available columns: {list(df.columns)}"
                )

            # Pull QM matrix (or use pre-scaled matrix from train-fitted scaler)
            if qm_scaled is not None:
                qm_vals = np.asarray(qm_scaled, dtype=np.float32)
                if qm_vals.shape != (len(df), len(QM_COLUMNS)):
                    raise ValueError(
                        f"qm_scaled must have shape ({len(df)}, {len(QM_COLUMNS)}), got {qm_vals.shape}."
                    )
            else:
                qm_vals = df[QM_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

            # Require finite values (now that you're using real raw QM, this should pass)
            finite_mask = np.isfinite(qm_vals).all(axis=1)
            n_total = len(df)
            n_good = int(finite_mask.sum())
            n_bad = n_total - n_good
            print(f"[EnzDataset] QM rows total={n_total} good={n_good} bad={n_bad}")

            if n_good == 0:
                raise ValueError(
                    "All QM rows are NaN/Inf. Check the QM columns in your CSV or qm_features.QM_COLUMNS."
                )

            if n_bad > 0:
                print(f"[EnzDataset] Dropping {n_bad} rows with NaN/Inf QM features.")
                qm_vals = qm_vals[finite_mask]
                self.x = self.x[finite_mask]
                self.y = self.y[finite_mask]
                self.seqs = list(np.array(self.seqs, dtype=object)[finite_mask])

            self.qm = qm_vals.astype(np.float32)

        if len(self.y) == 0:
            raise ValueError("Dataset ended up empty after filtering. Check your splits/QM columns.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x_i = torch.tensor(self.x[i], dtype=torch.float32)  # (4,L)
        y_i = torch.tensor(self.y[i], dtype=torch.float32)

        if self.use_qm:
            qm_i = torch.tensor(self.qm[i], dtype=torch.float32)  # (qm_dim,)
            return x_i, qm_i, y_i

        return x_i, y_i


def load_splits(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    seq_len: int = 8,
    batch_size: int = 64,
    use_qm: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    qm_train_scaled = None
    qm_val_scaled = None
    qm_test_scaled = None
    if use_qm:
        scaler = fit_qm_scaler(train_df)
        qm_train_scaled = apply_qm_scaler(train_df, scaler)
        qm_val_scaled = apply_qm_scaler(val_df, scaler)
        qm_test_scaled = apply_qm_scaler(test_df, scaler)

    train_ds = EnzDataset(train_df, expected_len=seq_len, use_qm=use_qm, qm_scaled=qm_train_scaled)
    val_ds = EnzDataset(val_df, expected_len=seq_len, use_qm=use_qm, qm_scaled=qm_val_scaled)
    test_ds = EnzDataset(test_df, expected_len=seq_len, use_qm=use_qm, qm_scaled=qm_test_scaled)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


@torch.no_grad()
def eval_accuracy_and_confmat(model, loader, loss_fn, device="cpu", model_type="cnn", use_qm: bool = False):
    model.eval()
    total_loss, n = 0.0, 0
    tp = tn = fp = fn = 0
    all_probs, all_labels = [], []

    for batch in loader:
        if use_qm:
            xb, qm, yb = batch
            qm = qm.to(device)
        else:
            xb, yb = batch
            qm = None

        xb = xb.to(device)
        yb = yb.to(device)

        if model_type == "cnn":
            logits = model(xb, qm) if use_qm else model(xb)
        elif model_type == "transformer":
            token_ids = xb.argmax(dim=1).long()
            logits = model(token_ids, qm_feats=qm) if use_qm else model(token_ids)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        loss = loss_fn(logits, yb)
        total_loss += loss.item() * len(yb)
        n += len(yb)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        tp += ((preds == 1) & (yb == 1)).sum().item()
        tn += ((preds == 0) & (yb == 0)).sum().item()
        fp += ((preds == 1) & (yb == 0)).sum().item()
        fn += ((preds == 0) & (yb == 1)).sum().item()

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(yb.detach().cpu().numpy())

    avg_loss = total_loss / max(1, n)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    conf = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

    if all_probs:
        probs_concat = np.concatenate(all_probs)
        labels_concat = np.concatenate(all_labels)
        auc = roc_auc_score(labels_concat, probs_concat) if len(np.unique(labels_concat)) == 2 else float("nan")
    else:
        auc = float("nan")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return avg_loss, acc, conf, auc, f1


# ─────────────────────────────────────────────────────────────────────────────
# Classical-baseline helpers (added for the SVM / RF / CNN benchmark).
# These are additive: nothing above this line is modified.
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

# The only two columns the classical baselines are allowed to see. Everything
# else in the qmproxy CSVs is either a QM-proxy feature (out of scope for this
# comparison) or label-derived and therefore leaky: meth_A / meth_B / total are
# the raw methylation counts the label is computed from, label_str is a 1:1
# alias of label, and key embeds label_str verbatim.
SEQ_LABEL_COLUMNS = ["sequence", "label"]


def load_seq_label(csv_path: str, expected_len: int = 8):
    """
    Load a split using ONLY the 'sequence' and 'label' columns.

    Every other column (QM-proxy energies/gradients, and the label-derived
    meth_A/meth_B/total/label_str/key) is dropped before any encoding happens,
    so it cannot leak into the features.

    Returns:
        x       : (N, 4, L) float32 one-hot, via the existing one_hot_seq()
        y       : (N,) int64 labels in {0, 1}
        n_neg   : count of DNMT3A (label 0)
        n_pos   : count of DNMT3B (label 1)
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in SEQ_LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df = df[SEQ_LABEL_COLUMNS].copy()

    if df["sequence"].isna().any() or df["label"].isna().any():
        raise ValueError(f"{csv_path} contains NaNs in 'sequence' or 'label'.")

    seqs = df["sequence"].astype(str).tolist()
    x = np.stack([one_hot_seq(s, expected_len=expected_len) for s in seqs]).astype(np.float32)

    y = df["label"].to_numpy()
    if y.dtype == object:
        y = pd.Series(y).map({"DNMT3A": 0, "DNMT3B": 1}).to_numpy()
    y = y.astype(np.int64)

    bad = set(np.unique(y)) - {0, 1}
    if bad:
        raise ValueError(f"{csv_path} has unexpected label values: {sorted(bad)}")

    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    return x, y, n_neg, n_pos


def flatten_onehot(x: np.ndarray) -> np.ndarray:
    """(N, 4, L) one-hot -> (N, 4*L) flat features for sklearn estimators."""
    return x.reshape(len(x), -1)


def full_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """
    Full metric suite for a binary classifier, computed exactly the way
    evaluate_metrics.py computes them for the VQC, so classical and quantum
    numbers are directly comparable.

    Positive class (1) is DNMT3B.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    both_classes = len(np.unique(y_true)) == 2
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob) if both_classes else float("nan"),
        "pr_auc": average_precision_score(y_true, y_prob) if both_classes else float("nan"),
        "precision_dnmt3b": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_dnmt3b": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_dnmt3b": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }