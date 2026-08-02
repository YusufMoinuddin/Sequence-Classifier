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