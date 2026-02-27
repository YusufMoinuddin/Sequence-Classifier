# enzyme_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Iterable
from sklearn.metrics import roc_auc_score  # for AUROC

from models.cnn_classifier import SeqCNN
from models.transformer_classifier import DNATransformerClassifier

BASES = "ACGT"


def one_hot_seq(seq: str, expected_len: int = 8) -> np.ndarray:
    """One-hot encode an 8-mer (NNNCGNNN). Returns array (4, L)."""
    seq = (seq or "").upper().strip()
    if len(seq) != expected_len:
        raise ValueError(f"Expected {expected_len}-mer, got len={len(seq)}: {seq!r}")
    arr = np.zeros((4, expected_len), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in BASES:
            arr[BASES.index(b), i] = 1.0
        else:
            # Handle N/other by uniform distribution
            arr[:, i] = 0.25
    return arr


class EnzDataset(Dataset):
    """Expects CSV with columns: sequence,label or sequence,meth_A,meth_B."""
    def __init__(self, df: pd.DataFrame, expected_len: int = 8):
        if "sequence" in df.columns:
            seq_col = "sequence"
        elif "motif" in df.columns:
            seq_col = "motif"
        elif "seq" in df.columns:
            seq_col = "seq"
        else:
            raise ValueError("DataFrame must have a 'sequence', 'motif', or 'seq' column.")

        self.expected_len = expected_len
        self.x = np.stack([one_hot_seq(s, expected_len=expected_len) for s in df[seq_col]])

        # map string labels to 0/1 if needed
        if "label" in df.columns and df["label"].dtype == object:
            lab_map = {"DNMT3A": 0.0, "DNMT3B": 1.0}
            self.y = df["label"].map(lab_map).astype(np.float32).values
        elif "label" in df.columns:
            self.y = df["label"].astype(np.float32).values
        elif {"meth_A", "meth_B"}.issubset(df.columns):
            diff = df["meth_B"].astype(float) - df["meth_A"].astype(float)
            self.y = (diff > 0).astype(np.float32).values
        else:
            raise ValueError("Need a 'label' column or meth_A/meth_B to derive labels.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # x: (4, L)
        return (
            torch.tensor(self.x[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.float32),
        )


@torch.no_grad()
def eval_accuracy_and_confmat(
    model,
    loader,
    loss_fn,
    device: str = "cpu",
    model_type: str = "cnn",
):
    """
    Evaluate model on a DataLoader and compute:
      - average loss
      - accuracy
      - confusion matrix
      - AUROC (if both classes present; otherwise NaN)
      - F1-score (if denominator > 0; otherwise 0.0)

    model_type: "cnn" or "transformer"
    """
    model.eval()
    total_loss, n = 0.0, 0
    tp = tn = fp = fn = 0

    all_probs = []
    all_labels = []

    for xb, yb in loader:
        xb = xb.to(device)  # (B,4,L)
        yb = yb.to(device)

        if model_type == "cnn":
            logits = model(xb)  # (B,)
        elif model_type == "transformer":
            # Convert one-hot (B,4,L) -> token ids (B,L) by argmax over channels
            token_ids = xb.argmax(dim=1).long()  # (B,L)
            logits = model(token_ids)            # (B,)
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

        all_probs.append(probs.cpu().numpy())
        all_labels.append(yb.cpu().numpy())

    avg_loss = total_loss / max(1, n)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    conf = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

    # --- AUROC ---
    if all_probs:
        probs_concat = np.concatenate(all_probs)
        labels_concat = np.concatenate(all_labels)
        if len(np.unique(labels_concat)) == 2:
            auc = roc_auc_score(labels_concat, probs_concat)
        else:
            auc = float("nan")
    else:
        auc = float("nan")

    # --- F1-score ---
    prec_den = tp + fp
    rec_den = tp + fn

    if prec_den > 0:
        precision = tp / prec_den
    else:
        precision = 0.0

    if rec_den > 0:
        recall = tp / rec_den
    else:
        recall = 0.0

    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return avg_loss, acc, conf, auc, f1


def train_model(
    train_csv: str = "deep_enzymology_train.csv",
    val_csv: str = "deep_enzymology_val.csv",
    test_csv: str = "deep_enzymology_test.csv",
    model_path: str = "enzyme_model.pt",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    model_type: str = "cnn",   # "cnn" or "transformer"
    seq_len: int = 8,
):
    """
    Train either the CNN or Transformer sequence classifier.

    model_type:
        - "cnn": uses SeqCNN and one-hot input (B,4,L)
        - "transformer": uses DNATransformerClassifier and token ids (B,L)
    """

    # HISTORY TRACKING
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "val_f1": [],
    }

    # Load splits
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Datasets & loaders
    train_loader = DataLoader(
        EnzDataset(train_df, expected_len=seq_len),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        EnzDataset(val_df, expected_len=seq_len),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        EnzDataset(test_df, expected_len=seq_len),
        batch_size=batch_size,
        shuffle=False,
    )

    # Build model
    if model_type == "cnn":
        model = SeqCNN().to(device)
    elif model_type == "transformer":
        model = DNATransformerClassifier(
            vocab_size=len(BASES),
            d_model=128,
            n_heads=4,
            num_layers=2,
            dim_feedforward=256,
            max_seq_len=seq_len,
            dropout=0.1,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)  # (B,4,L)
            yb = yb.to(device)

            opt.zero_grad()

            if model_type == "cnn":
                logits = model(xb)
            elif model_type == "transformer":
                token_ids = xb.argmax(dim=1).long()  # (B,L)
                logits = model(token_ids)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total += loss.item() * len(yb)
            n += len(yb)

        train_loss = total / max(1, n)
        val_loss, val_acc, val_conf, val_auc, val_f1 = eval_accuracy_and_confmat(
            model,
            val_loader,
            loss_fn,
            device=device,
            model_type=model_type,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | "
            f"val_auc={val_auc:.3f} | val_f1={val_f1:.3f} | CM={val_conf}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model with val_loss={best_val_loss:.4f}")

    # Final test metrics
    test_loss, test_acc, test_conf, test_auc, test_f1 = eval_accuracy_and_confmat(
        model,
        test_loader,
        loss_fn,
        device=device,
        model_type=model_type,
    )
    print(
        f"TEST | loss={test_loss:.4f} | acc={test_acc:.3f} | "
        f"auc={test_auc:.3f} | f1={test_f1:.3f} | CM={test_conf}"
    )

    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")

    return model, history


def predict_enzyme(
    seqs: Iterable[str],
    model_path: str = "enzyme_model.pt",
    device: str = "cpu",
    model_type: str = "cnn",
    seq_len: int = 8,
):
    """
    Simple prediction helper.

    model_type:
        - "cnn": loads SeqCNN and uses one-hot input
        - "transformer": loads DNATransformerClassifier and uses token ids
    """
    if model_type == "cnn":
        model = SeqCNN().to(device)
    elif model_type == "transformer":
        model = DNATransformerClassifier(
            vocab_size=len(BASES),
            d_model=128,
            n_heads=4,
            num_layers=2,
            dim_feedforward=256,
            max_seq_len=seq_len,
            dropout=0.1,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    seqs = list(seqs)

    if model_type == "cnn":
        X = torch.tensor(
            np.stack([one_hot_seq(s, expected_len=seq_len) for s in seqs]),
            dtype=torch.float32,
        ).to(device)  # (B,4,L)
        with torch.no_grad():
            probs = torch.sigmoid(model(X)).cpu().numpy()
    else:
        # Build token ids directly
        token_ids_list = []
        for s in seqs:
            s = (s or "").upper().strip()
            if len(s) != seq_len:
                raise ValueError(f"Expected sequence of length {seq_len}, got {len(s)}: {s!r}")
            token_ids = [BASES.index(b) if b in BASES else 0 for b in s]
            token_ids_list.append(token_ids)
        X = torch.tensor(token_ids_list, dtype=torch.long).to(device)  # (B,L)
        with torch.no_grad():
            probs = torch.sigmoid(model(X)).cpu().numpy()

    for s, p in zip(seqs, probs):
        pred = "DNMT3B" if p >= 0.5 else "DNMT3A"
        print(f"{s}: {pred}  (P(DNMT3B)={p:.3f})")

    return probs


if __name__ == "__main__":
    # Default: train CNN baseline
    model, history = train_model(
        train_csv="deep_enzymology_train.csv",
        val_csv="deep_enzymology_val.csv",
        test_csv="deep_enzymology_test.csv",
        model_path="enzyme_cnn.pt",
        epochs=10,
        batch_size=64,
        lr=1e-3,
        device="cpu",
        model_type="transformer",    # change to "transformer" to train the transformer
        seq_len=8,
    )
