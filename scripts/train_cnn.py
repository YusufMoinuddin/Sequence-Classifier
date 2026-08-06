import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# scripts/train_cnn.py
#
# Single-run debugging entry point for the 1D CNN baseline.
# Paper numbers come from benchmark_classical.py (5 seeds, mean +/- std).
import numpy as np
import torch
import torch.nn as nn

from src.enzyme_common import load_splits, eval_accuracy_and_confmat, full_binary_metrics
from models.cnn_classifier import SeqCNN

TRAIN_PATH = "data/deep_enzymology_qmproxy_train.csv"
VAL_PATH = "data/deep_enzymology_qmproxy_val.csv"
TEST_PATH = "data/deep_enzymology_qmproxy_test.csv"

SEED = 0
EPOCHS = 200
PATIENCE = 20


@torch.no_grad()
def full_metrics_for_loader(model, loader, device="cpu"):
    """Balanced Acc / PR-AUC / etc. via the shared metric helper."""
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
        labels.append(yb.numpy())
    return full_binary_metrics(np.concatenate(labels), np.concatenate(probs), threshold=0.5)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader, val_loader, test_loader = load_splits(
        train_csv=TRAIN_PATH,
        val_csv=VAL_PATH,
        test_csv=TEST_PATH,
        seq_len=8,
        batch_size=64,
    )

    device = "cpu"
    model = SeqCNN().to(device)

    # Class weighting matched to the VQC's pos_weight = n_neg / n_pos.
    y_train = train_loader.dataset.y
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32
    )
    print(f"Class weight for DNMT3B (positive class): {pos_weight.item():.4f}x")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improve = 0

    epochs = EPOCHS
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total += loss.item() * len(yb)
            n += len(yb)

        train_loss = total / max(1, n)
        val_loss, val_acc, val_conf, val_auc, val_f1 = eval_accuracy_and_confmat(
            model, val_loader, loss_fn, device=device, model_type="cnn"
        )

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | "
            f"val_auc={val_auc:.3f} | val_f1={val_f1:.3f} | CM={val_conf}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val improvement in {PATIENCE}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model with val_loss={best_val_loss:.4f}")

    test_loss, test_acc, test_conf, test_auc, test_f1 = eval_accuracy_and_confmat(
        model, test_loader, loss_fn, device=device, model_type="cnn"
    )
    m = full_metrics_for_loader(model, test_loader, device=device)
    print(
        f"TEST | loss={test_loss:.4f} | acc={m['accuracy']:.3f} | "
        f"bal_acc={m['balanced_acc']:.3f} | auc={m['roc_auc']:.3f} | "
        f"pr_auc={m['pr_auc']:.3f} | recall_3B={m['recall_dnmt3b']:.3f} | "
        f"f1={m['f1_dnmt3b']:.3f} | CM={test_conf}"
    )

    torch.save(model.state_dict(), "enzyme_cnn.pt")
    print("Saved trained model to enzyme_cnn.pt")


if __name__ == "__main__":
    main()
