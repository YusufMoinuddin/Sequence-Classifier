import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# scripts/train_cnn.py
import torch
import torch.nn as nn

from src.enzyme_common import load_splits, eval_accuracy_and_confmat
from models.cnn_classifier import SeqCNN


def main():
    train_loader, val_loader, test_loader = load_splits(
        train_csv="deep_enzymology_train.csv",
        val_csv="deep_enzymology_val.csv",
        test_csv="deep_enzymology_test.csv",
        seq_len=8,
        batch_size=64,
    )

    device = "cpu"
    model = SeqCNN().to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = None

    epochs = 1000
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
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model with val_loss={best_val_loss:.4f}")

    test_loss, test_acc, test_conf, test_auc, test_f1 = eval_accuracy_and_confmat(
        model, test_loader, loss_fn, device=device, model_type="cnn"
    )
    print(
        f"TEST | loss={test_loss:.4f} | acc={test_acc:.3f} | "
        f"auc={test_auc:.3f} | f1={test_f1:.3f} | CM={test_conf}"
    )

    torch.save(model.state_dict(), "enzyme_cnn.pt")
    print("Saved trained model to enzyme_cnn.pt")


if __name__ == "__main__":
    main()
