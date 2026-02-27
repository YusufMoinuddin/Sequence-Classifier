# scripts/train_transformer.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import importlib
import torch
import torch.nn as nn

import src.qm_features as qmf
importlib.reload(qmf)

import src.enzyme_common as ec
importlib.reload(ec)

from models.transformer_classifier import DNATransformerClassifier


def main():
    seq_len = 8

    train_loader, val_loader, test_loader = ec.load_splits(
        train_csv="data/deep_enzymology_qmproxy_train.csv",
        val_csv="data/deep_enzymology_qmproxy_val.csv",
        test_csv="data/deep_enzymology_qmproxy_test.csv",
        seq_len=seq_len,
        batch_size=64,
        use_qm=True,
    )

    device = "cpu"

    model = DNATransformerClassifier(
        vocab_size=len(ec.BASES),
        d_model=128,
        n_heads=4,
        num_layers=2,
        dim_feedforward=256,
        max_seq_len=seq_len,
        dropout=0.1,
        use_qm=True,
        qm_dim=4,
        qm_embed_dim=32,
    ).to(device)

    # --- compute pos_weight from TRAIN labels to handle imbalance ---
    # pos_weight = (#neg / #pos)
    y_all = []
    for batch in train_loader:
        _, _, yb = batch
        y_all.append(yb)
    y_all = torch.cat(y_all, dim=0)
    n_pos = float((y_all == 1).sum().item())
    n_neg = float((y_all == 0).sum().item())
    if n_pos == 0:
        raise ValueError("Training set has 0 positive samples; cannot compute pos_weight.")
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"pos_weight (neg/pos) = {pos_weight.item():.4f} (neg={n_neg:.0f}, pos={n_pos:.0f})")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- sanity check batch ---
    xb, qm, yb = next(iter(train_loader))
    print("Sanity check:")
    print("  xb:", tuple(xb.shape), "expected (B,4,8)")
    print("  qm:", tuple(qm.shape), "expected (B,4)")
    print("  yb:", tuple(yb.shape), "expected (B,)")
    print("  qm min/max:", float(qm.min()), float(qm.max()))
    print("  y unique:", torch.unique(yb))

    best_val_loss = float("inf")
    best_state = None

    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0

        for xb, qm, yb in train_loader:
            xb = xb.to(device)
            qm = qm.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            token_ids = xb.argmax(dim=1).long()
            logits = model(token_ids, qm_feats=qm)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total += loss.item() * len(yb)
            n += len(yb)

        train_loss = total / max(1, n)

        val_loss, val_acc, val_conf, val_auc, val_f1 = ec.eval_accuracy_and_confmat(
            model,
            val_loader,
            loss_fn,
            device=device,
            model_type="transformer",
            use_qm=True,
        )

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | "
            f"val_auc={val_auc:.3f} | val_f1={val_f1:.3f} | CM={val_conf}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model with val_loss={best_val_loss:.4f}")

    test_loss, test_acc, test_conf, test_auc, test_f1 = ec.eval_accuracy_and_confmat(
        model,
        test_loader,
        loss_fn,
        device=device,
        model_type="transformer",
        use_qm=True,
    )

    print(
        f"TEST | loss={test_loss:.4f} | acc={test_acc:.3f} | "
        f"auc={test_auc:.3f} | f1={test_f1:.3f} | CM={test_conf}"
    )

    torch.save(model.state_dict(), "enzyme_transformer_qm.pt")
    print("Saved trained model to enzyme_transformer_qm.pt")


if __name__ == "__main__":
    main()