# scripts/train_transformer.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import importlib
import time
import pandas as pd
import torch
import torch.nn as nn

import src.qm_features as qmf
importlib.reload(qmf)

import src.enzyme_common as ec
importlib.reload(ec)

from models.transformer_classifier import DNATransformerClassifier


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _inference_latency_ms_per_sample(model, loader, device: str, use_qm: bool, warmup_batches: int = 2):
    model.eval()
    times = []
    n_samples = 0
    for batch_idx, batch in enumerate(loader):
        if use_qm:
            xb, qm, _ = batch
            qm = qm.to(device)
        else:
            xb, _ = batch
            qm = None
        xb = xb.to(device)
        token_ids = xb.argmax(dim=1).long()
        if batch_idx < warmup_batches:
            _ = model(token_ids, qm_feats=qm) if use_qm else model(token_ids)
            continue
        t0 = time.perf_counter()
        _ = model(token_ids, qm_feats=qm) if use_qm else model(token_ids)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        n_samples += len(token_ids)
    if not times or n_samples == 0:
        return float("nan")
    return (sum(times) / n_samples) * 1000.0


def _train_one_model(use_qm: bool, seq_len: int = 8):
    train_csv = "data/deep_enzymology_qmproxy_train.csv"
    val_csv = "data/deep_enzymology_qmproxy_val.csv"
    test_csv = "data/deep_enzymology_qmproxy_test.csv"

    if use_qm:
        train_df = pd.read_csv(train_csv)
        var_report = qmf.qm_variance_report(train_df)
        print("\n[QM variance report on TRAIN split]")
        for k, v in var_report.items():
            print(f"  {k}: {v:.10f}")
        near_constant = [k for k, v in var_report.items() if v < 1e-8]
        if near_constant:
            print(f"WARNING: near-constant QM features detected: {near_constant}")

    train_loader, val_loader, test_loader = ec.load_splits(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        seq_len=seq_len,
        batch_size=64,
        use_qm=use_qm,
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
        use_qm=use_qm,
        qm_dim=4,
        qm_embed_dim=32,
    ).to(device)
    n_params = _count_params(model)
    print(f"Trainable parameters: {n_params}")

    # --- compute pos_weight from TRAIN labels to handle imbalance ---
    # pos_weight = (#neg / #pos)
    y_all = []
    for batch in train_loader:
        if use_qm:
            _, _, yb = batch
        else:
            _, yb = batch
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
    batch = next(iter(train_loader))
    if use_qm:
        xb, qm, yb = batch
    else:
        xb, yb = batch
        qm = None
    print("Sanity check:")
    print("  xb:", tuple(xb.shape), "expected (B,4,8)")
    if use_qm:
        print("  qm:", tuple(qm.shape), "expected (B,4)")
        print("  qm min/max:", float(qm.min()), float(qm.max()))
    print("  yb:", tuple(yb.shape), "expected (B,)")
    print("  y unique:", torch.unique(yb))

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    epochs = 150
    patience = 20
    no_improve_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0

        for batch in train_loader:
            if use_qm:
                xb, qm, yb = batch
                qm = qm.to(device)
            else:
                xb, yb = batch
                qm = None
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            token_ids = xb.argmax(dim=1).long()
            logits = model(token_ids, qm_feats=qm) if use_qm else model(token_ids)
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
            use_qm=use_qm,
        )

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | "
            f"val_auc={val_auc:.3f} | val_f1={val_f1:.3f} | CM={val_conf}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch} with val_loss={best_val_loss:.4f}")

    test_loss, test_acc, test_conf, test_auc, test_f1 = ec.eval_accuracy_and_confmat(
        model,
        test_loader,
        loss_fn,
        device=device,
        model_type="transformer",
        use_qm=use_qm,
    )
    latency_ms = _inference_latency_ms_per_sample(model, test_loader, device=device, use_qm=use_qm)

    print(
        f"TEST | loss={test_loss:.4f} | acc={test_acc:.3f} | "
        f"auc={test_auc:.3f} | f1={test_f1:.3f} | CM={test_conf} | "
        f"params={n_params} | latency_ms_per_sample={latency_ms:.6f}"
    )
    return model, {
        "loss": test_loss,
        "acc": test_acc,
        "auc": test_auc,
        "f1": test_f1,
        "confusion_matrix": test_conf,
        "params": n_params,
        "latency_ms_per_sample": latency_ms,
    }


def main():
    print("\n=== Baseline: Sequence-only ===")
    seq_model, seq_metrics = _train_one_model(use_qm=False, seq_len=8)
    torch.save(seq_model.state_dict(), "enzyme_transformer_seq_only.pt")
    print("Saved trained model to enzyme_transformer_seq_only.pt")

    print("\n=== Sequence + QM features ===")
    qm_model, qm_metrics = _train_one_model(use_qm=True, seq_len=8)
    torch.save(qm_model.state_dict(), "enzyme_transformer_qm.pt")
    print("Saved trained model to enzyme_transformer_qm.pt")

    print("\n=== Comparison (test split) ===")
    print(
        f"Sequence-only: acc={seq_metrics['acc']:.3f}, auc={seq_metrics['auc']:.3f}, "
        f"f1={seq_metrics['f1']:.3f}, params={seq_metrics['params']}, "
        f"latency_ms/sample={seq_metrics['latency_ms_per_sample']:.6f}, "
        f"CM={seq_metrics['confusion_matrix']}"
    )
    print(
        f"Sequence+QM : acc={qm_metrics['acc']:.3f}, auc={qm_metrics['auc']:.3f}, "
        f"f1={qm_metrics['f1']:.3f}, params={qm_metrics['params']}, "
        f"latency_ms/sample={qm_metrics['latency_ms_per_sample']:.6f}, "
        f"CM={qm_metrics['confusion_matrix']}"
    )


if __name__ == "__main__":
    main()