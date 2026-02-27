# scripts/train_gnn.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch_geometric.loader import DataLoader as GeoDataLoader

from src.graph_common import load_graphs
from models.gnn_classifier import SeqGCN

@torch.no_grad()
def eval_gnn(model, loader, loss_fn, device="cpu"):
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_y = [], []

    tp=tn=fp=fn=0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)                 # (B,)
        y = batch.y.view(-1).to(device)       # (B,)

        loss = loss_fn(logits, y)
        total_loss += loss.item() * y.numel()
        n += y.numel()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        preds = (probs >= 0.5).astype(int)
        y_int = y_np.astype(int)

        tn_, fp_, fn_, tp_ = confusion_matrix(y_int, preds, labels=[0,1]).ravel()
        tn += tn_; fp += fp_; fn += fn_; tp += tp_

        all_probs.append(probs)
        all_y.append(y_np)

    avg_loss = total_loss / max(1, n)
    probs = np.concatenate(all_probs) if all_probs else np.array([])
    y = np.concatenate(all_y) if all_y else np.array([])

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) == 2 else float("nan")
    f1 = f1_score(y.astype(int), (probs >= 0.5).astype(int), zero_division=0) if len(y) else 0.0
    conf = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    return avg_loss, acc, conf, auc, f1

def main():
    device = "cpu"

    train_graphs = load_graphs("deep_enzymology_train.csv")
    val_graphs   = load_graphs("deep_enzymology_val.csv")
    test_graphs  = load_graphs("deep_enzymology_test.csv")

    train_loader = GeoDataLoader(train_graphs, batch_size=64, shuffle=True)
    val_loader   = GeoDataLoader(val_graphs, batch_size=64, shuffle=False)
    test_loader  = GeoDataLoader(test_graphs, batch_size=64, shuffle=False)

    model = SeqGCN(in_dim=4, hidden=64).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, 11):
        model.train()
        total, n = 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()

            logits = model(batch)
            y = batch.y.view(-1).to(device)

            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total += loss.item() * y.numel()
            n += y.numel()

        train_loss = total / max(1, n)
        val_loss, val_acc, val_conf, val_auc, val_f1 = eval_gnn(model, val_loader, loss_fn, device=device)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | "
            f"val_auc={val_auc:.3f} | val_f1={val_f1:.3f} | CM={val_conf}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model with val_loss={best_val_loss:.4f}")

    test_loss, test_acc, test_conf, test_auc, test_f1 = eval_gnn(model, test_loader, loss_fn, device=device)
    print(
        f"TEST | loss={test_loss:.4f} | acc={test_acc:.3f} | "
        f"auc={test_auc:.3f} | f1={test_f1:.3f} | CM={test_conf}"
    )

    torch.save(model.state_dict(), "enzyme_gnn.pt")
    print("Saved trained model to enzyme_gnn.pt")

if __name__ == "__main__":
    main()