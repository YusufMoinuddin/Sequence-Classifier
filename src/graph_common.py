# src/graph_common.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

BASES = "ACGT"

def one_hot_base(b: str) -> torch.Tensor:
    v = torch.zeros(4, dtype=torch.float32)
    if b in BASES:
        v[BASES.index(b)] = 1.0
    else:
        v[:] = 0.25
    return v

def label_from_row(row) -> float:
    # Same label logic as enzyme_common.py
    if "label" in row and isinstance(row["label"], str):
        return 1.0 if row["label"] == "DNMT3B" else 0.0
    if "label" in row:
        return float(row["label"])
    if "meth_A" in row and "meth_B" in row:
        return 1.0 if float(row["meth_B"]) - float(row["meth_A"]) > 0 else 0.0
    raise ValueError("Need label or meth_A/meth_B")

def seq_to_graph(seq: str, y: float) -> Data:
    seq = (seq or "").upper().strip()
    L = len(seq)

    # Node features: (L, 4)
    x = torch.stack([one_hot_base(b) for b in seq], dim=0)

    # Edges: chain graph i<->i+1
    edges = []
    for i in range(L - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.float32))

def load_graphs(csv_path: str):
    df = pd.read_csv(csv_path)

    # Find sequence column
    if "sequence" in df.columns:
        seq_col = "sequence"
    elif "motif" in df.columns:
        seq_col = "motif"
    elif "seq" in df.columns:
        seq_col = "seq"
    else:
        raise ValueError("Need sequence/motif/seq column")

    graphs = []
    for _, row in df.iterrows():
        seq = str(row[seq_col])
        y = label_from_row(row)
        graphs.append(seq_to_graph(seq, y))
    return graphs
