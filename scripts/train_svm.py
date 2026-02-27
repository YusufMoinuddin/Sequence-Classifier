import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# scripts/train_svm.py
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

from src.enzyme_common import EnzDataset


def dataset_to_xy(csv_path: str, seq_len: int = 8):
    df = pd.read_csv(csv_path)
    ds = EnzDataset(df, expected_len=seq_len)

    # ds.x is (N,4,L). SVM wants (N, D)
    X = ds.x.reshape(len(ds.x), -1)  # flatten (4*L)
    y = ds.y.astype(int)
    return X, y


def main():
    seq_len = 8

    X_train, y_train = dataset_to_xy("deep_enzymology_train.csv", seq_len=seq_len)
    X_val, y_val = dataset_to_xy("deep_enzymology_val.csv", seq_len=seq_len)
    X_test, y_test = dataset_to_xy("deep_enzymology_test.csv", seq_len=seq_len)

    # SVM baseline
    clf = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(X_train, y_train)

    # --- VAL ---
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_pred = (val_probs >= 0.5).astype(int)

    val_acc = accuracy_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) == 2 else float("nan")
    val_f1 = f1_score(y_val, val_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_val, val_pred, labels=[0, 1]).ravel()

    print(
        f"VAL | acc={val_acc:.3f} | auc={val_auc:.3f} | f1={val_f1:.3f} | "
        f"CM={{'TP': {tp}, 'FP': {fp}, 'FN': {fn}, 'TN': {tn}}}"
    )

    # --- TEST ---
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= 0.5).astype(int)

    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) == 2 else float("nan")
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    print(
        f"TEST | acc={test_acc:.3f} | auc={test_auc:.3f} | f1={test_f1:.3f} | "
        f"CM={{'TP': {tp}, 'FP': {fp}, 'FN': {fn}, 'TN': {tn}}}"
    )


if __name__ == "__main__":
    main()
