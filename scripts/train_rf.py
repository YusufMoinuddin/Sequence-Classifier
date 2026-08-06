import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# scripts/train_rf.py
#
# Single-run debugging entry point for the Random Forest baseline.
# Paper numbers come from benchmark_classical.py (5 seeds, mean +/- std).
from src.enzyme_common import load_seq_label, flatten_onehot, full_binary_metrics
from models.rf_classifier import build_rf

TRAIN_PATH = "data/deep_enzymology_qmproxy_train.csv"
VAL_PATH = "data/deep_enzymology_qmproxy_val.csv"
TEST_PATH = "data/deep_enzymology_qmproxy_test.csv"

SEED = 0


def dataset_to_xy(csv_path: str, seq_len: int = 8):
    """Load a split using only 'sequence' + 'label', flattened for sklearn."""
    x, y, n_neg, n_pos = load_seq_label(csv_path, expected_len=seq_len)
    print(f"  {csv_path}: rows={len(y)} DNMT3A={n_neg} DNMT3B={n_pos}")
    return flatten_onehot(x), y


def report(split_name: str, y_true, y_prob):
    m = full_binary_metrics(y_true, y_prob, threshold=0.5)
    print(
        f"{split_name} | acc={m['accuracy']:.3f} | bal_acc={m['balanced_acc']:.3f} | "
        f"auc={m['roc_auc']:.3f} | pr_auc={m['pr_auc']:.3f} | "
        f"recall_3B={m['recall_dnmt3b']:.3f} | f1={m['f1_dnmt3b']:.3f} | "
        f"CM={{'TP': {m['TP']}, 'FP': {m['FP']}, 'FN': {m['FN']}, 'TN': {m['TN']}}}"
    )


def main():
    seq_len = 8

    X_train, y_train = dataset_to_xy(TRAIN_PATH, seq_len=seq_len)
    X_val, y_val = dataset_to_xy(VAL_PATH, seq_len=seq_len)
    X_test, y_test = dataset_to_xy(TEST_PATH, seq_len=seq_len)

    clf = build_rf(random_state=SEED)
    clf.fit(X_train, y_train)

    report("VAL ", y_val, clf.predict_proba(X_val)[:, 1])
    report("TEST", y_test, clf.predict_proba(X_test)[:, 1])


if __name__ == "__main__":
    main()
