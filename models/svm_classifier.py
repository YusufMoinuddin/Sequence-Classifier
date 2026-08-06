# models/svm_classifier.py
"""
SVM baseline for 8-mer DNA classification (DNMT3A vs DNMT3B).

Model factory only, mirroring models/cnn_classifier.py which holds just the
nn.Module. Training/eval lives in benchmark_classical.py and scripts/train_svm.py.
"""
from sklearn.svm import SVC


def build_svm(random_state: int = 42) -> SVC:
    """
    RBF-kernel SVM with balanced class weights.

    class_weight="balanced" reweights each class by n_samples / (n_classes *
    n_class_samples), giving DNMT3B ~5.4x the weight of DNMT3A on this split —
    the sklearn equivalent of the VQC's pos_weight = n_neg / n_pos.

    probability=True enables predict_proba (Platt scaling via internal CV),
    which is required for ROC-AUC and PR-AUC and is why random_state matters.
    """
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )
