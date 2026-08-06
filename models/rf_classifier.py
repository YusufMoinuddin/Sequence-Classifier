# models/rf_classifier.py
"""
Random Forest baseline for 8-mer DNA classification (DNMT3A vs DNMT3B).

Model factory only, matching the interface of models/svm_classifier.py.
"""
from sklearn.ensemble import RandomForestClassifier


def build_rf(random_state: int = 42) -> RandomForestClassifier:
    """
    Random Forest with balanced class weights.

    class_weight="balanced" matches the SVM baseline and is the sklearn
    equivalent of the VQC's pos_weight = n_neg / n_pos (~5.4x on this split).
    """
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
