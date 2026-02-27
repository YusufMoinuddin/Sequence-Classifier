# scripts/make_splits_qmproxy.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    in_csv = "data/deep_enzymology_motifs_with_qmproxy.csv"
    out_train = "data/deep_enzymology_qmproxy_train.csv"
    out_val = "data/deep_enzymology_qmproxy_val.csv"
    out_test = "data/deep_enzymology_qmproxy_test.csv"

    df = pd.read_csv(in_csv)

    # create numeric label if needed (you have both `label` and `label_str`)
    # We'll stratify on df["label"] if it's numeric, else label_str.
    strat_col = "label" if "label" in df.columns else "label_str"

    # 80/10/10 split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[strat_col] if strat_col in df.columns else None,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df[strat_col] if strat_col in temp_df.columns else None,
    )

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)

    print("Wrote:")
    print(" ", out_train, len(train_df))
    print(" ", out_val, len(val_df))
    print(" ", out_test, len(test_df))


if __name__ == "__main__":
    main()