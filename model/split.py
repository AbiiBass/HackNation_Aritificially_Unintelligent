"""
split.py
Step 3 of the Women's Hormonal Health ensemble model pipeline.

Splits both labeled cohorts into train/val/test (70/15/15), stratified by
their target so rare classes stay represented in every split.

IMPORTANT LEAKAGE NOTE (carries forward into Step 4):
On the PCOS cohort, `thyroid_dysfunction` was constructed directly from
`tsh` (see label_engineering.py). That means `tsh` can NOT be used as an
input feature when predicting `thyroid_dysfunction` or `joint_class` on
this cohort -- it would just be decoding the label, not predicting it.
`tsh` is fine to use as a feature when predicting `pcos_diagnosis` alone
(that label doesn't depend on it). This file only handles splitting; the
feature-exclusion happens in train.py, but flagging it here since it's a
direct consequence of how the split target is chosen.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
RANDOM_STATE = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


def stratified_three_way_split(df: pd.DataFrame, stratify_col: str):
    """70/15/15 split, stratified by stratify_col. Two-step: first peel off
    train, then split the remainder into val/test."""
    train_df, rest_df = train_test_split(
        df,
        train_size=TRAIN_FRAC,
        stratify=df[stratify_col],
        random_state=RANDOM_STATE,
    )
    # rest_df is 30% of data; split it 50/50 to get 15%/15% of the original
    val_df, test_df = train_test_split(
        rest_df,
        train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=rest_df[stratify_col],
        random_state=RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def report_split(name: str, train_df, val_df, test_df, stratify_col: str):
    print(f"\n[{name}] split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df[stratify_col].value_counts(normalize=True).sort_index()
        print(f"  {split_name} {stratify_col} distribution:\n{dist.to_string()}")


def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # --- PCOS cohort: stratify by the 4-class joint target ---
    pcos = pd.read_csv(os.path.join(DATA_DIR, "pcos_cohort_labeled.csv"))
    pcos_train, pcos_val, pcos_test = stratified_three_way_split(pcos, "joint_class")
    pcos_train.to_csv(os.path.join(SPLIT_DIR, "pcos_train.csv"), index=False)
    pcos_val.to_csv(os.path.join(SPLIT_DIR, "pcos_val.csv"), index=False)
    pcos_test.to_csv(os.path.join(SPLIT_DIR, "pcos_test.csv"), index=False)
    report_split("pcos", pcos_train, pcos_val, pcos_test, "joint_class")

    # --- Thyroid cohort: stratify by the binary diagnosis ---
    thyroid = pd.read_csv(os.path.join(DATA_DIR, "thyroid_cohort_labeled.csv"))
    thy_train, thy_val, thy_test = stratified_three_way_split(thyroid, "thyroid_dysfunction")
    thy_train.to_csv(os.path.join(SPLIT_DIR, "thyroid_train.csv"), index=False)
    thy_val.to_csv(os.path.join(SPLIT_DIR, "thyroid_val.csv"), index=False)
    thy_test.to_csv(os.path.join(SPLIT_DIR, "thyroid_test.csv"), index=False)
    report_split("thyroid", thy_train, thy_val, thy_test, "thyroid_dysfunction")

    print(f"\nAll splits saved to {SPLIT_DIR}/")


if __name__ == "__main__":
    main()