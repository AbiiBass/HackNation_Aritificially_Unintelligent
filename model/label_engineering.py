"""
label_engineering.py
Builds training labels from the two cleaned cohorts:
- PCOS cohort: pcos_diagnosis is a real label; thyroid_dysfunction is a
  proxy built from TSH (no clinician thyroid diagnosis available); joint_class
  combines both (0=neither, 1=thyroid only, 2=pcos only, 3=both).
- Thyroid cohort: thyroid_dysfunction is a real label (renamed from
  thyroid_diagnosis_binary); no PCOS equivalent exists for this cohort.

Also caps one PCOS-cohort LH value of 2018 mIU/mL (data-entry error) at a
fixed physiological ceiling rather than a statistical outlier rule, so real
high lab values (e.g. TSH in hypothyroid patients) aren't clipped.
"""

import os
import pandas as pd
import numpy as np
from pipeline_common import encode_joint_class, CLASS_NAMES, save_json_with_metadata

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# TSH reference range (mIU/L); values outside it are used as a proxy for
# thyroid dysfunction on the PCOS cohort (no clinician diagnosis available).
TSH_LOW_HYPERTHYROID = 0.4   # below this -> hyperthyroid-range
TSH_HIGH_HYPOTHYROID = 4.5   # above this -> hypothyroid-range


def cap_impossible_values(series: pd.Series, max_plausible: float) -> pd.Series:
    """Caps values above a fixed physiological ceiling (not a statistical
    rule), so real high lab values carrying disease signal aren't clipped.
    """
    capped = series.clip(upper=max_plausible)
    changed_mask = (capped != series) & series.notna()
    n_changed = changed_mask.sum()
    if n_changed:
        changed_vals = series[changed_mask].tolist()
        print(f"  [impossible-value cap] {series.name}: capped {n_changed} value(s) "
              f"{changed_vals} -> {max_plausible} (above physiological ceiling)")
    return capped


# Ceilings set above real extreme values, to catch only data-entry errors
# (like LH=2018).
PLAUSIBLE_MAX = {
    "lh": 200,     # menopause/PCOS extremes reach ~100-150 mIU/mL
    "fsh": 200,    # similar range to LH
    "amh": 50,     # severe PCOS can exceed 20 ng/mL
    # tsh not capped -- severe hypothyroidism can genuinely exceed 100 mIU/L
}


def thyroid_dysfunction_from_tsh(tsh: pd.Series) -> pd.Series:
    """Binary proxy: 1 if TSH outside the normal reference range."""
    return ((tsh < TSH_LOW_HYPERTHYROID) | (tsh > TSH_HIGH_HYPOTHYROID)).astype(int)


def build_pcos_labels() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "pcos_cohort.csv"))

    print("[pcos] fixing physiologically impossible values...")
    for col, ceiling in PLAUSIBLE_MAX.items():
        if col in df.columns:
            df[col] = cap_impossible_values(df[col], ceiling)

    print("[pcos] building thyroid_dysfunction proxy from TSH...")
    df["thyroid_dysfunction"] = thyroid_dysfunction_from_tsh(df["tsh"])

    # joint_class encoding is defined once in pipeline_common.py
    df["joint_class"] = encode_joint_class(df["pcos_diagnosis"], df["thyroid_dysfunction"])
    df["joint_class_name"] = df["joint_class"].map(CLASS_NAMES)

    return df


def build_thyroid_cohort_labels() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "thyroid_mixed_cohort.csv"))
    df["thyroid_dysfunction"] = df["thyroid_diagnosis_binary"]  # real label, renamed

    # Impute missing lab values with the median; print what changed.
    for col in ["tsh", "t3", "t4", "t4_uptake", "free_thyroxine_index", "age"]:
        if col in df.columns and df[col].isna().any():
            n_missing = df[col].isna().sum()
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  [thyroid] imputed {n_missing} missing '{col}' values with median ({median_val:.2f})")

    return df


def main():
    print("=== PCOS cohort ===")
    pcos_labeled = build_pcos_labels()
    pcos_out = os.path.join(DATA_DIR, "pcos_cohort_labeled.csv")
    pcos_labeled.to_csv(pcos_out, index=False)
    print(f"saved -> {pcos_out}\n")
    print("Joint class distribution:")
    print(pcos_labeled["joint_class_name"].value_counts())
    print(f"\nCross-tab (rows=PCOS, cols=thyroid_dysfunction):")
    print(pd.crosstab(pcos_labeled["pcos_diagnosis"], pcos_labeled["thyroid_dysfunction"]))

    print("\n=== Thyroid (mixed-sex) cohort ===")
    thyroid_labeled = build_thyroid_cohort_labels()
    thyroid_out = os.path.join(DATA_DIR, "thyroid_cohort_labeled.csv")
    thyroid_labeled.to_csv(thyroid_out, index=False)
    print(f"saved -> {thyroid_out}\n")
    print("Thyroid dysfunction distribution by sex:")
    print(thyroid_labeled.groupby("sex")["thyroid_dysfunction"].mean())


if __name__ == "__main__":
    main()