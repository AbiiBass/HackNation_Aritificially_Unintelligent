"""
label_engineering.py
Step 2 of the Women's Hormonal Health ensemble model pipeline.

Takes the two cleaned cohorts from data_prep.py and builds the actual
training labels:

1. PCOS cohort (data/pcos_cohort.csv):
   - pcos_diagnosis: already a REAL clinician label, used as-is.
   - thyroid_dysfunction: a PROXY label built from TSH using standard
     clinical reference ranges (this cohort has TSH but no clinician
     thyroid diagnosis, so we can't avoid a proxy here -- documented
     openly, per the hackathon's "don't hide assumptions" guidance).
   - joint_class: the 4-way target combining the two:
       0 = no PCOS, no thyroid dysfunction
       1 = no PCOS, thyroid dysfunction
       2 = PCOS, no thyroid dysfunction
       3 = PCOS, thyroid dysfunction
     This is a REAL joint label (both halves come from the same 541
     patients), not a synthetic pairing across separate datasets.

2. Thyroid cohort (data/thyroid_mixed_cohort.csv):
   - thyroid_dysfunction: already a REAL clinician label
     (thyroid_diagnosis_binary from Step 1), just renamed for
     consistency. No PCOS equivalent exists for this cohort (see the
     Model 1 "abstains on PCOS" design note from our discussion).

Also fixes a known data-quality issue: one PCOS-cohort patient has an
LH value of 2018 mIU/mL (physiologically impossible -- almost certainly
a decimal-point data-entry error, since her FSH is normal). We cap this
using a fixed physiological ceiling, NOT a statistical outlier rule --
clinical lab values are legitimately right-skewed, and a statistical
rule (like IQR-based capping) would clip away real disease signal (a
genuinely high TSH in a hypothyroid patient is exactly what our proxy
label needs to see, not an error to smooth over).
"""

import os
import pandas as pd
import numpy as np
from pipeline_common import encode_joint_class, CLASS_NAMES, save_json_with_metadata

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Standard clinical TSH reference range (mIU/L). Values outside this range
# are used here as a PROXY for thyroid dysfunction on the PCOS cohort,
# since we only have TSH (no T3/T4/clinician diagnosis) for those patients.
TSH_LOW_HYPERTHYROID = 0.4   # below this -> hyperthyroid-range
TSH_HIGH_HYPOTHYROID = 4.5   # above this -> hypothyroid-range


def cap_impossible_values(series: pd.Series, max_plausible: float) -> pd.Series:
    """Cap values above a PHYSIOLOGICALLY implausible ceiling.

    We deliberately do NOT use a statistical rule (like IQR-based capping)
    here: clinical lab values are legitimately right-skewed, and a
    statistical outlier rule would clip away real disease signal (e.g. a
    genuinely high TSH in a hypothyroid patient is exactly the thing our
    label needs to see, not an error to be smoothed over). Instead we cap
    only values that are impossible for a living human, using published
    reference-range ceilings with generous headroom.
    """
    capped = series.clip(upper=max_plausible)
    changed_mask = (capped != series) & series.notna()
    n_changed = changed_mask.sum()
    if n_changed:
        changed_vals = series[changed_mask].tolist()
        print(f"  [impossible-value cap] {series.name}: capped {n_changed} value(s) "
              f"{changed_vals} -> {max_plausible} (above physiological ceiling)")
    return capped


# Physiologically implausible ceilings, set generously above the highest
# values seen even in severe disease / menopause, so we only catch clear
# data-entry errors (like LH=2018) and never touch real extreme cases.
PLAUSIBLE_MAX = {
    "lh": 200,     # menopausal/PCOS extremes can reach ~100-150 mIU/mL
    "fsh": 200,    # similar range to LH
    "amh": 50,     # severe PCOS cases can exceed 20 ng/mL; 50 is a safe ceiling
    # tsh intentionally NOT capped -- severe hypothyroidism can genuinely
    # produce TSH >100 mIU/L, and that's exactly the signal our proxy label
    # depends on.
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

    # joint_class: 0/1/2/3, encoded via the single shared definition in
    # pipeline_common.py (was previously duplicated by hand -- see that
    # module's docstring for why that was a bug waiting to happen)
    df["joint_class"] = encode_joint_class(df["pcos_diagnosis"], df["thyroid_dysfunction"])
    df["joint_class_name"] = df["joint_class"].map(CLASS_NAMES)

    return df


def build_thyroid_cohort_labels() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "thyroid_mixed_cohort.csv"))
    df["thyroid_dysfunction"] = df["thyroid_diagnosis_binary"]  # real label, just renamed

    # Simple, transparent imputation for a handful of missing lab values;
    # flagged explicitly rather than silently filled, per the "don't hide
    # preprocessing choices" guidance.
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