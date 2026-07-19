"""
model3.py
Step 5 of the Women's Hormonal Health ensemble model pipeline.

Model 3 is NOT trained -- it's a deterministic rule-based scorer that
encodes actual clinical diagnostic criteria, exactly as sketched on the
whiteboard. It gives the meta-learner (Step 6) a signal grounded in
established medicine, independent of whatever patterns Models 1/2
picked up (or failed to) from limited data.

PCOS: Rotterdam criteria (2 of the following 3 required):
  1. Oligo/anovulation          -> irregular menstrual cycles
  2. Hyperandrogenism           -> hirsutism (clinical) OR LH/FSH > 2 (biochemical
                                    proxy -- we don't have a direct androgen assay,
                                    but an LH:FSH ratio above 2 is a well-documented
                                    PCOS marker used when testosterone isn't available)
  3. Polycystic ovarian morphology -> antral follicle count >= 12 in either ovary,
                                    OR elevated AMH (added post-review, see below)

POST-REVIEW FIX (critique #7): the original 2003 Rotterdam follicle-count
threshold (>=12 per ovary) is what this dataset's "Follicle No." columns
appear to measure. The 2023 International PCOS Guideline revised this,
but NOT to a simple new number -- it changed the counting METHOD itself,
to follicle number per ultrasound SECTION (FNPS >= 10) or ovarian volume
(>=10mL), which isn't the same measurement as a per-ovary total count.
We can't retroactively convert this dataset's counts to that method, so
we keep the original >=12 per-ovary threshold (documented as such, not
silently treated as current best practice) AND add the 2023 guideline's
other explicitly endorsed alternative: elevated AMH can substitute for
the ultrasound criterion entirely. We DO have AMH in this dataset, so
that part of the update is directly implementable. The AMH cutoff below
(5 ng/mL) is an approximation from published PCOS literature, not a
single universally standardized assay-specific value -- flagged as such.

Thyroid dysfunction: standard TSH reference range (same threshold used
to build the Step 2 proxy label -- see the note in main() about why that
makes the PCOS-cohort evaluation tautological by construction, and why
the mixed-sex cohort's REAL diagnosis is the meaningful test instead).

Output for each patient: a 0-3 PCOS criteria count (used both as a binary
call at >=2, and as a continuous risk score for AUC) and a binary thyroid
call. The meta-learner will consume the continuous scores, not just the
binary calls, so it has more to work with than a single yes/no per axis.
"""

import os
import json
import pandas as pd
from pipeline_common import save_json_with_metadata
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "splits")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

TSH_LOW_HYPERTHYROID = 0.4
TSH_HIGH_HYPOTHYROID = 4.5
LH_FSH_RATIO_THRESHOLD = 2.0
FOLLICLE_COUNT_THRESHOLD = 12  # 2003 Rotterdam per-ovary count; see docstring re: 2023 update
AMH_ELEVATED_THRESHOLD = 5.0   # ng/mL; approximate, literature-derived -- see docstring


def thyroid_criteria_score(tsh: pd.Series) -> pd.Series:
    """Binary clinical call: TSH outside the normal reference range."""
    return ((tsh < TSH_LOW_HYPERTHYROID) | (tsh > TSH_HIGH_HYPOTHYROID)).astype(int)


def pcos_criteria_score(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame with the 3 individual Rotterdam criteria flags,
    their sum (0-3, used as a continuous risk score), and the binary
    Rotterdam call (>= 2 of 3)."""
    oligo_anovulation = (df["cycle_regularity"] == 4).astype(int)

    lh_fsh_ratio = df["lh"] / df["fsh"].replace(0, pd.NA)
    hyperandrogenism = ((df["hirsutism"] == 1) | (lh_fsh_ratio > LH_FSH_RATIO_THRESHOLD)).astype(int)

    polycystic_morphology = (
        (df["follicle_count_left"] >= FOLLICLE_COUNT_THRESHOLD) |
        (df["follicle_count_right"] >= FOLLICLE_COUNT_THRESHOLD) |
        (df["amh"] >= AMH_ELEVATED_THRESHOLD)  # 2023 guideline: AMH as ultrasound alternative
    ).astype(int)

    criteria_count = oligo_anovulation + hyperandrogenism + polycystic_morphology
    rotterdam_call = (criteria_count >= 2).astype(int)

    return pd.DataFrame({
        "criterion_oligo_anovulation": oligo_anovulation,
        "criterion_hyperandrogenism": hyperandrogenism,
        "criterion_polycystic_morphology": polycystic_morphology,
        "pcos_criteria_count": criteria_count,
        "pcos_rotterdam_call": rotterdam_call,
    })


def evaluate_binary(y_true, y_pred, label: str):
    print(f"\n[{label}] confusion matrix (rows=actual, cols=predicted):\n{confusion_matrix(y_true, y_pred)}")
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- PCOS cohort: evaluate BOTH axes ---
    print("=" * 70)
    print("PCOS cohort (test split)")
    print("=" * 70)
    pcos_test = pd.read_csv(os.path.join(DATA_DIR, "pcos_test.csv"))

    pcos_scores = pcos_criteria_score(pcos_test)
    pcos_test = pd.concat([pcos_test, pcos_scores], axis=1)

    print("\n--- PCOS axis: Rotterdam criteria vs real pcos_diagnosis ---")
    auc = roc_auc_score(pcos_test["pcos_diagnosis"], pcos_test["pcos_criteria_count"])
    print(f"AUC (using 0-3 criteria count as risk score): {auc:.3f}")
    evaluate_binary(pcos_test["pcos_diagnosis"], pcos_test["pcos_rotterdam_call"], "PCOS (binary >=2 call)")

    print("\n--- Thyroid axis: TSH threshold vs thyroid_dysfunction proxy ---")
    print("NOTE: this comparison is tautological by construction -- Step 2's")
    print("proxy label used this exact same TSH threshold. A ~100% match here")
    print("confirms Model 3 correctly encodes the rule; it is NOT evidence of")
    print("real-world accuracy. See the mixed-sex cohort below for that.")
    thy_call = thyroid_criteria_score(pcos_test["tsh"])
    evaluate_binary(pcos_test["thyroid_dysfunction"], thy_call, "Thyroid (PCOS cohort, tautological check)")

    # --- Mixed-sex thyroid cohort: the MEANINGFUL thyroid test ---
    print("\n" + "=" * 70)
    print("Mixed-sex thyroid cohort (test split) -- real clinician diagnosis")
    print("=" * 70)
    thyroid_test = pd.read_csv(os.path.join(DATA_DIR, "thyroid_test.csv"))
    thy_call_real = thyroid_criteria_score(thyroid_test["tsh"])
    print("\n--- Thyroid axis: simple TSH threshold vs REAL clinician diagnosis ---")
    evaluate_binary(thyroid_test["thyroid_dysfunction"], thy_call_real,
                     "Thyroid (mixed-sex cohort, real diagnosis)")

    # Save the rule config (not a trained model -- just the thresholds, so
    # predict.py can apply the exact same logic at inference time)
    config = {
        "tsh_low_hyperthyroid": TSH_LOW_HYPERTHYROID,
        "tsh_high_hypothyroid": TSH_HIGH_HYPOTHYROID,
        "lh_fsh_ratio_threshold": LH_FSH_RATIO_THRESHOLD,
        "follicle_count_threshold": FOLLICLE_COUNT_THRESHOLD,
        "amh_elevated_threshold": AMH_ELEVATED_THRESHOLD,
        "pcos_rotterdam_min_criteria": 2,
        "notes": "follicle_count_threshold uses 2003 Rotterdam per-ovary counting; "
                 "amh_elevated_threshold added per 2023 guideline update as an "
                 "ultrasound alternative (approximate, not a single standardized cutoff).",
    }
    config_path = os.path.join(MODEL_DIR, "model3_config.json")
    save_json_with_metadata(config, config_path)
    print(f"\nSaved rule config -> {config_path}")


if __name__ == "__main__":
    main()