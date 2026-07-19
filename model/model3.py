"""
Deterministic rule-based scorer (not trained) giving the meta-learner a
signal grounded in clinical criteria, independent of Models 1/2.

PCOS: Rotterdam criteria, 2 of 3 required:
  1. Oligo/anovulation: irregular menstrual cycles.
  2. Hyperandrogenism: hirsutism, or LH/FSH ratio > 2 (biochemical proxy
     used when no direct androgen assay is available).
  3. Polycystic ovarian morphology: antral follicle count >= 12 per ovary
     (2003 Rotterdam method, matches this dataset's follicle counts; the
     2023 guideline changed the counting method itself, not just the
     number, so it can't be retrofit here), OR elevated AMH (2023
     guideline's accepted ultrasound alternative; 5 ng/mL cutoff is an
     approximate literature value, not a single standardized assay cutoff).

Thyroid dysfunction: TSH outside reference range (same threshold used to
build the proxy label, so the PCOS-cohort check is tautological -- the
mixed-sex cohort's real diagnosis is the meaningful test).

Outputs a 0-3 PCOS criteria count (binary call at >=2, plus continuous
score for AUC) and a binary thyroid call.
"""

import os
import pandas as pd
from pipeline_common import save_json_with_metadata
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "splits")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

TSH_LOW_HYPERTHYROID = 0.4
TSH_HIGH_HYPOTHYROID = 4.5
LH_FSH_RATIO_THRESHOLD = 2.0
FOLLICLE_COUNT_THRESHOLD = 12  # 2003 Rotterdam per-ovary count
AMH_ELEVATED_THRESHOLD = 5.0   # ng/mL, approximate literature value


def thyroid_criteria_score(tsh: pd.Series) -> pd.Series:
    """Binary clinical call: TSH outside the normal reference range."""
    return ((tsh < TSH_LOW_HYPERTHYROID) | (tsh > TSH_HIGH_HYPOTHYROID)).astype(int)


def pcos_criteria_score(df: pd.DataFrame) -> pd.DataFrame:
    """3 Rotterdam criteria flags, their sum (0-3), and the binary call (>=2 of 3)."""
    oligo_anovulation = (df["cycle_regularity"] == 4).astype(int)

    lh_fsh_ratio = df["lh"] / df["fsh"].replace(0, pd.NA)
    hyperandrogenism = ((df["hirsutism"] == 1) | (lh_fsh_ratio > LH_FSH_RATIO_THRESHOLD)).astype(int)

    polycystic_morphology = (
        (df["follicle_count_left"] >= FOLLICLE_COUNT_THRESHOLD) |
        (df["follicle_count_right"] >= FOLLICLE_COUNT_THRESHOLD) |
        (df["amh"] >= AMH_ELEVATED_THRESHOLD)  # AMH as ultrasound alternative
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

    # PCOS cohort: evaluate both axes
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

    # Mixed-sex thyroid cohort: the meaningful thyroid test
    print("\n" + "=" * 70)
    print("Mixed-sex thyroid cohort (test split) -- real clinician diagnosis")
    print("=" * 70)
    thyroid_test = pd.read_csv(os.path.join(DATA_DIR, "thyroid_test.csv"))
    thy_call_real = thyroid_criteria_score(thyroid_test["tsh"])
    print("\n--- Thyroid axis: simple TSH threshold vs REAL clinician diagnosis ---")
    evaluate_binary(thyroid_test["thyroid_dysfunction"], thy_call_real,
                     "Thyroid (mixed-sex cohort, real diagnosis)")

    # Save thresholds so predict.py can apply the same logic at inference time.
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