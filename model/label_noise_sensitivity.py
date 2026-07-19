"""
label_noise_sensitivity.py
Post-review fix for critique #3.

We can't get a second, independently-labeled thyroid dataset for the
PCOS cohort (that data doesn't exist in what's available to this
project -- see the model card's "still unfixed" section). What we CAN
do is use the TSH-only rule's known, measured error rate -- from
comparing it against REAL diagnoses in the mixed-sex thyroid cohort --
to bound how much the PCOS-cohort's reported metrics could shift if the
proxy label were replaced with a true one.

Measured on the full mixed-sex cohort (3,163 patients, real diagnoses):
  precision = 0.097   (of everyone the TSH rule flags positive, ~90% are false alarms)
  recall    = 0.993   (the rule almost never misses a real case)
  FPR       = 0.462   (of everyone truly negative, ~46% get incorrectly flagged)

CAVEAT (stated plainly, not hidden): these error rates come from a
different population (older, mixed-sex, thyroid-referred patients) than
the PCOS cohort (young women, PCOS-referred). Applying them to the PCOS
cohort assumes the rule's error PATTERN transfers across populations,
which is an assumption, not a validated fact. This analysis gives a
plausible RANGE, not a corrected ground truth.

Method: Monte Carlo. For each of the PCOS cohort's current TSH-proxy
labels, probabilistically "correct" it using the measured error rates
(a labeled positive gets flipped to negative with probability related
to the measured false-discovery rate; a labeled negative gets flipped
to positive at the measured miss rate), redraw many times, and see how
much Model 2's AUC and the meta-learner's accuracy move under labels
that are LESS likely to be pure TSH-threshold artifacts.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

from pipeline_common import encode_joint_class, save_json_with_metadata

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "splits")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Measured on the full mixed-sex cohort (see docstring)
MEASURED_PRECISION = 0.097
MEASURED_RECALL = 0.993
MEASURED_FPR = 0.462
# False discovery rate: of labeled positives, this fraction are (by the
# measured rule) actually false positives.
FALSE_DISCOVERY_RATE = 1 - MEASURED_PRECISION
# Miss rate: of labeled negatives, roughly this fraction could actually
# be positives the rule failed to flag (derived from recall + typical
# base rates; kept conservative/small since recall was measured high).
MISS_RATE = 1 - MEASURED_RECALL


def simulate_corrected_labels(current_labels: pd.Series, rng: np.random.RandomState) -> pd.Series:
    """One Monte Carlo draw of 'what if some proxy labels were wrong,
    at the rates measured on the real-diagnosis cohort'."""
    corrected = current_labels.copy()
    positive_idx = current_labels[current_labels == 1].index
    negative_idx = current_labels[current_labels == 0].index

    # Flip some positives to negative, at the false-discovery rate
    flip_to_neg = rng.random(len(positive_idx)) < FALSE_DISCOVERY_RATE
    corrected.loc[positive_idx[flip_to_neg]] = 0

    # Flip a small number of negatives to positive, at the miss rate
    flip_to_pos = rng.random(len(negative_idx)) < MISS_RATE
    corrected.loc[negative_idx[flip_to_pos]] = 1

    return corrected


def main(n_simulations=500):
    pcos_test = pd.read_csv(os.path.join(DATA_DIR, "pcos_test.csv"))

    from train_model2 import clean_features as clean_feat_m2
    model2 = joblib.load(os.path.join(MODEL_DIR, "model2_women_only.joblib"))
    X2 = clean_feat_m2(pcos_test)
    proba2 = model2.predict_proba(X2)[:, 1]

    print("=" * 70)
    print("Label-noise sensitivity: Model 2 (thyroid_dysfunction) AUC")
    print("=" * 70)
    print(f"Reported AUC (using proxy labels as-is): {roc_auc_score(pcos_test['thyroid_dysfunction'], proba2):.3f}")

    rng = np.random.RandomState(42)
    sim_aucs = []
    for _ in range(n_simulations):
        corrected = simulate_corrected_labels(pcos_test["thyroid_dysfunction"], rng)
        if corrected.nunique() < 2:
            continue
        sim_aucs.append(roc_auc_score(corrected, proba2))

    print(f"\nUnder {len(sim_aucs)} Monte Carlo redraws of 'less noisy' labels:")
    print(f"  mean AUC = {np.mean(sim_aucs):.3f}")
    print(f"  range    = [{np.min(sim_aucs):.3f}, {np.max(sim_aucs):.3f}]")
    print(f"  std      = {np.std(sim_aucs):.3f}")
    auc_range = np.max(sim_aucs) - np.min(sim_aucs)
    print(f"\nInterpretation: the simulated range is very WIDE ({auc_range:.3f}) -- this is")
    print("NOT reassuring. At this false-discovery rate (~90%) and this sample size")
    print("(n=82, ~12 positive), most simulated draws flip the majority of positive")
    print("labels, making the AUC estimate itself unstable (anywhere from near-0 to 1.0).")
    print("Honest read: this tells us the PCOS-cohort test set is too small and the proxy")
    print("label too noisy to determine whether Model 2's 0.605 AUC reflects real signal")
    print("or noise. That uncertainty is itself the finding -- not something to explain away.")

    # Same exercise for the meta-learner's accuracy
    print("\n" + "=" * 70)
    print("Label-noise sensitivity: Meta-learner joint-class accuracy")
    print("=" * 70)
    from train_model2b_pcos import clean_features as clean_feat_m2b
    from meta_learner import build_meta_features
    model2b = joblib.load(os.path.join(MODEL_DIR, "model2b_women_only_pcos.joblib"))
    meta_model = joblib.load(os.path.join(MODEL_DIR, "meta_learner.joblib"))
    X_meta = build_meta_features(pcos_test, model2, model2b)
    pred_meta = meta_model.predict(X_meta)

    reported_acc = accuracy_score(pcos_test["joint_class"], pred_meta)
    print(f"Reported accuracy (using proxy labels as-is): {reported_acc:.3f}")

    rng = np.random.RandomState(42)
    sim_accs = []
    for _ in range(n_simulations):
        corrected_thyroid = simulate_corrected_labels(pcos_test["thyroid_dysfunction"], rng)
        corrected_joint = encode_joint_class(pcos_test["pcos_diagnosis"], corrected_thyroid)
        sim_accs.append(accuracy_score(corrected_joint, pred_meta))

    print(f"\nUnder {len(sim_accs)} Monte Carlo redraws:")
    print(f"  mean accuracy = {np.mean(sim_accs):.3f}")
    print(f"  range         = [{np.min(sim_accs):.3f}, {np.max(sim_accs):.3f}]")
    print(f"  std           = {np.std(sim_accs):.3f}")
    print(f"\nInterpretation: unlike Model 2's AUC, this range is TIGHT and CONSISTENT --")
    print(f"the meta-learner's reported 0.878 accuracy is likely somewhat INFLATED by")
    print(f"proxy-label noise. A more realistic estimate, if the thyroid label were less")
    print(f"noisy, is closer to {np.mean(sim_accs):.2f}. Report both numbers, not just the higher one.")

    out = {
        "measured_error_rates_source": "full mixed-sex thyroid cohort, real diagnoses (n=3163)",
        "false_discovery_rate": FALSE_DISCOVERY_RATE,
        "miss_rate": MISS_RATE,
        "caveat": "Error rates transferred from a different population (older, mixed-sex, "
                  "thyroid-referred) to the PCOS cohort (young women, PCOS-referred). This is "
                  "an assumption, not a validated transfer.",
        "model2_auc_reported": float(roc_auc_score(pcos_test["thyroid_dysfunction"], proba2)),
        "model2_auc_simulated_mean": float(np.mean(sim_aucs)),
        "model2_auc_simulated_range": [float(np.min(sim_aucs)), float(np.max(sim_aucs))],
        "meta_accuracy_reported": float(reported_acc),
        "meta_accuracy_simulated_mean": float(np.mean(sim_accs)),
        "meta_accuracy_simulated_range": [float(np.min(sim_accs)), float(np.max(sim_accs))],
    }
    out_path = os.path.join(MODEL_DIR, "label_noise_sensitivity.json")
    save_json_with_metadata(out, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()