"""
predict.py
Step 7 of the Women's Hormonal Health ensemble model pipeline.

A single predict(patient) -> dict function that a FastAPI /predict
endpoint (or anything else) can call. Routes a patient record through
whichever models are actually applicable to the data available for
that patient, and returns both the individual model outputs and the
final combined joint prediction.

INPUT SCHEMA:
`patient` is a dict. Keys used by each model are listed below; any
model whose required keys are missing is skipped (not guessed at) and
clearly flagged as unavailable in the output.

Model 2 / Model 2b (women-only; require ALL of these):
  age, bmi, cycle_regularity (2=regular/4=irregular), cycle_length_days,
  weight_gain, hirsutism, skin_darkening, hair_loss, acne, fast_food,
  regular_exercise, bp_systolic, bp_diastolic, fsh, lh, fsh_lh_ratio,
  amh, prl, vit_d3, progesterone, random_blood_sugar,
  follicle_count_left, follicle_count_right, avg_follicle_size_left,
  avg_follicle_size_right, endometrium_mm, tsh

Model 3 (clinical criteria; requires):
  cycle_regularity, hirsutism, lh, fsh, follicle_count_left,
  follicle_count_right, tsh

Model 1 (male-skewed; requires the FULL clinical thyroid panel --
notably NOT part of a typical PCOS workup, which is the whole point):
  age, sex, pregnant, sick, tsh, t3, t4, t4_uptake, free_thyroxine_index

DEPLOYMENT HEURISTIC (documented, not re-validated against ground truth):
If Model 1 IS available for a patient (full thyroid panel present), its
thyroid_dysfunction probability is substituted for Model 2's in the
meta-learner's input slot, since Model 1's real-world AUC (0.995) is far
higher than Model 2's on that axis (0.605) -- this substitution has not
itself been validated against joint ground truth (no dataset in this
project has both a full thyroid panel AND a PCOS diagnosis on the same
patients), so treat blended predictions as a reasonable heuristic, not
as validated to the same standard as the meta-learner's own test metrics.
"""

import os
import numpy as np
import pandas as pd
import joblib

from pipeline_common import CLASS_NAMES
from train_model1 import encode_features as encode_model1_features, FEATURE_COLS as MODEL1_FEATURE_COLS
from train_model2 import clean_features as clean_model2_features, FEATURE_COLS as MODEL2_FEATURE_COLS
from train_model2b_pcos import clean_features as clean_model2b_features
from model3 import pcos_criteria_score, thyroid_criteria_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# POST-REVIEW FIX: these used to be hand-typed duplicates of FEATURE_COLS
# in train_model2.py/train_model2b_pcos.py/train_model1.py. Now derived
# directly from those modules' own constants, so a change there can no
# longer silently desync predict.py from what the models were trained on.
MODEL2_REQUIRED_KEYS = MODEL2_FEATURE_COLS
MODEL2B_REQUIRED_KEYS = MODEL2_FEATURE_COLS + ["tsh"]
MODEL3_REQUIRED_KEYS = ["cycle_regularity", "hirsutism", "lh", "fsh",
                        "follicle_count_left", "follicle_count_right", "tsh", "amh"]
MODEL1_REQUIRED_KEYS = MODEL1_FEATURE_COLS  # already ["age","sex","pregnant","sick","tsh","t3","t4","t4_uptake","free_thyroxine_index"]

_models_cache = {}


def _load_models():
    """Lazy-load + cache models so repeated predict() calls don't re-read
    from disk every time."""
    if _models_cache:
        return _models_cache
    for key, fname in [
        ("model1", "model1_generic_baseline.joblib"),
        ("model2", "model2_women_only.joblib"),
        ("model2b", "model2b_women_only_pcos.joblib"),
        ("meta", "meta_learner.joblib"),
    ]:
        path = os.path.join(MODEL_DIR, fname)
        _models_cache[key] = joblib.load(path) if os.path.exists(path) else None
    return _models_cache


def _has_keys(patient: dict, keys: list) -> bool:
    return all(k in patient and patient[k] is not None for k in keys)


def _patient_to_row(patient: dict) -> pd.DataFrame:
    """Wraps a single patient dict as a one-row DataFrame, the same shape
    the clean_features()/encode_features() functions expect (they were
    written for CSV-loaded DataFrames, not raw dicts)."""
    return pd.DataFrame([patient])


def _model2_features(patient: dict) -> pd.DataFrame:
    row = dict(patient)
    if row.get("cycle_regularity") == 5:
        row["cycle_regularity"] = 4  # same stray-value fix as training
    return clean_model2_features(_patient_to_row(row))


def _model2b_features(patient: dict) -> pd.DataFrame:
    row = dict(patient)
    if row.get("cycle_regularity") == 5:
        row["cycle_regularity"] = 4
    return clean_model2b_features(_patient_to_row(row))


def _model1_features(patient: dict) -> pd.DataFrame:
    row = dict(patient)
    # encode_model1_features expects raw 't'/'f' strings and 'male'/'female',
    # matching the CSVs it was written for -- normalize a few common input
    # shapes (booleans, 0/1 ints) to that same convention here, in ONE place,
    # rather than re-implementing the encoding logic by hand.
    row["pregnant"] = "t" if row.get("pregnant") in (1, "t", True) else "f"
    row["sick"] = "t" if row.get("sick") in (1, "t", True) else "f"
    return encode_model1_features(_patient_to_row(row))


def _model3_scores(patient: dict) -> dict:
    """Reuses the EXACT same rule functions from model3.py
    (Rotterdam criteria + AMH alternative + TSH threshold) instead of a
    second, separately-maintained copy of those thresholds."""
    row = _patient_to_row(patient)
    pcos_scores = pcos_criteria_score(row)
    thyroid_call = thyroid_criteria_score(row["tsh"])
    return {
        "pcos_criteria_count": int(pcos_scores["pcos_criteria_count"].iloc[0]),
        "pcos_rotterdam_call": int(pcos_scores["pcos_rotterdam_call"].iloc[0]),
        "thyroid_criteria_call": int(thyroid_call.iloc[0]),
    }


def predict(patient: dict) -> dict:
    """Main entry point. Returns a dict with every model's individual
    output (or None + a reason if that model couldn't run), plus the
    final joint prediction if enough models were available to produce
    one."""
    models = _load_models()
    result = {"model1": None, "model2_thyroid": None, "model2b_pcos": None, "model3": None, "joint": None}

    # Model 1: male-skewed, thyroid axis only (always abstains on PCOS)
    if models["model1"] is not None and _has_keys(patient, MODEL1_REQUIRED_KEYS):
        proba = models["model1"].predict_proba(_model1_features(patient))[0, 1]
        result["model1"] = {"thyroid_dysfunction_proba": float(proba), "pcos_proba": None,
                             "note": "Model 1 structurally cannot assess PCOS risk (no ovaries in its training "
                                     "population). NOTE: earlier docs called this model 'male-skewed', but its "
                                     "training cohort is actually 69% female -- sex-disaggregated evaluation "
                                     "(see model1_metrics.json) shows no meaningful AUC gap by sex on this data."}
    else:
        result["model1"] = {"available": False,
                             "reason": "Missing full thyroid panel (t3/t4/t4_uptake/free_thyroxine_index) -- "
                                       "typical of a PCOS-focused workup that doesn't order these."}

    # Model 2: women-only, thyroid axis
    model2_thyroid_proba = None
    if models["model2"] is not None and _has_keys(patient, MODEL2_REQUIRED_KEYS):
        model2_thyroid_proba = float(models["model2"].predict_proba(_model2_features(patient))[0, 1])
        result["model2_thyroid"] = {"thyroid_dysfunction_proba": model2_thyroid_proba}
    else:
        result["model2_thyroid"] = {"available": False, "reason": "Missing required PCOS-cohort features."}

    # Model 2b: women-only, PCOS axis
    model2_pcos_proba = None
    if models["model2b"] is not None and _has_keys(patient, MODEL2B_REQUIRED_KEYS):
        model2_pcos_proba = float(models["model2b"].predict_proba(_model2b_features(patient))[0, 1])
        result["model2b_pcos"] = {"pcos_proba": model2_pcos_proba}
    else:
        result["model2b_pcos"] = {"available": False, "reason": "Missing required PCOS-cohort features."}

    # Model 3: clinical criteria, both axes
    model3_scores = None
    if _has_keys(patient, MODEL3_REQUIRED_KEYS):
        model3_scores = _model3_scores(patient)
        result["model3"] = model3_scores
    else:
        result["model3"] = {"available": False, "reason": "Missing required clinical criteria inputs."}

    # Meta-learner: needs model2_thyroid_proba, model2_pcos_proba, and
    # model3's scores. If Model 1 is ALSO available, substitute its
    # (more reliable) thyroid probability into that input slot -- see
    # the deployment-heuristic note in the module docstring.
    if model2_thyroid_proba is not None and model2_pcos_proba is not None and model3_scores is not None and models["meta"] is not None:
        thyroid_input = model2_thyroid_proba
        heuristic_applied = False
        if result["model1"] and result["model1"].get("thyroid_dysfunction_proba") is not None:
            thyroid_input = result["model1"]["thyroid_dysfunction_proba"]
            heuristic_applied = True

        meta_features = pd.DataFrame([{
            "model2_thyroid_proba": thyroid_input,
            "model2_pcos_proba": model2_pcos_proba,
            "pcos_criteria_count": model3_scores["pcos_criteria_count"],
            "thyroid_criteria_call": model3_scores["thyroid_criteria_call"],
        }])
        joint_probs = models["meta"].predict_proba(meta_features)[0]
        joint_class = int(np.argmax(joint_probs))

        result["joint"] = {
            "class": joint_class,
            "class_name": CLASS_NAMES[joint_class],
            "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(joint_probs)},
            "used_model1_for_thyroid_input": heuristic_applied,
            "warning": (
                "This prediction substitutes Model 1's thyroid probability into a meta-learner "
                "input slot that was calibrated on Model 2's output distribution during training. "
                "This substitution has NOT been validated against real joint ground truth (no "
                "dataset used here has both a full thyroid panel AND a PCOS diagnosis on the same "
                "patients). Treat this specific prediction as a documented heuristic, not to the "
                "same evidence standard as the meta-learner's own reported test metrics."
            ) if heuristic_applied else None,
        }
    else:
        result["joint"] = {"available": False,
                            "reason": "Need Model 2, Model 2b, and Model 3 all available to run the meta-learner."}

    return result


if __name__ == "__main__":
    # Smoke test with a synthetic patient -- mirrors a real PCOS-cohort row.
    example_patient = {
        "age": 28, "bmi": 24.5, "cycle_regularity": 4, "cycle_length_days": 45,
        "weight_gain": 1, "hirsutism": 1, "skin_darkening": 1, "hair_loss": 0,
        "acne": 1, "fast_food": 1, "regular_exercise": 0, "bp_systolic": 120,
        "bp_diastolic": 80, "fsh": 5.2, "lh": 9.8, "fsh_lh_ratio": 0.53,
        "amh": 8.4, "prl": 15.0, "vit_d3": 20.0, "progesterone": 0.8,
        "random_blood_sugar": 95, "follicle_count_left": 14, "follicle_count_right": 16,
        "avg_follicle_size_left": 4.2, "avg_follicle_size_right": 4.5,
        "endometrium_mm": 7.0, "tsh": 2.1,
    }
    import json
    print(json.dumps(predict(example_patient), indent=2))