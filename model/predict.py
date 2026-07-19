"""
predict.py
Step 7 of the Women's Hormonal Health ensemble model pipeline.

The single function the rest of the system calls:

    from predict import predict
    result = predict(patient_dict)

`patient_dict` follows the PatientInput schema (see api.py's PatientInput /
pdf_to_json.py's TEMPLATE) -- 32 fields, ANY of which may be missing or None.
That's expected, not an error: a doctor's PDF upload will often only have
some of the fields a given model needs, and a hand-filled fallback form may
leave others blank. Each model below only runs if every field IT needs is
present; otherwise it abstains (returns None + a plain-English reason)
instead of silently guessing with a default value. This is the `_has_keys`
policy referenced in pdf_to_json.py and model_adapter.py's docstrings.

WHAT RUNS, AND WHEN
--------------------
Model 1  (generic clinical baseline, thyroid_dysfunction)
    needs: age, sex, pregnant, sick, tsh, t3, t4, t4_uptake, free_thyroxine_index
    i.e. a FULL clinical thyroid panel. Most PCOS-style workups won't have
    this -- that's the point being demonstrated, not a bug.

Model 2  (women-only, thyroid_dysfunction) and
Model 2b (women-only, pcos_diagnosis)
    both need the same 26-field PCOS-cohort feature set (age, bmi, cycle
    info, symptom checklist, hormone panel, follicle/ultrasound fields).

Model 3  (Rotterdam criteria + TSH threshold -- not trained, rule-based)
    PCOS axis needs: cycle_regularity, hirsutism, lh, fsh,
                      follicle_count_left, follicle_count_right
    Thyroid axis needs: tsh
    The two axes are scored independently, so a patient can get a PCOS
    criteria score even with no TSH, or vice versa.

Meta-learner (calibrated 4-way joint: neither / thyroid_only / pcos_only /
both)
    needs Model 2's thyroid proba + Model 2b's PCOS proba + BOTH of Model
    3's scores -- i.e. it can only run if Model 2, Model 2b, and Model 3
    (both axes) all ran. This mirrors exactly how meta_learner.py was
    trained: on PCOS-cohort meta-features only. Model 1 is structurally
    excluded from the meta-learner (see meta_learner.py's docstring) --
    it is reported separately, never fed into the joint call.

MODEL 1 SUBSTITUTION HEURISTIC (documented, not hidden):
When a patient happens to have BOTH a full thyroid panel (Model 1 ran)
AND enough PCOS-cohort fields (Model 2 ran), Model 1's thyroid estimate
is clinically more reliable -- it sees T3/T4/T4U/FTI, which Model 2 never
does. The meta-learner was never trained with Model 1 as an input feature
(that data doesn't exist for the PCOS cohort), so we cannot literally
splice its output into the meta-learner's math. What we DO is surface
this to the doctor as an explicit warning and flag
`used_model1_for_thyroid_input=True` on the joint result, so
model_adapter.py can tell the UI "Model 1's number is the trustworthy one
here, even though the joint class weights technically ran on Model 2's."

SCHEMA RECONCILIATION NOTE:
train_model1.py's `encode_features()` was written against the UCI
hypothyroid dataset's raw encoding, where `pregnant`/`sick` are the
literal strings "t"/"f". Everywhere else in this system (api.py's
PatientInput, pdf_to_json.py's TEMPLATE) represents the same fields as
0/1 integers, since that's a saner interface for callers. `_model1_row()`
below is the one place that reconciles the two -- converts the 0/1 ints
coming in at the boundary into the "t"/"f" strings `encode_features()`
expects -- so that mismatch doesn't leak out and doesn't require touching
the already-tested training script.
"""

import os
import joblib
import pandas as pd

from model3 import pcos_criteria_score, thyroid_criteria_score
from train_model1 import encode_features as _encode_features_model1, FEATURE_COLS as MODEL1_FEATURE_COLS
from train_model2 import clean_features as _clean_features_model2
from train_model2b_pcos import clean_features as _clean_features_model2b
from pipeline_common import CLASS_NAMES

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# --- field requirements, kept in one place and reused for both routing
# and for the human-readable "missing fields" messages. Must stay in sync
# with pdf_to_json.py's FIELDS_BY_MODEL. ---
FIELDS_MODEL1 = ["age", "sex", "pregnant", "sick", "tsh", "t3", "t4", "t4_uptake", "free_thyroxine_index"]

FIELDS_MODEL2_2B = [
    "age", "bmi", "cycle_regularity", "cycle_length_days", "weight_gain",
    "hirsutism", "skin_darkening", "hair_loss", "acne", "fast_food",
    "regular_exercise", "bp_systolic", "bp_diastolic", "fsh", "lh",
    "fsh_lh_ratio", "amh", "prl", "vit_d3", "progesterone",
    "random_blood_sugar", "follicle_count_left", "follicle_count_right",
    "avg_follicle_size_left", "avg_follicle_size_right", "endometrium_mm", "tsh",
]

FIELDS_MODEL3_PCOS = ["cycle_regularity", "hirsutism", "lh", "fsh",
                       "follicle_count_left", "follicle_count_right"]
FIELDS_MODEL3_THYROID = ["tsh"]

# Loaded once per process and reused -- these files don't change at runtime,
# and joblib.load() is not free enough to redo on every request.
_MODEL_CACHE = {}


def _load(filename):
    if filename not in _MODEL_CACHE:
        path = os.path.join(MODEL_DIR, filename)
        _MODEL_CACHE[filename] = joblib.load(path) if os.path.exists(path) else None
    return _MODEL_CACHE[filename]


def _missing_keys(patient: dict, keys) -> list:
    return [k for k in keys if patient.get(k) is None]


def _has_keys(patient: dict, keys) -> bool:
    """A field that's present but None counts as missing -- we never want
    a model trained on real labs silently treating "unknown" as a value."""
    return not _missing_keys(patient, keys)


def _row(patient: dict) -> pd.DataFrame:
    """Every helper this module reuses (clean_features_*, model3's scorers)
    is written against a DataFrame/Series, so single-patient predict()
    calls go through a one-row DataFrame rather than reimplementing that
    logic per-field here."""
    return pd.DataFrame([patient])


def _model1_row(patient: dict) -> pd.DataFrame:
    """See module docstring: bridges the 0/1-int schema used everywhere
    else in this system to the 't'/'f'-string schema train_model1.py's
    encode_features() expects."""
    bridged = dict(patient)
    bridged["pregnant"] = "t" if patient.get("pregnant") == 1 else "f"
    bridged["sick"] = "t" if patient.get("sick") == 1 else "f"
    return pd.DataFrame([bridged])


# ---------------------------------------------------------------------------
# Individual model runners -- each one abstains cleanly if its fields
# aren't present, and never raises for a missing/incomplete patient.
# ---------------------------------------------------------------------------

def _run_model1(patient: dict) -> dict:
    if not _has_keys(patient, FIELDS_MODEL1):
        return {"thyroid_dysfunction_proba": None,
                "reason": f"missing fields: {', '.join(_missing_keys(patient, FIELDS_MODEL1))}"}
    model = _load("model1_generic_baseline.joblib")
    if model is None:
        return {"thyroid_dysfunction_proba": None, "reason": "model1 artifact not found on disk"}
    X = _encode_features_model1(_model1_row(patient))[MODEL1_FEATURE_COLS]
    if X.isna().any(axis=None):
        return {"thyroid_dysfunction_proba": None,
                "reason": "sex must be 'male' or 'female' -- got an unrecognized value"}
    proba = float(model.predict_proba(X)[0, 1])
    return {"thyroid_dysfunction_proba": proba, "reason": None}


def _run_model2_thyroid(patient: dict) -> dict:
    if not _has_keys(patient, FIELDS_MODEL2_2B):
        return {"thyroid_dysfunction_proba": None,
                "reason": f"missing fields: {', '.join(_missing_keys(patient, FIELDS_MODEL2_2B))}"}
    model = _load("model2_women_only.joblib")
    if model is None:
        return {"thyroid_dysfunction_proba": None, "reason": "model2 artifact not found on disk"}
    X = _clean_features_model2(_row(patient))
    proba = float(model.predict_proba(X)[0, 1])
    return {"thyroid_dysfunction_proba": proba, "reason": None}


def _run_model2b_pcos(patient: dict) -> dict:
    if not _has_keys(patient, FIELDS_MODEL2_2B):
        return {"pcos_proba": None,
                "reason": f"missing fields: {', '.join(_missing_keys(patient, FIELDS_MODEL2_2B))}"}
    model = _load("model2b_women_only_pcos.joblib")
    if model is None:
        return {"pcos_proba": None, "reason": "model2b artifact not found on disk"}
    X = _clean_features_model2b(_row(patient))
    proba = float(model.predict_proba(X)[0, 1])
    return {"pcos_proba": proba, "reason": None}


def _run_model3(patient: dict) -> dict:
    """Both Rotterdam-criteria axes are scored independently, so a
    patient missing TSH can still get a PCOS criteria count, and vice
    versa -- unlike the trained models, there's no reason to require
    the union of both fields sets here."""
    result = {"pcos_criteria_count": None, "pcos_rotterdam_call": None,
              "thyroid_criteria_call": None, "reason": None}
    reasons = []
    row = _row(patient)

    if _has_keys(patient, FIELDS_MODEL3_PCOS):
        scores = pcos_criteria_score(row)
        result["pcos_criteria_count"] = int(scores["pcos_criteria_count"].iloc[0])
        result["pcos_rotterdam_call"] = bool(scores["pcos_rotterdam_call"].iloc[0])
    else:
        reasons.append(f"PCOS criteria missing: {', '.join(_missing_keys(patient, FIELDS_MODEL3_PCOS))}")

    if _has_keys(patient, FIELDS_MODEL3_THYROID):
        result["thyroid_criteria_call"] = bool(thyroid_criteria_score(row["tsh"]).iloc[0])
    else:
        reasons.append("Thyroid criteria missing: tsh")

    if reasons:
        result["reason"] = "; ".join(reasons)
    return result


def _run_joint(model1: dict, model2_thyroid: dict, model2b_pcos: dict, model3: dict) -> dict:
    """The calibrated 4-way joint prediction. Only runs if Model 2, Model
    2b, and BOTH Model 3 axes produced a value -- exactly the meta-feature
    set meta_learner.py was trained on (see build_meta_features there)."""
    missing = []
    if model2_thyroid.get("thyroid_dysfunction_proba") is None:
        missing.append("model2_thyroid_proba (Model 2 could not run)")
    if model2b_pcos.get("pcos_proba") is None:
        missing.append("model2_pcos_proba (Model 2b could not run)")
    if model3.get("pcos_criteria_count") is None:
        missing.append("pcos_criteria_count (Model 3 PCOS axis could not run)")
    if model3.get("thyroid_criteria_call") is None:
        missing.append("thyroid_criteria_call (Model 3 thyroid axis could not run)")

    if missing:
        return {"probabilities": None, "used_model1_for_thyroid_input": False,
                "warning": None, "reason": "joint ensemble needs: " + "; ".join(missing)}

    meta_model = _load("meta_learner.joblib")
    if meta_model is None:
        return {"probabilities": None, "used_model1_for_thyroid_input": False,
                "warning": None, "reason": "meta_learner artifact not found on disk"}

    # Column NAME and ORDER must match meta_learner.py's build_meta_features()
    # exactly -- sklearn's LogisticRegression.predict_proba does not re-sort
    # a DataFrame's columns to match what it saw at fit time.
    X_meta = pd.DataFrame([{
        "model2_thyroid_proba": model2_thyroid["thyroid_dysfunction_proba"],
        "model2_pcos_proba": model2b_pcos["pcos_proba"],
        "pcos_criteria_count": model3["pcos_criteria_count"],
        "thyroid_criteria_call": int(model3["thyroid_criteria_call"]),
    }])
    proba = meta_model.predict_proba(X_meta)[0]
    probabilities = {CLASS_NAMES[i]: float(p) for i, p in zip(meta_model.classes_, proba)}

    used_model1 = model1.get("thyroid_dysfunction_proba") is not None
    warning = None
    if used_model1:
        warning = (
            "A full clinical thyroid panel was available, so Model 1's more "
            "reliable thyroid_dysfunction probability should be shown to the "
            "doctor in place of Model 2's estimate. The joint class weights "
            "above still run on Model 2's number mechanically, since the "
            "meta-learner was never trained with Model 1 as an input feature "
            "(structurally unavailable for the PCOS cohort it was trained "
            "on -- see meta_learner.py)."
        )

    return {"probabilities": probabilities, "used_model1_for_thyroid_input": used_model1,
            "warning": warning, "reason": None}


def predict(patient: dict) -> dict:
    """The one function the rest of the system calls.

    Never raises on missing/partial input -- every field is optional, and
    every model abstains (rather than guessing) when it doesn't have what
    it needs. Returns a dict with keys: model1, model2_thyroid,
    model2b_pcos, model3, joint. See module docstring for each key's shape.
    """
    patient = dict(patient)  # never mutate the caller's dict

    model1_result = _run_model1(patient)
    model2_thyroid_result = _run_model2_thyroid(patient)
    model2b_pcos_result = _run_model2b_pcos(patient)
    model3_result = _run_model3(patient)
    joint_result = _run_joint(model1_result, model2_thyroid_result, model2b_pcos_result, model3_result)

    return {
        "model1": model1_result,
        "model2_thyroid": model2_thyroid_result,
        "model2b_pcos": model2b_pcos_result,
        "model3": model3_result,
        "joint": joint_result,
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            demo_patient = json.load(f)
    else:
        # A patient with every field filled in, so a smoke test exercises
        # every model AND the joint ensemble in one run.
        demo_patient = {
            "age": 28, "bmi": 27.5, "cycle_regularity": 4, "cycle_length_days": 45,
            "weight_gain": 1, "hirsutism": 1, "skin_darkening": 1, "hair_loss": 1,
            "acne": 1, "fast_food": 1, "regular_exercise": 0, "bp_systolic": 120,
            "bp_diastolic": 80, "fsh": 5.0, "lh": 12.0, "fsh_lh_ratio": 2.4,
            "amh": 7.2, "prl": 18.0, "vit_d3": 22.0, "progesterone": 0.8,
            "random_blood_sugar": 95, "follicle_count_left": 14, "follicle_count_right": 16,
            "avg_follicle_size_left": 4.0, "avg_follicle_size_right": 4.2,
            "endometrium_mm": 8.0, "tsh": 6.2, "sex": "female", "pregnant": 0,
            "sick": 0, "t3": 1.8, "t4": 7.5, "t4_uptake": 28.0, "free_thyroxine_index": 6.9,
        }

    print(json.dumps(predict(demo_patient), indent=2))
