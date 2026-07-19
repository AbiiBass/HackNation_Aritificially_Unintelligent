"""
model_adapter.py

Glue between the web app's PDF upload and the ensemble model:

    web app --(PDF)--> analyze_pdf() --(dict)--> web app renders it

analyze_pdf() is the one function app.py's study_analysis() route calls:

    1. pdf_to_json.pdf_to_patient_json(pdf_path)  -- PDF -> PatientInput dict
    2. predict.predict(patient_dict)              -- PatientInput -> ensemble result
    3. _to_display(result)                        -- ensemble result -> UI shape

OUTPUT CONTRACT (what the web app receives and renders):
{
  "diagnoses": [
      {"condition": str, "confidence": float in [0,1]},
      ...  # sorted descending by confidence -- row 0 is "Top Match"
  ],
  "summary": str,          # one paragraph, always present
  "warnings": [str, ...],  # caveats to show the doctor
  "joint_available": bool, # True if the calibrated 4-way ensemble ran
}

Shape stays the same regardless of which underlying models ran; the web
app template only ever reads diagnoses / summary / warnings.
"""
from pdf_to_json import pdf_to_patient_json
from predict import predict, FIELDS_MODEL1, FIELDS_MODEL2B

CONDITION_LABELS = {
    "neither": "Neither PCOS nor Thyroid Dysfunction",
    "thyroid_only": "Thyroid Dysfunction Only",
    "pcos_only": "PCOS Only",
    "both": "Both PCOS and Thyroid Dysfunction",
}

# Clinical names for fields, so doctors see "TSH" / "AMH", not internal keys.
FIELD_LABELS = {
    "age": "age", "bmi": "BMI", "cycle_regularity": "menstrual cycle regularity",
    "cycle_length_days": "cycle length", "weight_gain": "weight gain history",
    "hirsutism": "hirsutism", "skin_darkening": "skin darkening", "hair_loss": "hair loss",
    "acne": "acne", "fast_food": "diet history", "regular_exercise": "exercise history",
    "bp_systolic": "blood pressure", "bp_diastolic": "blood pressure",
    "fsh": "FSH", "lh": "LH", "fsh_lh_ratio": "FSH/LH ratio", "amh": "AMH",
    "prl": "prolactin", "vit_d3": "vitamin D", "progesterone": "progesterone",
    "random_blood_sugar": "blood sugar", "follicle_count_left": "ovarian follicle count",
    "follicle_count_right": "ovarian follicle count", "avg_follicle_size_left": "follicle size",
    "avg_follicle_size_right": "follicle size", "endometrium_mm": "endometrial thickness",
    "tsh": "TSH", "t3": "T3", "t4": "T4", "t4_uptake": "T4 uptake",
    "free_thyroxine_index": "free thyroxine index", "sex": "sex",
    "pregnant": "pregnancy status", "sick": "acute illness status",
}


def _missing_labels(patient: dict, keys) -> list:
    """Clinical names for the fields in `keys` missing from `patient`,
    de-duplicated, in order (e.g. ["TSH", "T3"])."""
    labels = []
    for k in keys:
        if patient.get(k) is None:
            label = FIELD_LABELS.get(k, k)
            if label not in labels:
                labels.append(label)
    return labels


def analyze_pdf(pdf_path: str) -> dict:
    """Full pipeline: PDF on disk -> display-ready result dict."""
    patient = pdf_to_patient_json(pdf_path)
    result = predict(patient)
    return _to_display(result, patient)


def _to_display(result: dict, patient: dict) -> dict:
    joint = result.get("joint") or {}
    warnings = []

    # Case 1: full joint estimate available
    if joint.get("probabilities"):
        diagnoses = [
            {"condition": CONDITION_LABELS[name], "confidence": float(prob)}
            for name, prob in joint["probabilities"].items()
        ]
        diagnoses.sort(key=lambda d: d["confidence"], reverse=True)
        top = diagnoses[0]

        summary = (
            f"Most likely result: {top['condition']} ({top['confidence'] * 100:.0f}% likelihood). "
            f"This is based on the patient's menstrual cycle and symptom history, hormone panel, "
            f"and ultrasound findings, together with their thyroid lab results."
        )

        if joint.get("used_model1_for_thyroid_input"):
            warnings.append(
                "This patient has a complete thyroid lab panel on file, so the thyroid result "
                "above reflects those labs directly -- a more reliable read than estimating "
                "thyroid risk from hormone and symptom data alone."
            )

        return {
            "diagnoses": diagnoses,
            "summary": summary,
            "warnings": warnings,
            "joint_available": True,
        }

    # Case 2: fall back to individual model results, note what's missing
    diagnoses = []

    thyroid_proba = None
    thyroid_source = None
    m2t = result.get("model2_thyroid") or {}
    m1 = result.get("model1") or {}
    if m1.get("thyroid_dysfunction_proba") is not None:
        thyroid_proba = m1["thyroid_dysfunction_proba"]
        thyroid_source = "the patient's thyroid lab panel"
    elif m2t.get("thyroid_dysfunction_proba") is not None:
        thyroid_proba = m2t["thyroid_dysfunction_proba"]
        thyroid_source = "hormone and symptom findings"

    if thyroid_proba is not None:
        diagnoses.append({"condition": "Thyroid Dysfunction", "confidence": float(thyroid_proba)})

    m2b = result.get("model2b_pcos") or {}
    if m2b.get("pcos_proba") is not None:
        diagnoses.append({"condition": "PCOS", "confidence": float(m2b["pcos_proba"])})

    m3 = result.get("model3") or {}
    if m3.get("pcos_rotterdam_call") is not None:
        count = m3["pcos_criteria_count"]
        meets = m3["pcos_rotterdam_call"]
        warnings.append(
            f"By the Rotterdam clinical criteria, this patient meets {count} of 3 diagnostic "
            f"criteria for PCOS -- {'this meets' if meets else 'this falls short of'} the "
            f"standard threshold of 2 or more for a clinical PCOS call."
        )

    if diagnoses:
        diagnoses.sort(key=lambda d: d["confidence"], reverse=True)
        top = diagnoses[0]
        missing = _missing_labels(patient, FIELDS_MODEL1) if thyroid_proba is None else []
        missing += _missing_labels(patient, FIELDS_MODEL2B) if not m2b.get("pcos_proba") else []
        missing_note = f" Missing from this document: {', '.join(missing)}." if missing else ""
        summary = (
            f"A full combined PCOS/thyroid estimate wasn't possible because some information was "
            f"missing from this document. Based on what was available, the strongest result is "
            f"{top['condition']} at {top['confidence'] * 100:.0f}% likelihood, from {thyroid_source or 'the available findings'}."
            f"{missing_note}"
        )
    else:
        missing = _missing_labels(patient, FIELDS_MODEL1) + _missing_labels(patient, FIELDS_MODEL2B)
        missing = list(dict.fromkeys(missing))
        diagnoses = [{"condition": "Unknown - Insufficient Data", "confidence": 0.0}]
        summary = (
            "No result could be produced for this patient -- key information needed for either "
            "the PCOS or thyroid assessment was missing from this document."
            + (f" Missing: {', '.join(missing)}." if missing else "")
        )

    warnings.insert(0, "A combined PCOS/thyroid estimate could not be generated -- see the result below.")

    return {
        "diagnoses": diagnoses,
        "summary": summary,
        "warnings": warnings,
        "joint_available": False,
    }