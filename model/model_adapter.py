"""
model_adapter.py

The glue between the web app's PDF upload and the ensemble model.

    web app --(PDF)--> analyze_pdf() --(dict)--> web app renders it

analyze_pdf() is the ONE function app.py's study_analysis() route should
call. It does three things, each already built separately:

    1. pdf_to_json.pdf_to_patient_json(pdf_path)  -- PDF -> PatientInput dict
    2. predict.predict(patient_dict)              -- PatientInput -> ensemble result
    3. _to_display(result)                        -- ensemble result -> UI shape

Steps 1 and 2 already exist as their own tested modules (pdf_to_json.py,
predict.py). This file only adds step 3 and the one-line orchestration,
because nothing today converts predict()'s rich per-model dict into
something study_analysis.html can render.

OUTPUT CONTRACT (what the web app receives and should render):
{
  "diagnoses": [
      {"condition": str, "confidence": float in [0,1]},
      ...  # sorted descending by confidence -- render row 0 as "Top Match"
  ],
  "summary": str,          # one paragraph, always present, always human-readable
  "warnings": [str, ...],  # zero or more caveats to surface to the doctor
  "joint_available": bool, # True if the calibrated 4-way ensemble ran
}

This shape is intentionally stable regardless of which underlying models
ran -- the web app template never needs to know whether it's looking at
a full joint prediction or a partial fallback. It only ever reads
diagnoses / summary / warnings.
"""
from pdf_to_json import pdf_to_patient_json
from predict import predict

CONDITION_LABELS = {
    "neither": "Neither PCOS nor Thyroid Dysfunction",
    "thyroid_only": "Thyroid Dysfunction Only",
    "pcos_only": "PCOS Only",
    "both": "Both PCOS and Thyroid Dysfunction",
}


def analyze_pdf(pdf_path: str) -> dict:
    """Full pipeline: PDF on disk -> display-ready result dict."""
    patient = pdf_to_patient_json(pdf_path)
    result = predict(patient)
    return _to_display(result, patient)


def _to_display(result: dict, patient: dict) -> dict:
    joint = result.get("joint") or {}
    warnings = []

    # --- Case 1: the calibrated 4-way joint ensemble ran ---
    if joint.get("probabilities"):
        diagnoses = [
            {"condition": CONDITION_LABELS[name], "confidence": float(prob)}
            for name, prob in joint["probabilities"].items()
        ]
        diagnoses.sort(key=lambda d: d["confidence"], reverse=True)
        top = diagnoses[0]

        summary = (
            f"The ensemble model's strongest signal is '{top['condition']}' "
            f"at {top['confidence'] * 100:.0f}% confidence, combining the "
            f"women-only hormone model, Rotterdam clinical criteria, and "
            f"(when available) the general thyroid model."
        )

        if joint.get("used_model1_for_thyroid_input"):
            warnings.append(
                "This patient had a full thyroid panel available, so the general "
                "thyroid model's (more reliable) output was substituted into the "
                "ensemble in place of the women-only model's thyroid estimate. "
                + (joint.get("warning") or "")
            )

        return {
            "diagnoses": diagnoses,
            "summary": summary,
            "warnings": warnings,
            "joint_available": True,
        }

    # --- Case 2: joint ensemble couldn't run -- fall back to whichever
    # individual models DID have enough data, and say plainly why the
    # calibrated joint estimate isn't available. ---
    diagnoses = []
    missing_reasons = []

    thyroid_proba = None
    thyroid_source = None
    m2t = result.get("model2_thyroid") or {}
    m1 = result.get("model1") or {}
    if m1.get("thyroid_dysfunction_proba") is not None:
        thyroid_proba = m1["thyroid_dysfunction_proba"]
        thyroid_source = "general thyroid model"
    elif m2t.get("thyroid_dysfunction_proba") is not None:
        thyroid_proba = m2t["thyroid_dysfunction_proba"]
        thyroid_source = "women-only hormone model"
    elif m2t.get("reason"):
        missing_reasons.append(f"Thyroid Dysfunction: {m2t['reason']}")

    if thyroid_proba is not None:
        diagnoses.append({"condition": "Thyroid Dysfunction", "confidence": float(thyroid_proba)})

    m2b = result.get("model2b_pcos") or {}
    if m2b.get("pcos_proba") is not None:
        diagnoses.append({"condition": "PCOS", "confidence": float(m2b["pcos_proba"])})
    elif m2b.get("reason"):
        missing_reasons.append(f"PCOS: {m2b['reason']}")

    m3 = result.get("model3") or {}
    if m3.get("pcos_rotterdam_call") is not None:
        warnings.append(
            f"Rule-based Rotterdam criteria (not a probability): "
            f"{m3['pcos_criteria_count']}/3 criteria met "
            f"({'meets' if m3['pcos_rotterdam_call'] else 'does not meet'} the "
            f">=2 threshold for a clinical PCOS call)."
        )

    if diagnoses:
        diagnoses.sort(key=lambda d: d["confidence"], reverse=True)
        source_note = f" (from the {thyroid_source})" if thyroid_source else ""
        summary = (
            "The full calibrated joint ensemble could not run for this patient "
            "because some required fields were missing, so the results below are "
            f"from individual models only{source_note}, not the validated joint estimate. "
            + " ".join(missing_reasons)
        )
    else:
        diagnoses = [{"condition": "Unknown - Insufficient Data", "confidence": 0.0}]
        summary = (
            "No model could produce a prediction for this patient -- required "
            "fields were missing from the uploaded document. " + " ".join(missing_reasons)
        )

    warnings.insert(0, "Joint (combined PCOS x Thyroid) prediction unavailable -- see summary.")

    return {
        "diagnoses": diagnoses,
        "summary": summary,
        "warnings": warnings,
        "joint_available": False,
    }