"""
Tests for model_adapter.py's doctor-facing output: the result keys
(diagnoses/summary/warnings/joint_available), and that the text never
leaks internal ML jargon or raw field names.
"""
import pytest

from model_adapter import _to_display
from predict import predict

FULL_PATIENT = {
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

# ML jargon that must never appear in doctor-facing text.
JARGON_TERMS = [
    "ensemble", "meta-learner", "meta learner", "calibrated",
    "model 1", "model1", "model 2", "model2", "model 3", "model3",
    "joint ensemble", "logistic regression", "xgboost",
]


def _patient(missing=()):
    p = dict(FULL_PATIENT)
    for k in missing:
        p[k] = None
    return p


def _assert_jargon_free(text: str):
    lowered = text.lower()
    for term in JARGON_TERMS:
        assert term not in lowered, f"found ML jargon {term!r} in doctor-facing text: {text!r}"


def test_output_contract_keys_present_full_case():
    result = predict(_patient())
    display = _to_display(result, _patient())
    assert set(display.keys()) == {"diagnoses", "summary", "warnings", "joint_available"}
    assert display["joint_available"] is True
    assert isinstance(display["diagnoses"], list) and display["diagnoses"]
    assert isinstance(display["summary"], str) and display["summary"]
    assert isinstance(display["warnings"], list)


def test_diagnoses_sorted_descending_by_confidence():
    result = predict(_patient())
    display = _to_display(result, _patient())
    confidences = [d["confidence"] for d in display["diagnoses"]]
    assert confidences == sorted(confidences, reverse=True)


def test_full_case_summary_and_warning_are_jargon_free():
    patient = _patient()
    result = predict(patient)
    display = _to_display(result, patient)
    _assert_jargon_free(display["summary"])
    for w in display["warnings"]:
        _assert_jargon_free(w)


def test_full_case_summary_states_top_result_and_confidence():
    patient = _patient()
    result = predict(patient)
    display = _to_display(result, patient)
    top = display["diagnoses"][0]
    assert top["condition"] in display["summary"]
    assert f"{top['confidence'] * 100:.0f}%" in display["summary"]


def test_fallback_case_is_jargon_free_and_uses_clinical_field_names():
    # Missing the full thyroid panel forces the fallback path.
    patient = _patient(missing=["tsh", "t3", "t4", "t4_uptake", "free_thyroxine_index"])
    result = predict(patient)
    display = _to_display(result, patient)
    assert display["joint_available"] is False
    _assert_jargon_free(display["summary"])
    for w in display["warnings"]:
        _assert_jargon_free(w)
    # Missing labs should use clinical names, not raw field keys.
    assert "free_thyroxine_index" not in display["summary"]
    assert "t4_uptake" not in display["summary"]


def test_insufficient_data_case_has_no_result_but_still_jargon_free():
    patient = {"age": 28, "sex": "female"}
    result = predict(patient)
    display = _to_display(result, patient)
    assert display["diagnoses"][0]["condition"] == "Unknown - Insufficient Data"
    _assert_jargon_free(display["summary"])
