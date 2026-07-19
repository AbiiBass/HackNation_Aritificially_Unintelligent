"""
Integration tests for predict.py's ensemble pipeline, using the trained
artifacts in model/models/. Covers routing, abstention, and the
meta-learner combination step.
"""
import pandas as pd
import pytest

from predict import predict, FIELDS_MODEL1, FIELDS_MODEL2, FIELDS_MODEL2B
from train_model1 import FEATURE_COLS as MODEL1_TRAIN_COLS
from train_model2 import FEATURE_COLS as MODEL2_TRAIN_COLS
from train_model2b_pcos import FEATURE_COLS as MODEL2B_TRAIN_COLS
from pipeline_common import CLASS_NAMES

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


def _patient(missing=()):
    p = dict(FULL_PATIENT)
    for k in missing:
        p[k] = None
    return p


# --- predict.py's field lists should match each model's training columns. ---

def test_fields_model1_matches_training_columns():
    assert set(FIELDS_MODEL1) == set(MODEL1_TRAIN_COLS)


def test_fields_model2_matches_model2_training_columns_exactly():
    # model2 is trained without tsh (would leak the label), unlike model2b.
    assert set(FIELDS_MODEL2) == set(MODEL2_TRAIN_COLS)


def test_fields_model2b_matches_model2b_training_columns_exactly():
    assert set(FIELDS_MODEL2B) == set(MODEL2B_TRAIN_COLS)


def test_model2_thyroid_runs_without_tsh_since_it_never_uses_it():
    """Model 2 doesn't use tsh as a feature, so it should still run without it."""
    patient = _patient(missing=["tsh"])
    result = predict(patient)
    proba = result["model2_thyroid"]["thyroid_dysfunction_proba"]
    assert proba is not None
    assert 0.0 <= proba <= 1.0
    assert result["model2b_pcos"]["pcos_proba"] is None
    assert "tsh" in (result["model2b_pcos"]["reason"] or "")


# --- Individual model abstention ---

def test_model1_abstains_with_reason_when_thyroid_panel_missing():
    patient = _patient(missing=["tsh", "t3", "t4", "t4_uptake", "free_thyroxine_index"])
    result = predict(patient)
    assert result["model1"]["thyroid_dysfunction_proba"] is None
    assert "missing fields" in result["model1"]["reason"]


def test_model1_runs_when_full_panel_present():
    result = predict(_patient())
    proba = result["model1"]["thyroid_dysfunction_proba"]
    assert proba is not None
    assert 0.0 <= proba <= 1.0


def test_model3_axes_are_independent_of_each_other():
    # PCOS axis should score fine without TSH.
    result = predict(_patient(missing=["tsh"]))
    assert result["model3"]["pcos_criteria_count"] is not None
    assert result["model3"]["thyroid_criteria_call"] is None
    # Thyroid axis should score fine without PCOS fields.
    result2 = predict(_patient(missing=["cycle_regularity", "hirsutism"]))
    assert result2["model3"]["thyroid_criteria_call"] is not None
    assert result2["model3"]["pcos_criteria_count"] is None


# --- Joint / meta-learner ---

def test_joint_runs_and_produces_a_valid_4class_distribution():
    result = predict(_patient())
    joint = result["joint"]
    probs = joint["probabilities"]
    assert probs is not None
    assert set(probs.keys()) == set(CLASS_NAMES.values())
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert pytest.approx(sum(probs.values()), abs=1e-6) == 1.0


def test_joint_abstains_when_meta_features_incomplete():
    # Missing a model3 PCOS field alone should block the joint call.
    patient = _patient(missing=["follicle_count_left", "follicle_count_right"])
    result = predict(patient)
    assert result["joint"]["probabilities"] is None
    assert "joint ensemble needs" in result["joint"]["reason"]


def test_model1_substitution_heuristic_flag():
    # Both Model 1 and the joint ensemble can run, so substitution should fire.
    result = predict(_patient())
    assert result["joint"]["used_model1_for_thyroid_input"] is True
    assert result["joint"]["warning"]

    # Model 1 can't run without the full thyroid panel, so nothing to substitute.
    result2 = predict(_patient(missing=["t3", "t4", "t4_uptake", "free_thyroxine_index"]))
    assert result2["joint"]["used_model1_for_thyroid_input"] is False
