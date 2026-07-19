"""
Checks the shipped code against the original plan: three models
(male-skewed, women-only, clinical-criteria) feeding a meta-learner
that outputs a 4-way joint class for PCOS x thyroid dysfunction.

Deviations from the plan are pinned down explicitly below.
"""
import inspect

from pipeline_common import CLASS_NAMES
from model_adapter import CONDITION_LABELS
import model3
import meta_learner
import predict


# Plan's 3 models map to code as: male-skewed -> Model 1,
# women-only -> Model 2 (thyroid) + Model 2b (PCOS), clinical-criteria -> Model 3.

def test_model3_is_rule_based_not_trained():
    """Model 3 should be rule-based, not a trained model."""
    source = inspect.getsource(model3)
    assert "joblib.load" not in source
    assert "fit(" not in source
    assert "import xgboost" not in source
    assert "sklearn.linear_model" not in source


def test_joint_output_is_the_planned_4way_split():
    """A=PCOS, B=thyroid dysfunction."""
    assert set(CLASS_NAMES.values()) == {"neither", "thyroid_only", "pcos_only", "both"}
    assert len(CLASS_NAMES) == 4


def test_display_labels_cover_every_joint_class():
    """Every joint class needs a display label, or predictions can't render."""
    assert set(CONDITION_LABELS.keys()) == set(CLASS_NAMES.values())


def test_meta_learner_feature_contract_matches_between_training_and_serving():
    """Training and serving must build the meta-feature row with the same names, same order."""
    import re

    expected_names = {
        "model2_thyroid_proba", "model2_pcos_proba",
        "pcos_criteria_count", "thyroid_criteria_call",
    }

    def _ordered_meta_feature_names(source: str) -> list:
        # Extract known meta-feature keys in first-seen order.
        seen = []
        for name in re.findall(r'"(\w+)":', source):
            if name in expected_names and name not in seen:
                seen.append(name)
        return seen

    serving_cols = _ordered_meta_feature_names(inspect.getsource(predict._run_joint))
    training_cols = _ordered_meta_feature_names(inspect.getsource(meta_learner.build_meta_features))

    assert set(serving_cols) == expected_names, f"serving-time meta-features changed: {serving_cols}"
    assert set(training_cols) == expected_names, f"training-time meta-features changed: {training_cols}"
    assert serving_cols == training_cols, (
        f"meta-feature column ORDER mismatch: serving={serving_cols} "
        f"vs training={training_cols}"
    )


def test_deviation_model1_excluded_from_meta_learner():
    """Model 1's output never feeds the meta-learner, since its training
    cohort lacks Model 1's required labs for any patient. This is an
    intentional deviation from the plan -- update this test, don't delete it,
    if that changes."""
    meta_features = {
        "model2_thyroid_proba", "model2_pcos_proba",
        "pcos_criteria_count", "thyroid_criteria_call",
    }
    training_source = inspect.getsource(meta_learner.build_meta_features)
    for feat in meta_features:
        assert feat in training_source
    assert "model1" not in training_source.lower()
