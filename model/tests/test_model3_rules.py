"""
Unit tests for model3.py's rule-based scorers, using synthetic rows.
Boundary values matter since thresholds are strict/non-strict on purpose.
"""
import pandas as pd
import pytest

from model3 import (
    thyroid_criteria_score,
    pcos_criteria_score,
    TSH_LOW_HYPERTHYROID,
    TSH_HIGH_HYPOTHYROID,
    LH_FSH_RATIO_THRESHOLD,
    FOLLICLE_COUNT_THRESHOLD,
    AMH_ELEVATED_THRESHOLD,
)


def _row(**overrides):
    base = dict(
        cycle_regularity=2,   # 2 = regular (not the oligo/anovulation flag)
        hirsutism=0,
        lh=5.0,
        fsh=5.0,               # ratio 1.0, below threshold
        follicle_count_left=5,
        follicle_count_right=5,
        amh=2.0,
    )
    base.update(overrides)
    return pd.DataFrame([base])


# --- thyroid axis: TSH threshold, inclusive on the boundary itself ---

@pytest.mark.parametrize("tsh,expected", [
    (TSH_LOW_HYPERTHYROID, 0),        # exactly at the low boundary -> normal
    (TSH_LOW_HYPERTHYROID - 0.01, 1),  # just below -> hyperthyroid-range
    (TSH_HIGH_HYPOTHYROID, 0),         # exactly at the high boundary -> normal
    (TSH_HIGH_HYPOTHYROID + 0.01, 1),  # just above -> hypothyroid-range
    (2.5, 0),                          # mid-range normal
])
def test_thyroid_criteria_score_boundaries(tsh, expected):
    result = thyroid_criteria_score(pd.Series([tsh]))
    assert result.iloc[0] == expected


# --- PCOS axis: each of the 3 Rotterdam criteria, and their combination ---

def test_no_criteria_met_gives_zero_and_no_rotterdam_call():
    scores = pcos_criteria_score(_row())
    assert scores["pcos_criteria_count"].iloc[0] == 0
    assert scores["pcos_rotterdam_call"].iloc[0] == 0


def test_oligo_anovulation_flag_is_cycle_regularity_4():
    scores = pcos_criteria_score(_row(cycle_regularity=4))
    assert scores["criterion_oligo_anovulation"].iloc[0] == 1
    scores_regular = pcos_criteria_score(_row(cycle_regularity=2))
    assert scores_regular["criterion_oligo_anovulation"].iloc[0] == 0


def test_hyperandrogenism_via_clinical_hirsutism():
    scores = pcos_criteria_score(_row(hirsutism=1))
    assert scores["criterion_hyperandrogenism"].iloc[0] == 1


def test_hyperandrogenism_via_lh_fsh_ratio_boundary():
    # exactly 2.0 should not count (strict '>' in the code)
    at_threshold = _row(lh=10.0, fsh=5.0)  # ratio == 2.0
    scores = pcos_criteria_score(at_threshold)
    assert scores["criterion_hyperandrogenism"].iloc[0] == 0

    above_threshold = _row(lh=10.01, fsh=5.0)  # ratio just above 2.0
    scores = pcos_criteria_score(above_threshold)
    assert scores["criterion_hyperandrogenism"].iloc[0] == 1


def test_hyperandrogenism_handles_zero_fsh_without_crashing():
    # fsh=0 would divide-by-zero; the code replaces 0 with NA first.
    scores = pcos_criteria_score(_row(fsh=0, lh=5.0, hirsutism=1))
    assert scores["criterion_hyperandrogenism"].iloc[0] == 1
    # NaN ratio should not count as elevated on its own.
    scores_no_hirsutism = pcos_criteria_score(_row(fsh=0, lh=5.0, hirsutism=0))
    assert scores_no_hirsutism["criterion_hyperandrogenism"].iloc[0] == 0


def test_polycystic_morphology_via_follicle_count_either_ovary():
    left = pcos_criteria_score(_row(follicle_count_left=FOLLICLE_COUNT_THRESHOLD))
    assert left["criterion_polycystic_morphology"].iloc[0] == 1
    right = pcos_criteria_score(_row(follicle_count_right=FOLLICLE_COUNT_THRESHOLD))
    assert right["criterion_polycystic_morphology"].iloc[0] == 1
    below = pcos_criteria_score(_row(
        follicle_count_left=FOLLICLE_COUNT_THRESHOLD - 1,
        follicle_count_right=FOLLICLE_COUNT_THRESHOLD - 1,
    ))
    assert below["criterion_polycystic_morphology"].iloc[0] == 0


def test_polycystic_morphology_via_amh_as_ultrasound_alternative():
    # Elevated AMH alone can satisfy this criterion (2023 guideline).
    scores = pcos_criteria_score(_row(
        follicle_count_left=5, follicle_count_right=5, amh=AMH_ELEVATED_THRESHOLD,
    ))
    assert scores["criterion_polycystic_morphology"].iloc[0] == 1


def test_rotterdam_call_requires_at_least_two_of_three():
    # Exactly one criterion met -> count=1, no call
    one_met = pcos_criteria_score(_row(cycle_regularity=4))
    assert one_met["pcos_criteria_count"].iloc[0] == 1
    assert one_met["pcos_rotterdam_call"].iloc[0] == 0

    # Two criteria met -> count=2, call=1
    two_met = pcos_criteria_score(_row(cycle_regularity=4, hirsutism=1))
    assert two_met["pcos_criteria_count"].iloc[0] == 2
    assert two_met["pcos_rotterdam_call"].iloc[0] == 1

    # All three -> count=3, call=1
    three_met = pcos_criteria_score(_row(
        cycle_regularity=4, hirsutism=1, follicle_count_left=FOLLICLE_COUNT_THRESHOLD,
    ))
    assert three_met["pcos_criteria_count"].iloc[0] == 3
    assert three_met["pcos_rotterdam_call"].iloc[0] == 1
