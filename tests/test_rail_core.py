import math

import pytest

from experiments.rail_core import (
    AdmissionParams,
    admission_diagnostics,
    admission_score,
    contamination_contract,
)


def test_admission_score_matches_reference_formula():
    params = AdmissionParams(
        tau_min=0.8,
        tau_max=6.0,
        k=1.2,
        theta=0.60,
        w_delta=1.0,
        w_features=0.02,
        w_edits=0.12,
        w_focus=0.03,
    )

    delta_sec = 2.4
    num_features = 5
    edit_count = 1
    focus_seconds = 1.5

    beta = 0.02 * num_features + 0.12 * edit_count + 0.03 * focus_seconds
    z_fast = 1.2 * (delta_sec - (0.8 + beta))
    z_slow = 1.2 * ((6.0 + beta) - delta_sec)
    expected = (1.0 / (1.0 + math.exp(-z_fast))) * (1.0 / (1.0 + math.exp(-z_slow)))

    assert admission_score(delta_sec, num_features, edit_count, focus_seconds, params) == expected


def test_out_of_window_decisions_score_lower_than_deliberate_decision():
    params = AdmissionParams(tau_min=1.5, tau_max=4.8, k=2.8, theta=0.38)

    hasty = admission_score(0.2, num_features=20, edit_count=0, focus_seconds=0.1, params=params)
    deliberate = admission_score(
        2.8, num_features=20, edit_count=1, focus_seconds=2.0, params=params
    )
    confused = admission_score(9.0, num_features=20, edit_count=1, focus_seconds=4.0, params=params)

    assert hasty < deliberate
    assert confused < deliberate


def test_diagnostics_expose_eligibility_and_intermediates():
    params = AdmissionParams(tau_min=0.8, tau_max=6.0, k=1.2, theta=0.60)

    diagnostics = admission_diagnostics(
        delta_sec=2.4,
        num_features=5,
        edit_count=1,
        focus_seconds=1.5,
        params=params,
    )

    assert diagnostics["eligible"] is True
    assert set(diagnostics) == {
        "score",
        "delta_sec",
        "beta",
        "z_fast",
        "z_slow",
        "s_fast",
        "s_slow",
        "eligible",
    }


def test_contamination_contract_matches_bayes_bound():
    feedback_is_correct = [False, False, False, False, True, True, True, True, True, True]
    admitted = [True, False, False, False, True, True, True, True, True, False]

    contract = contamination_contract(feedback_is_correct, admitted, contamination_budget=0.20)

    assert contract["total_feedback"] == 10
    assert contract["contaminated_feedback"] == 4
    assert contract["admitted_feedback"] == 6
    assert contract["contaminated_admissions"] == 1
    assert contract["false_admission_rate"] == 0.25
    assert contract["clean_admission_rate"] == pytest.approx(5 / 6)
    assert contract["admitted_contamination_rate"] == pytest.approx(1 / 6)
    assert contract["contamination_bound"] == pytest.approx(1 / 6)
    assert contract["contamination_prevented"] == 3
    assert contract["feedback_sacrificed"] == 4
    assert contract["admission_efficiency"] == pytest.approx(0.75)
    assert contract["satisfies_budget"] is True


def test_contamination_contract_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        contamination_contract([True], [True, False])

    with pytest.raises(ValueError):
        contamination_contract([True], [True], contamination_budget=1.1)
