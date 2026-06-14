"""Tests for the formal contamination-theory module."""

import math

import pytest

from experiments.theory import (
    admission_efficiency,
    bayes_post_admission_contamination,
    expected_contaminated_admissions,
    hoeffding_contamination_bound,
    monte_carlo_contamination,
    pareto_frontier,
    pareto_lower_envelope,
    required_horizon_for_budget,
)


def test_bayes_bound_matches_paper_example():
    val = bayes_post_admission_contamination(0.3, 0.2, 0.9)
    assert val == pytest.approx(0.06 / (0.06 + 0.63), rel=1e-6)


def test_bayes_bound_edge_cases():
    assert bayes_post_admission_contamination(0.0, 0.0, 0.0) == 0.0
    assert bayes_post_admission_contamination(1.0, 0.0, 1.0) == 0.0
    with pytest.raises(ValueError):
        bayes_post_admission_contamination(-0.1, 0.5, 0.5)


def test_admission_efficiency_perfect_tradeoff_is_one():
    val = admission_efficiency(
        pi=0.3, alpha_always=1.0, alpha_policy=0.5, rho_always=1.0, rho_policy=1.0
    )
    assert val == pytest.approx(1.0)


def test_admission_efficiency_no_change_returns_zero():
    val = admission_efficiency(
        pi=0.3, alpha_always=1.0, alpha_policy=1.0, rho_always=1.0, rho_policy=1.0
    )
    assert val == 0.0


def test_admission_efficiency_better_than_perfect_when_policy_relaxes_clean():
    val = admission_efficiency(
        pi=0.3, alpha_always=1.0, alpha_policy=0.5, rho_always=0.5, rho_policy=1.0
    )
    assert math.isinf(val)


def test_hoeffding_bound_in_unit_interval():
    val = hoeffding_contamination_bound(1000, 0.3, 0.2, epsilon=0.05)
    assert 0.0 <= val <= 1.0


def test_required_horizon_finds_smallest_feasible_n():
    n = required_horizon_for_budget(
        contamination_budget=60.0,
        confidence=0.95,
        base_contamination_rate=0.3,
        false_admission_rate=0.2,
    )
    assert n > 0
    eps = 60.0 / n - 0.06
    assert eps > 0
    assert hoeffding_contamination_bound(n, 0.3, 0.2, eps) <= 0.05


def test_required_horizon_returns_minus_one_when_infeasible():
    n = required_horizon_for_budget(
        contamination_budget=0.05,
        confidence=0.95,
        base_contamination_rate=0.3,
        false_admission_rate=0.2,
    )
    assert n == -1


def test_monte_carlo_agrees_with_bayes_within_tolerance():
    out = monte_carlo_contamination(pi=0.3, alpha=0.2, rho=0.9, horizon=4_000, runs=80, seed=11)
    analytical = out["analytical_post_admission_contamination"]
    empirical = out["mean_post_admission_contamination"]
    assert abs(analytical - empirical) < 0.02
    expected = out["analytical_expected_contamination"]
    assert abs(out["mean_contaminated_admissions"] - expected) / expected < 0.1


def test_pareto_frontier_monotone_in_theta():
    scores = [0.05 * i for i in range(20)] + [0.4 + 0.03 * i for i in range(20)]
    correct = [False] * 20 + [True] * 20
    pts = pareto_frontier(scores, correct)
    yields = [p.yield_rate for p in pts]
    assert all(yields[i] >= yields[i + 1] - 1e-9 for i in range(len(yields) - 1))


def test_pareto_envelope_is_minimal_subset():
    scores = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    correct = [False, False, True, True, True, True]
    frontier = pareto_frontier(scores, correct)
    envelope = pareto_lower_envelope(frontier)
    assert len(envelope) <= len(frontier)
    assert all(
        envelope[i].yield_rate <= envelope[i + 1].yield_rate for i in range(len(envelope) - 1)
    )


def test_expected_contaminated_admissions_validation():
    with pytest.raises(ValueError):
        expected_contaminated_admissions(-1, 0.1, 0.1)
    with pytest.raises(ValueError):
        expected_contaminated_admissions(10, 1.5, 0.1)
