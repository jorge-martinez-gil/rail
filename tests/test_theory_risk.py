"""Tests for the excess-risk bound module."""

from __future__ import annotations

import math

import pytest

from experiments.theory_risk import (
    crossover_loss_gap,
    empirical_excess_risk,
    excess_risk_bound,
    optimal_admission_threshold,
    regime_dominance_condition,
)

# ---------------------------------------------------------------------------
# Bound structure.
# ---------------------------------------------------------------------------


def test_bound_decomposes_correctly():
    b = excess_risk_bound(
        horizon=100,
        contamination_rate=0.05,
        diameter=2.0,
        gradient_bound=3.0,
        loss_gap=0.5,
    )
    assert b.statistical == pytest.approx(2.0 * 3.0 / math.sqrt(100))
    assert b.contamination == pytest.approx(0.05 * 0.5)
    assert b.total == pytest.approx(b.statistical + b.contamination)


def test_bound_decreases_with_horizon():
    a = excess_risk_bound(horizon=10, contamination_rate=0.0).total
    b = excess_risk_bound(horizon=1000, contamination_rate=0.0).total
    assert b < a


def test_bound_increases_with_contamination():
    a = excess_risk_bound(horizon=100, contamination_rate=0.0).total
    b = excess_risk_bound(horizon=100, contamination_rate=0.5).total
    assert b > a


def test_bound_validates_inputs():
    with pytest.raises(ValueError):
        excess_risk_bound(horizon=0, contamination_rate=0.1)
    with pytest.raises(ValueError):
        excess_risk_bound(horizon=10, contamination_rate=1.5)


# ---------------------------------------------------------------------------
# Empirical instantiation -- ties contract to bound.
# ---------------------------------------------------------------------------


def test_empirical_excess_risk_recovers_contract_value():
    """When alpha = 0 (no contaminated admissions) the bound's contamination
    term must vanish regardless of horizon."""
    b = empirical_excess_risk(
        base_contamination_rate=0.30,
        false_admission_rate=0.0,
        clean_admission_rate=0.80,
        horizon=200,
    )
    assert b.contamination == pytest.approx(0.0)


def test_empirical_excess_risk_monotone_in_alpha():
    """Higher false-admission rate -> larger bound."""
    b_low = empirical_excess_risk(
        base_contamination_rate=0.30,
        false_admission_rate=0.05,
        clean_admission_rate=0.80,
        horizon=200,
    )
    b_high = empirical_excess_risk(
        base_contamination_rate=0.30,
        false_admission_rate=0.50,
        clean_admission_rate=0.80,
        horizon=200,
    )
    assert b_high.total > b_low.total


# ---------------------------------------------------------------------------
# Optimization.
# ---------------------------------------------------------------------------


def test_optimal_threshold_prefers_clean_when_loss_gap_is_large():
    """When the contamination penalty dominates, picking the *strictest*
    threshold (lowest contamination) should minimize the bound -- even at
    the cost of a much smaller horizon."""

    def yield_fn(theta: float) -> float:
        return max(0.05, 1.0 - theta)

    def cont_fn(theta: float) -> float:
        return max(0.0, 0.5 * (1.0 - theta))

    theta_star, _b = optimal_admission_threshold(
        theta_grid=[0.1, 0.3, 0.5, 0.7, 0.9],
        yield_at_theta=yield_fn,
        contamination_at_theta=cont_fn,
        horizon_total=5_000,
        diameter=1.0,
        gradient_bound=1.0,
        loss_gap=10.0,  # huge penalty -> prefer stricter theta
    )
    assert theta_star == 0.9


def test_optimal_threshold_prefers_permissive_when_loss_gap_is_tiny():
    def yield_fn(theta: float) -> float:
        return max(0.05, 1.0 - theta)

    def cont_fn(theta: float) -> float:
        return max(0.0, 0.5 * (1.0 - theta))

    theta_star, _b = optimal_admission_threshold(
        theta_grid=[0.1, 0.3, 0.5, 0.7, 0.9],
        yield_at_theta=yield_fn,
        contamination_at_theta=cont_fn,
        horizon_total=1_000_000,
        diameter=1.0,
        gradient_bound=1.0,
        loss_gap=0.001,  # tiny penalty -> prefer permissive theta for big T
    )
    assert theta_star == 0.1


# ---------------------------------------------------------------------------
# Dominance condition.
# ---------------------------------------------------------------------------


def test_dominance_check_returns_consistent_margin():
    chk = regime_dominance_condition(
        rail_contamination=0.02,
        competitor_contamination=0.20,
        rail_yield=0.6,
        competitor_yield=0.95,
        horizon_total=2000,
        loss_gap=1.0,
    )
    assert chk.margin == pytest.approx(chk.competitor_bound - chk.rail_bound)
    assert chk.holds is (chk.margin > 0.0)


def test_dominance_check_rail_wins_for_high_loss_gap():
    chk_low = regime_dominance_condition(
        rail_contamination=0.02,
        competitor_contamination=0.20,
        rail_yield=0.6,
        competitor_yield=0.95,
        horizon_total=2000,
        loss_gap=0.01,
    )
    chk_high = regime_dominance_condition(
        rail_contamination=0.02,
        competitor_contamination=0.20,
        rail_yield=0.6,
        competitor_yield=0.95,
        horizon_total=2000,
        loss_gap=1.0,
    )
    # As Delta grows the contamination term swamps the statistical term and
    # RAIL's certificate eventually dominates.
    assert chk_high.margin > chk_low.margin


# ---------------------------------------------------------------------------
# Crossover loss gap.
# ---------------------------------------------------------------------------


def test_crossover_loss_gap_is_inf_when_rail_more_contaminated():
    """If RAIL's certified contamination is *higher* than the competitor's,
    no finite crossover exists -- the function returns +inf."""
    val = crossover_loss_gap(
        rail_contamination=0.30,
        competitor_contamination=0.10,
        rail_yield=0.6,
        competitor_yield=0.9,
        horizon_total=1000,
    )
    assert math.isinf(val) and val > 0


def test_crossover_loss_gap_consistent_with_dominance_check():
    """At Delta == crossover, both bounds should be equal (within rounding)."""
    args = dict(
        rail_contamination=0.05,
        competitor_contamination=0.20,
        rail_yield=0.6,
        competitor_yield=0.9,
        horizon_total=2000,
        diameter=1.0,
        gradient_bound=1.0,
    )
    delta_star = crossover_loss_gap(**args)
    assert math.isfinite(delta_star)
    chk = regime_dominance_condition(**args, loss_gap=delta_star)
    assert abs(chk.margin) < 1e-6
