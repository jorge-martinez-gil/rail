"""Excess-risk bound for online learning under a certified contamination rate.

The contamination contract in :mod:`experiments.theory` certifies a bound
``c_hat = pi * bar_alpha / (pi * bar_alpha + (1 - pi) * underline_rho)`` on
the probability that any admitted feedback is corrupted. This module turns
that contract into a *learning* guarantee: a bound on the excess risk of
the online recalibration learner, as a function of the certified
contamination rate.

Theorem 3 (informal). Let the online learner be projected stochastic
sub-gradient descent on a Lipschitz, convex surrogate loss with diameter
``D`` and gradient bound ``G``, using step-size ``eta_t = D / (G * sqrt(t))``
on a clean stream. Let ``c <= c_hat`` be the certified post-admission
contamination rate and let ``Delta`` be the bounded loss-difference between
clean and adversarial labels (``|l(theta, (x, y_clean)) - l(theta, (x,
y_corrupt))| <= Delta`` uniformly in theta). Then, for T admitted events,
the expected excess risk of the iterate average is

    E[R(theta_bar) - R(theta*)] <= D * G / sqrt(T)   +   c * Delta.

The first term is the classical online-to-batch SGD rate (Zinkevich 2003;
Cesa-Bianchi & Lugosi 2006); the second is the *contamination penalty*
induced by the residual fraction of corrupted labels passing the gate.

Implications for the paper.
    * RAIL-gated decouples the two terms: by lowering c at the cost of a
      smaller T, it shifts mass from the contamination penalty to the
      statistical term. The optimal operating point is at
      d/d(theta) [D G / sqrt(T(theta)) + c(theta) Delta] = 0, which we
      compute numerically in :func:`optimal_admission_threshold`.
    * Loss-/confidence-gated baselines do not certify c: they admit any
      sample with low loss/high confidence, including confidently-wrong
      labels (the worst kind under this bound). Their effective c is
      bounded below by the irreducible model-confidence error on the
      contaminated subpopulation.

The bound is derived in the journal appendix (Lemma A.1 + Theorem A.2).
This module provides:

* :func:`excess_risk_bound` -- the upper bound itself.
* :func:`optimal_admission_threshold` -- the operating point that
  minimizes the bound for a given (pi, calibration curves, T_total).
* :func:`empirical_excess_risk` -- the value the bound certifies, given
  the harness's measured (clean_admission_rate, contamination_rate)
  curves.
* :func:`regime_dominance_condition` -- the inequality on (c, Delta, D,
  G, T) that determines when V-gating provably dominates a loss/score
  gate at the same operating yield.

References
----------
* Zinkevich, "Online Convex Programming and Generalized Infinitesimal
  Gradient Ascent," ICML 2003.
* Cesa-Bianchi & Lugosi, "Prediction, Learning, and Games," 2006.
* Natarajan, Dhillon, Ravikumar, Tewari, "Learning with Noisy Labels,"
  NeurIPS 2013 (loss-correction interpretation of the contamination
  penalty).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

try:
    from .theory import bayes_post_admission_contamination
except ImportError:  # pragma: no cover
    from theory import bayes_post_admission_contamination  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Core bound.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExcessRiskBound:
    statistical: float  # D G / sqrt(T)
    contamination: float  # c * Delta
    horizon: int
    contamination_rate: float
    diameter: float
    gradient_bound: float
    loss_gap: float

    @property
    def total(self) -> float:
        return self.statistical + self.contamination

    def as_dict(self) -> dict:
        return {
            "statistical": self.statistical,
            "contamination": self.contamination,
            "total": self.total,
            "horizon": self.horizon,
            "contamination_rate": self.contamination_rate,
            "diameter": self.diameter,
            "gradient_bound": self.gradient_bound,
            "loss_gap": self.loss_gap,
        }


def excess_risk_bound(
    horizon: int,
    contamination_rate: float,
    diameter: float = 1.0,
    gradient_bound: float = 1.0,
    loss_gap: float = 1.0,
) -> ExcessRiskBound:
    """Compute the theorem's upper bound on excess risk."""

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0.0 <= contamination_rate <= 1.0:
        raise ValueError("contamination_rate must lie in [0, 1]")
    if diameter <= 0.0 or gradient_bound <= 0.0 or loss_gap < 0.0:
        raise ValueError("diameter and gradient_bound must be positive; loss_gap >= 0")

    statistical = diameter * gradient_bound / math.sqrt(horizon)
    contamination = contamination_rate * loss_gap
    return ExcessRiskBound(
        statistical=float(statistical),
        contamination=float(contamination),
        horizon=int(horizon),
        contamination_rate=float(contamination_rate),
        diameter=float(diameter),
        gradient_bound=float(gradient_bound),
        loss_gap=float(loss_gap),
    )


# ---------------------------------------------------------------------------
# Operating-point optimization.
# ---------------------------------------------------------------------------


def optimal_admission_threshold(
    theta_grid: Sequence[float],
    yield_at_theta: Callable[[float], float],
    contamination_at_theta: Callable[[float], float],
    horizon_total: int,
    diameter: float = 1.0,
    gradient_bound: float = 1.0,
    loss_gap: float = 1.0,
) -> tuple[float, ExcessRiskBound]:
    """Pick the theta in ``theta_grid`` that minimizes the excess-risk bound."""

    if horizon_total < 1:
        raise ValueError("horizon_total must be >= 1")
    if not theta_grid:
        raise ValueError("theta_grid must be non-empty")

    best: tuple[float, ExcessRiskBound] | None = None
    for theta in sorted(theta_grid):
        y = float(yield_at_theta(theta))
        c = float(contamination_at_theta(theta))
        T = max(1, round(y * horizon_total))
        b = excess_risk_bound(
            horizon=T,
            contamination_rate=c,
            diameter=diameter,
            gradient_bound=gradient_bound,
            loss_gap=loss_gap,
        )
        if best is None or b.total < best[1].total - 1e-12:
            best = (float(theta), b)
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Empirical instantiation.
# ---------------------------------------------------------------------------


def empirical_excess_risk(
    base_contamination_rate: float,
    false_admission_rate: float,
    clean_admission_rate: float,
    horizon: int,
    diameter: float = 1.0,
    gradient_bound: float = 1.0,
    loss_gap: float = 1.0,
) -> ExcessRiskBound:
    """Use the contamination contract to obtain a concrete bound."""

    c = bayes_post_admission_contamination(
        base_contamination_rate=base_contamination_rate,
        false_admission_rate=false_admission_rate,
        clean_admission_rate=clean_admission_rate,
    )
    return excess_risk_bound(
        horizon=horizon,
        contamination_rate=c,
        diameter=diameter,
        gradient_bound=gradient_bound,
        loss_gap=loss_gap,
    )


# ---------------------------------------------------------------------------
# Regime-dominance condition.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DominanceCheck:
    holds: bool
    rail_bound: float
    competitor_bound: float
    margin: float  # positive = RAIL strictly better
    rail_yield: float
    competitor_yield: float


def regime_dominance_condition(
    rail_contamination: float,
    competitor_contamination: float,
    rail_yield: float,
    competitor_yield: float,
    horizon_total: int,
    diameter: float = 1.0,
    gradient_bound: float = 1.0,
    loss_gap: float = 1.0,
) -> DominanceCheck:
    """Compare the excess-risk bound for RAIL against another gate."""

    rail_T = max(1, round(rail_yield * horizon_total))
    comp_T = max(1, round(competitor_yield * horizon_total))
    rail = excess_risk_bound(
        horizon=rail_T,
        contamination_rate=rail_contamination,
        diameter=diameter,
        gradient_bound=gradient_bound,
        loss_gap=loss_gap,
    )
    comp = excess_risk_bound(
        horizon=comp_T,
        contamination_rate=competitor_contamination,
        diameter=diameter,
        gradient_bound=gradient_bound,
        loss_gap=loss_gap,
    )
    margin = comp.total - rail.total
    return DominanceCheck(
        holds=margin > 0.0,
        rail_bound=rail.total,
        competitor_bound=comp.total,
        margin=margin,
        rail_yield=rail_yield,
        competitor_yield=competitor_yield,
    )


# ---------------------------------------------------------------------------
# Closed-form crossover loss_gap.
# ---------------------------------------------------------------------------


def crossover_loss_gap(
    rail_contamination: float,
    competitor_contamination: float,
    rail_yield: float,
    competitor_yield: float,
    horizon_total: int,
    diameter: float = 1.0,
    gradient_bound: float = 1.0,
) -> float:
    """Smallest loss_gap above which RAIL's excess-risk bound is smaller.

    Setting RAIL.total = competitor.total and solving for ``Delta`` using the
    same integer-rounded horizons that the bound itself uses (so the closed
    form matches the discretised comparison exactly):

        DG/sqrt(T_R) + c_R * Delta = DG/sqrt(T_C) + c_C * Delta
        Delta * (c_C - c_R) = DG * (1/sqrt(T_R) - 1/sqrt(T_C))
        Delta_* = DG * (1/sqrt(T_R) - 1/sqrt(T_C)) / (c_C - c_R)

    Returns 0 when the closed form is non-positive (RAIL dominates even at
    Delta=0) and ``+inf`` when ``c_R >= c_C`` (no Delta makes RAIL win via
    the contamination term).
    """

    if rail_contamination >= competitor_contamination:
        return float("inf")
    rail_T = max(1, round(rail_yield * horizon_total))
    comp_T = max(1, round(competitor_yield * horizon_total))
    num = diameter * gradient_bound * (1.0 / math.sqrt(rail_T) - 1.0 / math.sqrt(comp_T))
    den = competitor_contamination - rail_contamination  # > 0 here
    val = num / den
    return max(0.0, float(val))


__all__ = [
    "DominanceCheck",
    "ExcessRiskBound",
    "crossover_loss_gap",
    "empirical_excess_risk",
    "excess_risk_bound",
    "optimal_admission_threshold",
    "regime_dominance_condition",
]
