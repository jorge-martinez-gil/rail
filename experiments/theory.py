"""Formal contamination-control results for the RAIL ingress layer.

The paper makes two empirical claims that we tighten here with analytical
results:

1. **Bayes bound (Proposition 1).** Under the operator's per-event admission
   model, the post-admission contamination probability is

   .. math::

       P(C = 1 \\mid A = 1)
           \\;=\\; \\frac{\\pi\\alpha}{\\pi\\alpha + (1-\\pi)\\rho},

   where :math:`\\pi=P(C=1)` is the base contamination rate,
   :math:`\\alpha = P(A=1\\mid C=1)` is the false-admission rate, and
   :math:`\\rho = P(A=1\\mid C=0)` is the clean-admission rate. The empirical
   plug-in is :func:`experiments.rail_core.contamination_contract`.

2. **Expected contamination on a horizon of N events (Proposition 2).** The
   expected number of contaminated admissions over an N-event horizon is
   :math:`\\mathbb{E}[K_N] = N\\,\\pi\\,\\alpha`, and a Hoeffding tail bound
   yields, for any :math:`\\varepsilon > 0`,

   .. math::

       P\\!\\left( K_N \\ge N\\,(\\pi\\alpha + \\varepsilon) \\right)
           \\;\\le\\; \\exp(-2 N \\varepsilon^2).

   This lets a deployment translate an operational risk budget into a
   horizon: see :func:`required_horizon_for_budget`.

3. **Yield-vs-contamination Pareto frontier (Lemma 1).** For a parametric
   admission family indexed by the threshold ``theta``, sweeping ``theta``
   traces a monotone non-increasing trade-off between admitted-feedback
   yield and false-admission rate; the lower-left envelope of the resulting
   :math:`(Y, C)` points is the operational Pareto frontier. We compute it
   in :func:`pareto_frontier`.

All routines are dependency-light (only the standard library and
:mod:`experiments.rail_core`) so they can be invoked from the test suite
without pulling NumPy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

try:
    from .rail_core import contamination_contract
except ImportError:  # pragma: no cover
    from rail_core import contamination_contract


# ---------------------------------------------------------------------------
# Proposition 1: closed-form Bayes bound.
# ---------------------------------------------------------------------------


def bayes_post_admission_contamination(
    base_contamination_rate: float,
    false_admission_rate: float,
    clean_admission_rate: float,
) -> float:
    """Analytical post-admission contamination probability.

    Implements ``P(C=1 | A=1) = pi*alpha / (pi*alpha + (1-pi)*rho)``.

    Returns ``0.0`` when no event would be admitted (denominator vanishes);
    that corresponds to the degenerate case ``alpha = rho = 0``.
    """

    for value, name in (
        (base_contamination_rate, "base_contamination_rate"),
        (false_admission_rate, "false_admission_rate"),
        (clean_admission_rate, "clean_admission_rate"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    numerator = base_contamination_rate * false_admission_rate
    denominator = numerator + (1.0 - base_contamination_rate) * clean_admission_rate
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def admission_efficiency(
    pi: float, alpha_always: float, alpha_policy: float, rho_always: float, rho_policy: float
) -> float:
    """Analytical Admission Efficiency relative to the ``Always`` policy.

    Mirrors the AE metric defined in Section 4.4 of the paper but at the
    population level. ``alpha_always`` is the false-admission rate of the
    ``Always`` baseline (typically 1.0) and ``rho_always`` is its
    clean-admission rate.

    AE is undefined when the policy sacrifices no yield relative to
    ``Always`` (denominator zero). Returns ``inf`` in that case so callers
    can detect it.
    """

    delta_contam = (alpha_always - alpha_policy) * pi
    delta_yield = (alpha_always - alpha_policy) * pi + (rho_always - rho_policy) * (1.0 - pi)
    if delta_yield <= 0.0:
        return float("inf") if delta_contam > 0.0 else 0.0
    return delta_contam / delta_yield


# ---------------------------------------------------------------------------
# Proposition 2: horizon-aware concentration bound.
# ---------------------------------------------------------------------------


def expected_contaminated_admissions(
    horizon: int, base_contamination_rate: float, false_admission_rate: float
) -> float:
    """Expected number of contaminated admissions over ``horizon`` events."""

    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if not 0.0 <= base_contamination_rate <= 1.0:
        raise ValueError("base_contamination_rate must lie in [0, 1]")
    if not 0.0 <= false_admission_rate <= 1.0:
        raise ValueError("false_admission_rate must lie in [0, 1]")
    return horizon * base_contamination_rate * false_admission_rate


def hoeffding_contamination_bound(
    horizon: int, base_contamination_rate: float, false_admission_rate: float, epsilon: float
) -> float:
    """Hoeffding upper bound on P(K_N >= N*(pi*alpha + epsilon)).

    Returns ``exp(-2 N epsilon^2)`` clipped to ``[0, 1]``.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 <= base_contamination_rate <= 1.0:
        raise ValueError("base_contamination_rate must lie in [0, 1]")
    if not 0.0 <= false_admission_rate <= 1.0:
        raise ValueError("false_admission_rate must lie in [0, 1]")
    if horizon <= 0:
        return 1.0
    return min(1.0, math.exp(-2.0 * horizon * epsilon * epsilon))


def bennett_contamination_bound(
    horizon: int,
    base_contamination_rate: float,
    false_admission_rate: float,
    epsilon: float,
) -> float:
    """Bennett's inequality upper bound on P(K_N >= N*(pi*alpha + epsilon)).

    Bennett's inequality (Bennett, 1962) is tighter than Hoeffding whenever
    the per-event variance is small relative to the range. For the
    contamination count K_N, each indicator is Bernoulli(pi*alpha), so the
    variance per trial is ``sigma^2 = pi*alpha*(1-pi*alpha)``. The bound is

    .. math::

        P(K_N \\ge N(\\mu + \\varepsilon)) \\;\\le\\;
            \\exp\\!\\left(-\\frac{N\\,\\sigma^2}{b^2}\\,h\\!\\left(
                \\frac{b\\,\\varepsilon}{\\sigma^2}\\right)\\right),

    where :math:`b = 1 - \\mu` is the maximum excess above the mean,
    :math:`h(u) = (1+u)\\ln(1+u) - u` is the Bennett function, and
    :math:`\\mu = \\pi\\alpha`.

    For small :math:`\\varepsilon` this is strictly tighter than the Hoeffding
    bound ``exp(-2 N epsilon^2)``.

    Returns the bound clipped to ``[0, 1]``.

    References
    ----------
    * Bennett, "Probability Inequalities for the Sum of Independent Random
      Variables," J. Am. Stat. Assoc., 1962.
    * Maurer & Pontil, "Empirical Bernstein Bounds and Sample-Variance
      Penalization," COLT 2009 (for the empirical variant).
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 <= base_contamination_rate <= 1.0:
        raise ValueError("base_contamination_rate must lie in [0, 1]")
    if not 0.0 <= false_admission_rate <= 1.0:
        raise ValueError("false_admission_rate must lie in [0, 1]")
    if horizon <= 0:
        return 1.0

    mu = base_contamination_rate * false_admission_rate
    # Variance of a Bernoulli(mu) indicator.
    sigma2 = mu * (1.0 - mu)
    if sigma2 < 1e-15:
        # Near-zero variance: event is near-deterministic; Hoeffding is exact.
        return min(1.0, math.exp(-2.0 * horizon * epsilon * epsilon))

    # Maximum excess above the mean for a [0,1] bounded variable is b = 1 - mu.
    b = 1.0 - mu
    u = b * epsilon / sigma2
    # Bennett function h(u) = (1+u)*ln(1+u) - u; numerically stable for all u > 0.
    h = (1.0 + u) * math.log(1.0 + u) - u
    exponent = -(horizon * sigma2 / (b * b)) * h
    return min(1.0, math.exp(exponent))


def variance_aware_horizon(
    contamination_budget: float,
    confidence: float,
    base_contamination_rate: float,
    false_admission_rate: float,
) -> int:
    """Smallest horizon N certified by Bennett's bound (tighter than Hoeffding).

    Analogous to :func:`required_horizon_for_budget` but uses
    :func:`bennett_contamination_bound` instead of the Hoeffding bound. Because
    Bennett exploits the variance of the admission indicator, the certified
    horizon is typically smaller (i.e., the same confidence is achieved sooner).

    Returns ``-1`` if the population mean already exhausts the per-event budget.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if contamination_budget <= 0.0:
        raise ValueError("contamination_budget must be positive")

    pi_alpha = base_contamination_rate * false_admission_rate
    n = 1
    while True:
        per_event = contamination_budget / n
        if per_event <= pi_alpha:
            return -1
        eps = per_event - pi_alpha
        bound = bennett_contamination_bound(n, base_contamination_rate, false_admission_rate, eps)
        if bound <= 1.0 - confidence:
            return n
        # Jump forward: Hoeffding closed form gives a safe lower bound on N.
        log_term = -math.log(1.0 - confidence)
        n_heuristic = math.ceil(log_term / (2.0 * eps * eps))
        if n_heuristic > n:
            n = n_heuristic
        else:
            n += 1
        if n > 10**9:  # pragma: no cover
            return -1


def required_horizon_for_budget(
    contamination_budget: float,
    confidence: float,
    base_contamination_rate: float,
    false_admission_rate: float,
) -> int:
    """Smallest horizon N such that the Hoeffding bound certifies the budget.

    Returns the smallest :math:`N` satisfying

    .. math::

        N\\,(\\pi\\alpha + \\varepsilon) \\le B \\quad \\text{and} \\quad
        \\exp(-2 N \\varepsilon^2) \\le 1 - p,

    where :math:`B` is the absolute contamination budget per ``horizon``,
    :math:`p = \\text{confidence}` and :math:`\\varepsilon` is chosen as the
    slack between ``budget/N`` and the population mean. We use the
    closed-form

    .. math::

        N = \\max\\!\\left\\{ 1,\\; \\left\\lceil
            \\frac{-\\ln(1 - p)}{2\\,\\varepsilon^2}\\right\\rceil \\right\\}

    after solving :math:`\\varepsilon = B/N - \\pi\\alpha` jointly via
    bisection over feasible ``N``.

    Returns ``-1`` if no horizon satisfies the budget (i.e. the population
    mean already exceeds the per-event budget).
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if contamination_budget <= 0.0:
        raise ValueError("contamination_budget must be positive")
    pi_alpha = base_contamination_rate * false_admission_rate
    log_term = -math.log(1.0 - confidence)

    # Find smallest N >= 1 s.t. eps(N) = B/N - pi_alpha > 0 AND
    #                      exp(-2 N eps^2) <= 1 - p.
    # Note: eps(N) is decreasing in N (B/N decreases), so feasibility shrinks
    # as N grows. We try increasing N until either the budget per event is
    # exhausted or the Hoeffding inequality is satisfied.
    n = 1
    while True:
        per_event = contamination_budget / n
        if per_event <= pi_alpha:
            return -1
        eps = per_event - pi_alpha
        bound = math.exp(-2.0 * n * eps * eps)
        if bound <= 1.0 - confidence:
            return n
        # required N from Hoeffding alone (closed form)
        n_needed = math.ceil(log_term / (2.0 * eps * eps))
        if n_needed > n:
            n = n_needed
        else:
            n += 1
        if n > 10**9:  # pragma: no cover - sanity bound
            return -1


# ---------------------------------------------------------------------------
# Lemma 1: Pareto frontier of yield vs contamination.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParetoPoint:
    """A point on the (yield, contamination) curve indexed by ``theta``."""

    theta: float
    yield_rate: float
    contamination_rate: float
    admission_efficiency: float


def pareto_frontier(
    scores: Sequence[float],
    feedback_is_correct: Sequence[bool],
    thetas: Sequence[float] | None = None,
) -> list[ParetoPoint]:
    """Compute the yield-vs-contamination frontier for a score family.

    Parameters
    ----------
    scores : sequence of float
        Admission scores per event (e.g. ``V`` for RAIL or ``max_prob`` for
        a confidence baseline).
    feedback_is_correct : sequence of bool
        Whether each event's label is correct (``True``) or contaminated
        (``False``).
    thetas : sequence of float, optional
        Threshold grid to sweep. Defaults to a 51-point grid on ``[0, 1]``.

    Returns
    -------
    list[ParetoPoint]
        Points sorted by ``theta``. Yield rate is the fraction of admitted
        events; contamination rate is the fraction of admitted events that
        are contaminated.
    """

    if len(scores) != len(feedback_is_correct):
        raise ValueError("scores and feedback_is_correct must have the same length")
    if thetas is None:
        thetas = [i / 50.0 for i in range(51)]

    n = len(scores)
    out: list[ParetoPoint] = []
    for theta in thetas:
        admitted = [s >= theta for s in scores]
        contract = contamination_contract(feedback_is_correct, admitted)
        y = contract["admitted_feedback"] / n if n else 0.0
        c = contract["admitted_contamination_rate"]
        out.append(
            ParetoPoint(
                theta=float(theta),
                yield_rate=float(y),
                contamination_rate=float(c),
                admission_efficiency=float(contract["admission_efficiency"]),
            )
        )
    out.sort(key=lambda p: p.theta)
    return out


def pareto_lower_envelope(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Lower-left envelope of a Pareto cloud (minimise contamination at each yield).

    Greedy O(n log n) algorithm: sort by descending yield, keep points that
    strictly improve on the running-minimum contamination.
    """

    sorted_pts = sorted(points, key=lambda p: (-p.yield_rate, p.contamination_rate))
    envelope: list[ParetoPoint] = []
    best = math.inf
    for p in sorted_pts:
        if p.contamination_rate < best - 1e-12:
            envelope.append(p)
            best = p.contamination_rate
    envelope.sort(key=lambda p: p.yield_rate)
    return envelope


# ---------------------------------------------------------------------------
# Monte Carlo verification of Propositions 1 and 2.
# ---------------------------------------------------------------------------


def monte_carlo_contamination(
    pi: float,
    alpha: float,
    rho: float,
    horizon: int = 5_000,
    runs: int = 200,
    seed: int = 0,
    sampler: Callable[[int, int], list[bool]] | None = None,
) -> dict[str, float]:
    """Simulate ``runs`` independent horizons and return empirical contamination.

    Used by the test suite to verify that the closed-form Bayes bound and the
    Hoeffding tail bound are consistent with simulation. ``sampler`` is an
    optional injection seam: it receives ``(seed_offset, horizon)`` and
    returns a list of ``horizon`` booleans for ``C``; defaults to Bernoulli.
    """

    import random

    rng = random.Random(seed)
    contam_emp = []
    yield_emp = []
    post_admit_contam = []
    for r in range(runs):
        run_rng = random.Random(rng.random())
        admitted_count = 0
        contam_admitted = 0
        if sampler is not None:
            c_samples = sampler(r, horizon)
        else:
            c_samples = [run_rng.random() < pi for _ in range(horizon)]
        for c in c_samples:
            if c:
                a = run_rng.random() < alpha
            else:
                a = run_rng.random() < rho
            if a:
                admitted_count += 1
                if c:
                    contam_admitted += 1
        contam_emp.append(contam_admitted)
        yield_emp.append(admitted_count)
        post_admit_contam.append(contam_admitted / admitted_count if admitted_count > 0 else 0.0)

    return {
        "mean_post_admission_contamination": sum(post_admit_contam) / runs,
        "mean_contaminated_admissions": sum(contam_emp) / runs,
        "mean_admitted_yield": sum(yield_emp) / runs,
        "analytical_post_admission_contamination": bayes_post_admission_contamination(
            pi, alpha, rho
        ),
        "analytical_expected_contamination": expected_contaminated_admissions(horizon, pi, alpha),
    }


__all__ = [
    "ParetoPoint",
    "admission_efficiency",
    "bayes_post_admission_contamination",
    "bennett_contamination_bound",
    "expected_contaminated_admissions",
    "hoeffding_contamination_bound",
    "monte_carlo_contamination",
    "pareto_frontier",
    "pareto_lower_envelope",
    "required_horizon_for_budget",
    "variance_aware_horizon",
]
