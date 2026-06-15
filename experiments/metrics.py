"""Operational admission-quality metrics for the RAIL evaluation.

The paper reports four headline metrics: Macro-F1, admission yield,
contamination reduction, and Admission Efficiency. To strengthen the
operational story expected by a Information Systems reviewer we add four further metrics
that target the *temporal* dimension of contamination:

* :func:`time_to_first_contamination` -- index of the first contaminated
  admission. Sensitive to early-stream behaviour where downstream learners
  are most vulnerable.
* :func:`contamination_half_life` -- the smallest horizon at which
  half of the contamination budget is consumed under the current policy.
* :func:`cumulative_contamination_curve` -- monotone curve of contaminated
  admissions vs. number of admissions, useful for paper figures.
* :func:`admission_yield_loss` -- fraction of clean feedback that the policy
  unnecessarily rejected (Type II), the complementary metric to
  contamination reduction.

We also provide :func:`burn_in_estimator_from_paper` that recovers
``(tau_min, tau_max)`` from a vector of Delta values using the *exact*
quantile rule reported in Section 3.1 of the paper (mean +/- 1.5 sigma
clamped to the [0.10, 0.90] percentiles).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, pstdev


@dataclass(frozen=True)
class CumulativePoint:
    admission_index: int
    cumulative_contaminated: int
    cumulative_admitted: int


def time_to_first_contamination(
    feedback_is_correct: Sequence[bool], admitted: Sequence[bool]
) -> int:
    """Index of the first contaminated admission, or ``-1`` if none.

    Indexing follows the input order so callers can interpret it as a
    horizon length when the stream is presented chronologically.
    """

    if len(feedback_is_correct) != len(admitted):
        raise ValueError("inputs must have the same length")
    for i, (ok, a) in enumerate(zip(feedback_is_correct, admitted, strict=False)):
        if a and not ok:
            return i
    return -1


def contamination_half_life(feedback_is_correct: Sequence[bool], admitted: Sequence[bool]) -> int:
    """Horizon at which half of the run's contaminated admissions are reached.

    Returns ``-1`` if no contaminated admission ever occurs. The value is the
    smallest index ``i`` such that the prefix ``feedback[:i+1]`` contains
    >= half of the total contaminated admissions.
    """

    if len(feedback_is_correct) != len(admitted):
        raise ValueError("inputs must have the same length")
    total = sum(1 for ok, a in zip(feedback_is_correct, admitted, strict=False) if a and not ok)
    if total == 0:
        return -1
    target = math.ceil(total / 2)
    running = 0
    for i, (ok, a) in enumerate(zip(feedback_is_correct, admitted, strict=False)):
        if a and not ok:
            running += 1
            if running >= target:
                return i
    return -1  # unreachable given the total check, but explicit


def cumulative_contamination_curve(
    feedback_is_correct: Sequence[bool], admitted: Sequence[bool]
) -> list[CumulativePoint]:
    """Cumulative contaminated-vs-admitted curve, one point per admission."""

    if len(feedback_is_correct) != len(admitted):
        raise ValueError("inputs must have the same length")
    out: list[CumulativePoint] = []
    contaminated = 0
    admitted_running = 0
    for _i, (ok, a) in enumerate(zip(feedback_is_correct, admitted, strict=False)):
        if a:
            admitted_running += 1
            if not ok:
                contaminated += 1
            out.append(
                CumulativePoint(
                    admission_index=admitted_running,
                    cumulative_contaminated=contaminated,
                    cumulative_admitted=admitted_running,
                )
            )
    return out


def admission_yield_loss(feedback_is_correct: Sequence[bool], admitted: Sequence[bool]) -> float:
    """Fraction of clean feedback unnecessarily rejected (Type II error)."""

    if len(feedback_is_correct) != len(admitted):
        raise ValueError("inputs must have the same length")
    clean = [ok for ok in feedback_is_correct if ok]
    if not clean:
        return 0.0
    rejected_clean = sum(
        1 for ok, a in zip(feedback_is_correct, admitted, strict=False) if ok and not a
    )
    return rejected_clean / len(clean)


def area_under_contamination_curve(
    feedback_is_correct: Sequence[bool], admitted: Sequence[bool]
) -> float:
    """Area under the cumulative-contamination curve (AUCC).

    The cumulative contamination curve (CCC) plots the running fraction of
    contaminated admissions against the running number of total admissions.
    A policy that never admits contaminated events produces a flat zero curve
    (AUCC = 0.0). The ``Always`` baseline yields AUCC = base_contamination_rate.
    RAIL's AUCC should be strictly below the ``Always`` baseline.

    We compute the trapezoid integral over the discrete CCC points, normalised
    by the total number of admissions so the result lies in ``[0, 1]`` and is
    comparable across streams of different lengths.

    Returns ``0.0`` if no events are admitted.
    """

    if len(feedback_is_correct) != len(admitted):
        raise ValueError("inputs must have the same length")
    points = cumulative_contamination_curve(feedback_is_correct, admitted)
    if not points:
        return 0.0
    total_admitted = points[-1].cumulative_admitted
    if total_admitted == 0:
        return 0.0

    # Trapezoid integration over (x=admission_index/total, y=contamination_frac).
    area = 0.0
    prev_x = 0.0
    prev_y = 0.0
    for pt in points:
        x = pt.cumulative_admitted / total_admitted
        y = pt.cumulative_contaminated / pt.cumulative_admitted
        area += 0.5 * (x - prev_x) * (prev_y + y)
        prev_x = x
        prev_y = y
    return float(area)


def normalised_contamination_gain(
    feedback_is_correct: Sequence[bool], admitted: Sequence[bool]
) -> float:
    """Normalised Contamination Gain (NCG).

    Measures the fractional reduction in post-admission contamination rate
    achieved by a policy relative to the ``Always-admit`` baseline:

    .. math::

        \\text{NCG} = 1 - \\frac{c_{\\text{policy}}}{c_{\\text{base}}}

    where :math:`c_{\\text{policy}}` is the admitted contamination rate and
    :math:`c_{\\text{base}}` is the overall base contamination rate.

    * NCG = 1.0 means perfect filtering (no contamination admitted).
    * NCG = 0.0 means the policy gives no benefit over always-admitting.
    * NCG < 0.0 means the policy *worsens* contamination (admits more
      contaminated events proportionally than the base rate).

    Returns ``0.0`` if the base contamination rate is zero (no filtering possible).
    """

    if len(feedback_is_correct) != len(admitted):
        raise ValueError("inputs must have the same length")
    n = len(feedback_is_correct)
    if n == 0:
        return 0.0
    contaminated = sum(1 for ok in feedback_is_correct if not ok)
    base_rate = contaminated / n if n > 0 else 0.0
    if base_rate == 0.0:
        return 0.0
    admitted_total = sum(1 for a in admitted if a)
    if admitted_total == 0:
        return 1.0
    contaminated_admitted = sum(
        1 for ok, a in zip(feedback_is_correct, admitted, strict=False) if a and not ok
    )
    policy_rate = contaminated_admitted / admitted_total
    return float(1.0 - policy_rate / base_rate)


def burn_in_estimator_from_paper(
    deltas: Sequence[float],
    sigma_multiplier: float = 1.5,
    clip_low: float = 0.10,
    clip_high: float = 0.90,
    min_floor: float = 0.3,
    max_ceiling: float = 30.0,
) -> tuple[float, float]:
    """Reproduce the paper's burn-in recipe verbatim.

    The paper estimates the admissible window from a 300-alert burn-in by
    centring on the empirical mean of :math:`\\Delta`, widening by
    :math:`\\pm 1.5\\sigma`, and clipping to the 10th/90th percentiles.
    Implementing this as a single function makes it usable both in tests
    and as a calibration helper in production deployments.
    """

    cleaned = [float(d) for d in deltas if math.isfinite(d) and d >= 0.0]
    if len(cleaned) < 5:
        return min_floor, max_ceiling
    mu = mean(cleaned)
    sd = pstdev(cleaned)
    lo_raw = mu - sigma_multiplier * sd
    hi_raw = mu + sigma_multiplier * sd
    sorted_d = sorted(cleaned)

    def _q(p: float) -> float:
        idx = p * (len(sorted_d) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return sorted_d[lo]
        frac = idx - lo
        return sorted_d[lo] * (1.0 - frac) + sorted_d[hi] * frac

    q_lo = _q(clip_low)
    q_hi = _q(clip_high)
    lo = max(min_floor, min(lo_raw, q_lo))
    hi = min(max_ceiling, max(hi_raw, q_hi))
    if hi <= lo:
        hi = lo + 1e-3
    return float(lo), float(hi)


__all__ = [
    "CumulativePoint",
    "admission_yield_loss",
    "area_under_contamination_curve",
    "burn_in_estimator_from_paper",
    "contamination_half_life",
    "cumulative_contamination_curve",
    "normalised_contamination_gain",
    "time_to_first_contamination",
]
