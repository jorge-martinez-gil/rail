"""Shared RAIL admission scoring primitives.

The browser console and experiment scripts implement the same vigilance score.
This module is the canonical Python reference and is small enough to audit
end-to-end. Equations follow Section 3 of the paper.

Public surface
--------------

* :class:`AdmissionParams` -- frozen, validated parameter bundle.
* :func:`admission_score` / :func:`admission_diagnostics` -- per-event scoring.
* :func:`contamination_contract` -- post-hoc empirical contamination bound
  (Bayes plug-in, see :mod:`experiments.theory` for the analytical companion).
* :class:`OnlineBurnInCalibrator` -- streaming P^2-quantile estimator that
  recovers ``(tau_min, tau_max)`` from observed anchored-deliberation values,
  matching the paper's "short burn-in of 300 reviewed alerts" procedure.
* :func:`burn_in_window_from_samples` -- offline counterpart used in tests
  and to validate the online estimator.

All per-event operations are :math:`\\mathcal{O}(1)` in time and memory, with
no external dependencies, so this file can be imported in lightweight
settings (edge gateways, browser bridges, embedded tests).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypedDict


@dataclass(frozen=True)
class AdmissionParams:
    """Parameters for the RAIL vigilance/admission score."""

    tau_min: float = 0.8
    tau_max: float = 6.0
    k: float = 1.2
    theta: float = 0.60
    w_delta: float = 1.0
    w_features: float = 0.02
    w_edits: float = 0.12
    w_focus: float = 0.03

    def __post_init__(self) -> None:
        if not math.isfinite(self.tau_min) or not math.isfinite(self.tau_max):
            raise ValueError("tau_min and tau_max must be finite")
        if self.tau_min < 0.0 or self.tau_max < 0.0:
            raise ValueError("tau_min and tau_max must be non-negative")
        if self.tau_min >= self.tau_max:
            raise ValueError("tau_min must be strictly less than tau_max")
        if self.k <= 0.0:
            raise ValueError("k must be positive")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")
        if self.w_delta <= 0.0:
            raise ValueError("w_delta must be positive")
        for name in ("w_features", "w_edits", "w_focus"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


class AdmissionDiagnostics(TypedDict):
    score: float
    delta_sec: float
    beta: float
    z_fast: float
    z_slow: float
    s_fast: float
    s_slow: float
    eligible: bool


class ContaminationContract(TypedDict):
    total_feedback: int
    admitted_feedback: int
    contaminated_feedback: int
    clean_feedback: int
    contaminated_admissions: int
    clean_admissions: int
    base_contamination_rate: float
    false_admission_rate: float
    clean_admission_rate: float
    admitted_contamination_rate: float
    contamination_bound: float
    contamination_prevented: int
    feedback_sacrificed: int
    admission_efficiency: float
    satisfies_budget: bool


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def complexity_bonus(
    num_features: int,
    edit_count: int,
    focus_seconds: float,
    params: AdmissionParams,
) -> float:
    return (
        params.w_features * float(num_features)
        + params.w_edits * float(edit_count)
        + params.w_focus * float(focus_seconds)
    )


def admission_diagnostics(
    delta_sec: float,
    num_features: int,
    edit_count: int,
    focus_seconds: float,
    params: AdmissionParams = AdmissionParams(),
) -> AdmissionDiagnostics:
    beta = complexity_bonus(num_features, edit_count, focus_seconds, params)
    z_fast = params.k * ((params.w_delta * delta_sec) - (params.tau_min + beta))
    z_slow = params.k * ((params.tau_max + beta) - (params.w_delta * delta_sec))
    s_fast = sigmoid(z_fast)
    s_slow = sigmoid(z_slow)
    score = s_fast * s_slow
    return {
        "score": float(score),
        "delta_sec": float(delta_sec),
        "beta": float(beta),
        "z_fast": float(z_fast),
        "z_slow": float(z_slow),
        "s_fast": float(s_fast),
        "s_slow": float(s_slow),
        "eligible": bool(score >= params.theta),
    }


def admission_score(
    delta_sec: float,
    num_features: int,
    edit_count: int,
    focus_seconds: float,
    params: AdmissionParams = AdmissionParams(),
) -> float:
    return admission_diagnostics(
        delta_sec=delta_sec,
        num_features=num_features,
        edit_count=edit_count,
        focus_seconds=focus_seconds,
        params=params,
    )["score"]


def contamination_contract(
    feedback_is_correct: Sequence[bool],
    admitted: Sequence[bool],
    contamination_budget: float = 0.05,
) -> ContaminationContract:
    if len(feedback_is_correct) != len(admitted):
        raise ValueError("feedback_is_correct and admitted must have the same length")
    if not 0.0 <= contamination_budget <= 1.0:
        raise ValueError("contamination_budget must lie in [0, 1]")
    total_feedback = len(feedback_is_correct)
    contaminated_feedback = sum(1 for ok in feedback_is_correct if not ok)
    clean_feedback = total_feedback - contaminated_feedback
    admitted_feedback = sum(1 for ok in admitted if ok)
    contaminated_admissions = sum(
        1 for ok, gate in zip(feedback_is_correct, admitted) if gate and not ok
    )
    clean_admissions = admitted_feedback - contaminated_admissions
    base_contamination_rate = contaminated_feedback / total_feedback if total_feedback else 0.0
    false_admission_rate = (
        contaminated_admissions / contaminated_feedback if contaminated_feedback else 0.0
    )
    clean_admission_rate = clean_admissions / clean_feedback if clean_feedback else 0.0
    admitted_contamination_rate = (
        contaminated_admissions / admitted_feedback if admitted_feedback else 0.0
    )
    numerator = base_contamination_rate * false_admission_rate
    denominator = numerator + (1.0 - base_contamination_rate) * clean_admission_rate
    contamination_bound = numerator / denominator if denominator > 0.0 else 0.0
    contamination_prevented = contaminated_feedback - contaminated_admissions
    feedback_sacrificed = total_feedback - admitted_feedback
    admission_efficiency = (
        contamination_prevented / feedback_sacrificed if feedback_sacrificed else 0.0
    )
    return {
        "total_feedback": int(total_feedback),
        "admitted_feedback": int(admitted_feedback),
        "contaminated_feedback": int(contaminated_feedback),
        "clean_feedback": int(clean_feedback),
        "contaminated_admissions": int(contaminated_admissions),
        "clean_admissions": int(clean_admissions),
        "base_contamination_rate": float(base_contamination_rate),
        "false_admission_rate": float(false_admission_rate),
        "clean_admission_rate": float(clean_admission_rate),
        "admitted_contamination_rate": float(admitted_contamination_rate),
        "contamination_bound": float(contamination_bound),
        "contamination_prevented": int(contamination_prevented),
        "feedback_sacrificed": int(feedback_sacrificed),
        "admission_efficiency": float(admission_efficiency),
        "satisfies_budget": bool(contamination_bound <= contamination_budget),
    }


class _P2Quantile:
    """Single P^2 quantile estimator (Jain & Chlamtac, CACM 1985)."""

    __slots__ = ("p", "_n", "_q", "_npos", "_dpos_inc")

    def __init__(self, p: float) -> None:
        if not 0.0 < p < 1.0:
            raise ValueError("quantile probability must lie in (0, 1)")
        self.p = float(p)
        self._n = 0
        self._q: list[float] = []
        self._npos: list[float] = []
        self._dpos_inc = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]

    def update(self, x: float) -> None:
        x = float(x)
        if self._n < 5:
            self._q.append(x)
            self._n += 1
            if self._n == 5:
                self._q.sort()
                self._npos = [1.0, 2.0, 3.0, 4.0, 5.0]
            return
        if x < self._q[0]:
            self._q[0] = x
            k = 0
        elif x >= self._q[4]:
            self._q[4] = x
            k = 3
        else:
            k = 0
            for i in range(4):
                if self._q[i] <= x < self._q[i + 1]:
                    k = i
                    break
        for i in range(k + 1, 5):
            self._npos[i] += 1.0
        self._n += 1
        desired = [1.0 + (self._n - 1) * inc for inc in self._dpos_inc]
        for i in (1, 2, 3):
            d = desired[i] - self._npos[i]
            if (d >= 1.0 and self._npos[i + 1] - self._npos[i] > 1.0) or (
                d <= -1.0 and self._npos[i - 1] - self._npos[i] < -1.0
            ):
                d_sign = 1.0 if d >= 0 else -1.0
                qi_p = self._q[i] + (d_sign / (self._npos[i + 1] - self._npos[i - 1])) * (
                    (self._npos[i] - self._npos[i - 1] + d_sign)
                    * (self._q[i + 1] - self._q[i])
                    / (self._npos[i + 1] - self._npos[i])
                    + (self._npos[i + 1] - self._npos[i] - d_sign)
                    * (self._q[i] - self._q[i - 1])
                    / (self._npos[i] - self._npos[i - 1])
                )
                if self._q[i - 1] < qi_p < self._q[i + 1]:
                    self._q[i] = qi_p
                else:
                    j = int(d_sign)
                    self._q[i] = self._q[i] + d_sign * (self._q[i + j] - self._q[i]) / (
                        self._npos[i + j] - self._npos[i]
                    )
                self._npos[i] += d_sign

    def value(self) -> float:
        if self._n == 0:
            raise ValueError("no samples observed yet")
        if self._n < 5:
            ordered = sorted(self._q)
            idx = self.p * (len(ordered) - 1)
            lo = int(math.floor(idx))
            hi = int(math.ceil(idx))
            if lo == hi:
                return float(ordered[lo])
            frac = idx - lo
            return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)
        return float(self._q[2])

    @property
    def count(self) -> int:
        return self._n


@dataclass
class OnlineBurnInCalibrator:
    """Streaming calibrator that recovers (tau_min, tau_max) online."""

    min_samples: int = 300
    low: float = 0.10
    high: float = 0.90
    min_floor: float = 0.3
    max_ceiling: float = 30.0
    _low_estimator: _P2Quantile = field(init=False, repr=False)
    _high_estimator: _P2Quantile = field(init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if not 0.0 < self.low < self.high < 1.0:
            raise ValueError("must satisfy 0 < low < high < 1")
        if self.min_floor < 0.0 or self.max_ceiling <= self.min_floor:
            raise ValueError("invalid (min_floor, max_ceiling) bounds")
        self._low_estimator = _P2Quantile(self.low)
        self._high_estimator = _P2Quantile(self.high)

    def update(self, delta_sec: float) -> None:
        if not math.isfinite(delta_sec) or delta_sec < 0.0:
            return
        self._low_estimator.update(delta_sec)
        self._high_estimator.update(delta_sec)
        self._count += 1

    def extend(self, deltas: Iterable[float]) -> None:
        for d in deltas:
            self.update(d)

    @property
    def count(self) -> int:
        return self._count

    def is_ready(self) -> bool:
        return self._count >= self.min_samples

    def window(self) -> tuple[float, float]:
        if self._count == 0:
            return self.min_floor, self.max_ceiling
        lo = max(self.min_floor, self._low_estimator.value())
        hi = min(self.max_ceiling, self._high_estimator.value())
        if hi <= lo:
            hi = lo + 1e-3
        return float(lo), float(hi)

    def admission_params(self, base: AdmissionParams | None = None) -> AdmissionParams:
        base = base or AdmissionParams()
        lo, hi = self.window()
        return AdmissionParams(
            tau_min=lo,
            tau_max=hi,
            k=base.k,
            theta=base.theta,
            w_delta=base.w_delta,
            w_features=base.w_features,
            w_edits=base.w_edits,
            w_focus=base.w_focus,
        )


def burn_in_window_from_samples(
    deltas: Sequence[float],
    low: float = 0.10,
    high: float = 0.90,
    min_floor: float = 0.3,
    max_ceiling: float = 30.0,
) -> tuple[float, float]:
    if not deltas:
        return min_floor, max_ceiling
    sorted_d = sorted(float(d) for d in deltas if math.isfinite(d) and d >= 0.0)
    if not sorted_d:
        return min_floor, max_ceiling

    def _q(p: float) -> float:
        idx = p * (len(sorted_d) - 1)
        lo_i = int(math.floor(idx))
        hi_i = int(math.ceil(idx))
        if lo_i == hi_i:
            return sorted_d[lo_i]
        frac = idx - lo_i
        return sorted_d[lo_i] * (1.0 - frac) + sorted_d[hi_i] * frac

    lo = max(min_floor, _q(low))
    hi = min(max_ceiling, _q(high))
    if hi <= lo:
        hi = lo + 1e-3
    return float(lo), float(hi)
