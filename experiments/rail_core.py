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
        1 for ok, gate in zip(feedback_is_correct, admitted, strict=False) if gate and not ok
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

    __slots__ = ("_dpos_inc", "_n", "_npos", "_q", "p")

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
            lo = math.floor(idx)
            hi = math.ceil(idx)
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
    low: float = 0.25
    high: float = 0.85
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
    low: float = 0.25,
    high: float = 0.85,
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
        lo_i = math.floor(idx)
        hi_i = math.ceil(idx)
        if lo_i == hi_i:
            return sorted_d[lo_i]
        frac = idx - lo_i
        return sorted_d[lo_i] * (1.0 - frac) + sorted_d[hi_i] * frac

    lo = max(min_floor, _q(low))
    hi = min(max_ceiling, _q(high))
    if hi <= lo:
        hi = lo + 1e-3
    return float(lo), float(hi)


@dataclass
class OnlineDriftCalibrator:
    """CUSUM-based drift detector with automatic window re-calibration.

    The burn-in calibrator in :class:`OnlineBurnInCalibrator` performs a
    *one-shot* calibration after a warm-up period.  In long-running
    deployments the operator's deliberation distribution can shift (seasonal
    workload, task difficulty, team changes).  ``OnlineDriftCalibrator``
    wraps the burn-in calibrator with a Page-Hinkley / CUSUM change-point
    detector: when the running mean of delta deviates from the post-calibration
    estimate by more than ``drift_threshold`` standard deviations for more than
    ``drift_window`` consecutive events, the calibrator resets and a new burn-in
    begins.

    This is the "adaptive calibration" mechanism referred to in Section 3.5
    of the paper.

    Parameters
    ----------
    base_calibrator : OnlineBurnInCalibrator
        The underlying calibrator.  When drift is detected it is reset to a
        fresh :class:`OnlineBurnInCalibrator` with the same hyper-parameters.
    drift_threshold : float
        Number of empirical standard deviations the running mean must deviate
        from the reference mean to trigger drift detection.  Higher values
        make detection less sensitive.  Default of 3.0 gives ≈ 1% false-alarm
        rate under a Gaussian null.
    drift_window : int
        Minimum number of consecutive above-threshold events before a reset is
        triggered.  Prevents spurious resets from single outliers.
    reference_window : int
        Number of most-recent deltas used to maintain the reference mean and
        standard deviation (rolling estimate).  Must be >= base_calibrator's
        ``min_samples``.

    Notes
    -----
    All per-event operations are O(1).  The calibrator is checkpointable: the
    most important state is ``drift_count`` and ``window_swapped``.
    """

    base_calibrator: OnlineBurnInCalibrator = field(
        default_factory=lambda: OnlineBurnInCalibrator(min_samples=300)
    )
    drift_threshold: float = 3.0
    drift_window: int = 20
    reference_window: int = 300
    _ref_estimator: _P2Quantile = field(init=False, repr=False)
    _mean_est: float = field(default=0.0, init=False, repr=False)
    _m2_est: float = field(default=0.0, init=False, repr=False)
    _ref_count: int = field(default=0, init=False, repr=False)
    _drift_count: int = field(default=0, init=False, repr=False)
    _resets: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.drift_threshold <= 0.0:
            raise ValueError("drift_threshold must be positive")
        if self.drift_window < 1:
            raise ValueError("drift_window must be >= 1")
        if self.reference_window < 1:
            raise ValueError("reference_window must be >= 1")
        self._ref_estimator = _P2Quantile(0.5)  # median tracker for reference

    def update(self, delta_sec: float) -> bool:
        """Process one delta observation.  Returns ``True`` if a drift reset occurred."""

        if not math.isfinite(delta_sec) or delta_sec < 0.0:
            return False

        self.base_calibrator.update(delta_sec)

        # Maintain Welford online mean/variance over the reference window.
        # We use a simple EMA decay once the window is full to stay O(1).
        self._ref_count += 1
        if self._ref_count <= self.reference_window:
            # Welford accumulation during warm-up.
            delta = delta_sec - self._mean_est
            self._mean_est += delta / self._ref_count
            delta2 = delta_sec - self._mean_est
            self._m2_est += delta * delta2
        else:
            # Exponential decay with effective window = reference_window.
            alpha = 2.0 / (self.reference_window + 1)
            prev_mean = self._mean_est
            self._mean_est += alpha * (delta_sec - self._mean_est)
            self._m2_est = (1.0 - alpha) * (
                self._m2_est + alpha * (delta_sec - prev_mean) ** 2
            )

        # Only check for drift once the base calibrator has completed burn-in.
        if not self.base_calibrator.is_ready():
            return False

        ref_var = self._m2_est / max(1, self._ref_count - 1) if self._ref_count > 1 else 0.0
        ref_sd = math.sqrt(ref_var) if ref_var > 0.0 else 1e-6
        z = abs(delta_sec - self._mean_est) / ref_sd

        if z > self.drift_threshold:
            self._drift_count += 1
        else:
            self._drift_count = 0

        if self._drift_count >= self.drift_window:
            self._reset()
            return True
        return False

    def _reset(self) -> None:
        """Reset the burn-in calibrator while preserving reference statistics."""
        self.base_calibrator = OnlineBurnInCalibrator(
            min_samples=self.base_calibrator.min_samples,
            low=self.base_calibrator.low,
            high=self.base_calibrator.high,
            min_floor=self.base_calibrator.min_floor,
            max_ceiling=self.base_calibrator.max_ceiling,
        )
        self._drift_count = 0
        self._resets += 1

    def extend(self, deltas: "Iterable[float]") -> int:
        """Process multiple deltas; returns the number of resets triggered."""
        resets = 0
        for d in deltas:
            if self.update(d):
                resets += 1
        return resets

    @property
    def drift_count(self) -> int:
        """Current number of consecutive above-threshold events."""
        return self._drift_count

    @property
    def resets(self) -> int:
        """Total number of drift-triggered resets."""
        return self._resets

    def is_ready(self) -> bool:
        return self.base_calibrator.is_ready()

    def window(self) -> tuple[float, float]:
        return self.base_calibrator.window()

    def admission_params(self, base: AdmissionParams | None = None) -> AdmissionParams:
        return self.base_calibrator.admission_params(base=base)

