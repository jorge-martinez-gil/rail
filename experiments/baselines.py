"""Additional noise-robust admission baselines for the RAIL experiments.

The existing experiment harness (``experiments.rail_paper``) ships three
strong sample-selection baselines: ``confidence_gated``, ``loss_gated``,
and ``margin_gated``. To strengthen the empirical comparison expected by a
Information Systems submission we add four classical noise-robust families, each exposed as
a small, dependency-light Python class with the same ``decide(event)``
contract so they can be dropped into any replay loop:

* :class:`CoTeachingGate` -- two-network sample selection (Han et al., 2018);
  filters the bottom-``R(t)``-percentile of losses computed by two
  independently maintained mini-models.
* :class:`SelfPacedGate` -- self-paced / curriculum admission (Kumar et al.,
  2010; Jiang et al., 2018); admits samples whose loss is below an
  age-dependent threshold ``lambda(t)``.
* :class:`JointAgreementGate` -- JoCoR-style joint-agreement filter (Wei
  et al., 2020); admits only samples on which two views agree *and* whose
  combined loss is small.
* :class:`DynamicQuantileGate` -- running-quantile threshold over an
  arbitrary admission score; tracks regime changes online without retraining.

The baselines are intentionally engine-agnostic: each receives the per-event
loss/score/agreement signal that the harness already computes and returns a
boolean admission decision. This keeps the comparison clean -- nothing in
the rail core needs to change, and the existing scoring/labelling pipeline
is reused.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

try:
    from .rail_core import _P2Quantile  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from rail_core import _P2Quantile  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Co-teaching (Han et al., NeurIPS 2018).
# ---------------------------------------------------------------------------


@dataclass
class CoTeachingGate:
    """Two-view sample selection inspired by Co-teaching.

    The original method trains two networks and lets each select the small-
    loss instances for the other. We expose the *selection rule* as an
    admission gate that takes two per-event loss signals (``loss_a``,
    ``loss_b``) -- the harness can compute them from any two models or two
    feature subsets.

    Parameters
    ----------
    forget_rate : float
        Final fraction of large-loss events to reject. Typically equal to
        the assumed noise rate.
    ramp_steps : int
        Number of warm-up steps over which the rejection fraction ramps
        linearly from 0 to ``forget_rate`` (paper's R(t)).
    window : int
        Sliding window of recent (loss_a, loss_b) pairs used to estimate
        the per-side quantile threshold. Larger windows are more stable but
        less reactive to drift.
    """

    forget_rate: float = 0.3
    ramp_steps: int = 1000
    window: int = 500
    _hist_a: deque[float] = field(default_factory=deque, init=False, repr=False)
    _hist_b: deque[float] = field(default_factory=deque, init=False, repr=False)
    _t: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.forget_rate < 1.0:
            raise ValueError("forget_rate must lie in [0, 1)")
        if self.ramp_steps < 0:
            raise ValueError("ramp_steps must be non-negative")
        if self.window < 1:
            raise ValueError("window must be >= 1")
        self._hist_a = deque(maxlen=self.window)
        self._hist_b = deque(maxlen=self.window)

    def _current_rate(self) -> float:
        if self.ramp_steps == 0:
            return self.forget_rate
        progress = min(1.0, self._t / self.ramp_steps)
        return self.forget_rate * progress

    def decide(self, loss_a: float, loss_b: float) -> bool:
        self._hist_a.append(float(loss_a))
        self._hist_b.append(float(loss_b))
        self._t += 1
        if len(self._hist_a) < 5:
            return True  # not enough history yet
        rate = self._current_rate()
        keep_frac = 1.0 - rate
        thr_a = _quantile(self._hist_a, keep_frac)
        thr_b = _quantile(self._hist_b, keep_frac)
        # Symmetric co-selection: model A selects for B and vice versa.
        return (loss_a <= thr_a) and (loss_b <= thr_b)


# ---------------------------------------------------------------------------
# Self-paced curriculum admission (Kumar et al., 2010; MentorNet, 2018).
# ---------------------------------------------------------------------------


@dataclass
class SelfPacedGate:
    """Curriculum admission with a growing loss tolerance ``lambda(t)``.

    Admits an event iff its loss is below ``lambda(t) = lambda_0 *
    growth ** floor(t / step)``. ``growth > 1`` means the curriculum
    relaxes over time; ``growth = 1`` is a fixed threshold.
    """

    lambda_0: float = 0.5
    growth: float = 1.05
    step: int = 200
    cap: float = 50.0
    _t: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lambda_0 <= 0.0:
            raise ValueError("lambda_0 must be positive")
        if self.growth <= 0.0:
            raise ValueError("growth must be positive")
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if self.cap <= 0.0:
            raise ValueError("cap must be positive")

    def current_lambda(self) -> float:
        epochs = self._t // self.step
        return min(self.cap, self.lambda_0 * (self.growth**epochs))

    def decide(self, loss: float) -> bool:
        lam = self.current_lambda()
        self._t += 1
        return float(loss) <= lam


# ---------------------------------------------------------------------------
# JoCoR-style joint-agreement filter (Wei et al., CVPR 2020).
# ---------------------------------------------------------------------------


@dataclass
class JointAgreementGate:
    """Joint-agreement gate using two predictions and their combined loss.

    Admits an event iff (a) the two predictions agree on the label, and
    (b) their averaged loss is in the keep fraction of recent losses.
    """

    keep_fraction: float = 0.7
    window: int = 500
    _hist: deque[float] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_fraction <= 1.0:
            raise ValueError("keep_fraction must lie in (0, 1]")
        if self.window < 1:
            raise ValueError("window must be >= 1")
        self._hist = deque(maxlen=self.window)

    def decide(self, pred_a: int, pred_b: int, loss_a: float, loss_b: float) -> bool:
        if pred_a != pred_b:
            return False
        combined = 0.5 * (float(loss_a) + float(loss_b))
        self._hist.append(combined)
        if len(self._hist) < 5:
            return True
        thr = _quantile(self._hist, self.keep_fraction)
        return combined <= thr


# ---------------------------------------------------------------------------
# Dynamic-quantile threshold over any admission score.
# ---------------------------------------------------------------------------


@dataclass
class DynamicQuantileGate:
    """Streaming-quantile threshold over an arbitrary admission score.

    Useful as a "no learned model" baseline that still adapts to regime
    shifts: at each step the gate admits the event iff its score is in the
    top ``upper`` fraction of recent scores. Tracking is done via a P^2
    quantile estimator (O(1) memory, O(1) per update).
    """

    upper: float = 0.6
    warmup: int = 50
    _estimator: _P2Quantile = field(init=False, repr=False)
    _seen: int = field(default=0, init=False, repr=False)
    _last_value: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.upper < 1.0:
            raise ValueError("upper must lie in (0, 1)")
        if self.warmup < 1:
            raise ValueError("warmup must be >= 1")
        # We track the (1 - upper) quantile: admit iff score >= q_(1-upper).
        self._estimator = _P2Quantile(1.0 - self.upper)

    def decide(self, score: float) -> bool:
        self._estimator.update(float(score))
        self._seen += 1
        if self._seen < self.warmup:
            return True
        try:
            thr = self._estimator.value()
        except ValueError:  # pragma: no cover - safe-guarded by warmup
            return True
        self._last_value = thr
        return float(score) >= thr

    @property
    def current_threshold(self) -> float:
        return self._last_value


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _quantile(values: Iterable[float], q: float) -> float:
    """Linear-interpolation quantile over a small iterable (used for windows)."""

    vs = sorted(float(v) for v in values if math.isfinite(v))
    if not vs:
        raise ValueError("no values")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    idx = q * (len(vs) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vs[lo]
    frac = idx - lo
    return vs[lo] * (1.0 - frac) + vs[hi] * frac


__all__ = [
    "CoTeachingGate",
    "DynamicQuantileGate",
    "JointAgreementGate",
    "SelfPacedGate",
]
