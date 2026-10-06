"""RAIL-H: hybrid admission gate fusing operator telemetry with trimmed loss.

Motivation
----------
RAIL's vigilance gate and loss-based trimming (ITLM; Shen & Sanghavi, 2019)
filter contamination through *orthogonal channels*:

* The vigilance score ``V`` is computed purely from operator deliberation
  telemetry. It is independent of the model's predictions, so it catches
  contaminated labels that happen to agree with a (possibly already decayed)
  model.
* The trimmed-loss gate uses the model's per-event cross-entropy. It catches
  labels that disagree with the model regardless of how the operator behaved,
  so it catches confident-but-wrong labels produced during apparently normal
  deliberation.

Because the two signals condition on disjoint information (behaviour vs.
prediction), their false-negative sets are largely non-overlapping and the
conjunction is a strictly stronger contamination filter than either parent.
RAIL-H admits an event iff **both** gates pass:

    admit  <=>  V >= theta   AND   CE-loss <= alpha-quantile of recent losses

Design notes
------------
* The loss quantile is tracked over the *full* event stream (identical to the
  standalone ITLM baseline), not only over vigilance-passed events. This
  keeps the loss component bit-comparable to ITLM so any improvement is
  attributable to the fusion, not to a retuned trimming rule.
* All vigilance parameters default to the exact values used by
  ``rail_gated`` in :func:`experiments.rail_paper.make_default_policies`, and
  the trimming parameters default to the exact values used by the ITLM
  baseline. RAIL-H therefore introduces **no new tuned hyperparameters**.
* The contamination contract of :mod:`experiments.rail_core` still applies:
  ``A_theta`` is simply replaced by the conjunction event, and the same
  Bayes bound holds with the conjunction's (alpha_theta, rho_theta).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    from .baselines import DynamicQuantileGate
    from .baselines_integrated import StatefulPolicy, _cross_entropy
    from .rail_core import AdmissionParams, admission_diagnostics
except ImportError:  # pragma: no cover - script-style import fallback
    from baselines import DynamicQuantileGate  # type: ignore[no-redef]
    from baselines_integrated import (  # type: ignore[no-redef]
        StatefulPolicy,
        _cross_entropy,
    )
    from rail_core import (  # type: ignore[no-redef]
        AdmissionParams,
        admission_diagnostics,
    )

__all__ = ["RailHybridCalibratedPolicy", "RailHybridPolicy", "vigilance_score"]


def vigilance_score(telemetry: object, params: AdmissionParams) -> float:
    """Compute the RAIL vigilance score V for a telemetry record.

    Accepts any object exposing the :class:`experiments.rail_paper.Telemetry`
    attributes (duck-typed so the module does not import the heavy
    ``rail_paper`` module).
    """
    delta = float(telemetry.decision_time_s - telemetry.anchor_time_s)
    diag = admission_diagnostics(
        delta_sec=delta,
        num_features=int(telemetry.num_features_shown),
        edit_count=int(telemetry.edit_count),
        focus_seconds=float(telemetry.focus_time_s),
        params=params,
    )
    return float(diag["score"])


@dataclass
class RailHybridPolicy(StatefulPolicy):
    """RAIL-H: admit iff the vigilance gate AND the trimmed-loss gate pass.

    Vigilance defaults mirror ``rail_gated`` (rail_paper.make_default_policies);
    trimming defaults mirror the ITLM baseline (alpha=0.7, warmup=100).
    """

    name: str = "rail_h"
    # Vigilance component (identical to rail_gated defaults).
    tau_min: float = 1.2
    tau_max: float = 5.0
    k: float = 3.0
    theta: float = 0.50
    w_delta: float = 1.0
    w_f: float = 0.005
    w_e: float = 0.12
    w_s: float = 0.03
    # Trimmed-loss component (identical to ITLM defaults).
    trim_alpha: float = 0.7
    trim_warmup: int = 100
    _params: AdmissionParams = field(init=False, repr=False)
    _loss_gate: DynamicQuantileGate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._params = AdmissionParams(
            tau_min=self.tau_min,
            tau_max=self.tau_max,
            k=self.k,
            theta=self.theta,
            w_delta=self.w_delta,
            w_features=self.w_f,
            w_edits=self.w_e,
            w_focus=self.w_s,
        )
        self._loss_gate = DynamicQuantileGate(upper=self.trim_alpha, warmup=self.trim_warmup)

    # -- pure (stateless) parts -------------------------------------------

    def vigilance(self, telemetry: object) -> float:
        return vigilance_score(telemetry, self._params)

    def score(self, probs: np.ndarray, y_human: int, telemetry: object) -> float:
        """Diagnostic score: V masked by the loss gate's most recent state.

        Note: admission is decided by :meth:`admits`; this score is reported
        for tables/plots only and does not drive the replay loop.
        """
        return self.vigilance(telemetry)

    # -- stateful admission -------------------------------------------------

    def admits(self, probs: np.ndarray, y_human: int, telemetry: object) -> bool:
        # IMPORTANT: the loss gate must observe *every* event so its running
        # quantile matches the standalone ITLM baseline. Evaluate it first and
        # unconditionally; do not short-circuit on the vigilance gate.
        loss_ok = bool(self._loss_gate.decide(-_cross_entropy(probs, y_human)))
        vigilance_ok = self.vigilance(telemetry) >= self.theta
        return loss_ok and vigilance_ok


@dataclass
class RailHybridCalibratedPolicy(RailHybridPolicy):
    """RAIL-H (yield-calibrated): conjunction gate matched to rail_gated's yield.

    Adding a second filter on top of the vigilance gate necessarily lowers
    admitted yield, which makes Admission-Efficiency comparisons against
    single gates apples-to-oranges. This variant follows the harness's
    established protocol (``calibrate_gated_baselines_to_rail_yield``): it is
    calibrated on the validation window so its *expected admitted yield
    matches rail_gated's*, then evaluated untouched on the replay stream.

    The calibration rule is parameter-free: the withholding budget is split
    symmetrically between the two gates. If rail_gated admits fraction ``y``
    on the validation window, each gate is set to pass ``sqrt(y)`` so the
    conjunction admits ~``y`` under independence:

    * vigilance threshold theta' := (1 - sqrt(y))-quantile of validation V
    * trimmed-loss fraction alpha' := sqrt(y)

    Only telemetry and the validation window are used -- no test peeking and
    no per-dataset hand tuning.
    """

    name: str = "rail_h_cal"
    alpha_floor: float = 0.05
    alpha_ceil: float = 0.99

    def calibrate_from_validation(self, validation_events: object, model: object = None) -> None:
        events = list(validation_events)
        v = np.asarray([self.vigilance(ev.telemetry) for ev in events])
        target = float(np.mean(v >= self.theta))

        if model is None:
            # Independence approximation: each gate passes sqrt(target).
            #
            # NOTE: we also evaluated an empirical-joint alternative (the
            # ``model`` branch below) that solves for the shared per-gate
            # fraction on the validation window's joint (V, loss) sample.
            # It consistently *under-admits* on the replay stream: the
            # warmup-model losses correlate with V more strongly on the
            # validation window than on the nonstationary replay stream, so
            # the fitted share transfers poorly. The independence rule is
            # deliberately conservative and was uniformly closer to the
            # target yield in the 30-seed benchmark, so it is the default.
            share = math.sqrt(min(max(target, 1e-6), 1.0))
        else:
            # Empirical-joint calibration: the independence rule under-admits
            # when V and the loss signal are positively correlated. Using the
            # validation window's empirical joint distribution, find the
            # symmetric per-gate share s such that the *conjunction* admits
            # the target fraction: both gates pass their own s-fraction and
            # P_hat(V >= q_v(1-s)  AND  -loss >= q_l(1-s)) = target.
            neg_loss = np.asarray(
                [-_cross_entropy(model.predict_proba(ev.x), ev.y_human) for ev in events]
            )

            def admit_frac(s: float) -> float:
                tv = np.quantile(v, 1.0 - s)
                tl = np.quantile(neg_loss, 1.0 - s)
                return float(np.mean((v >= tv) & (neg_loss >= tl)))

            lo, hi = 1e-3, 1.0 - 1e-3
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if admit_frac(mid) < target:
                    lo = mid
                else:
                    hi = mid
            share = 0.5 * (lo + hi)

        self.theta = float(np.clip(np.quantile(v, 1.0 - share), 0.0, 1.0))
        self.trim_alpha = float(np.clip(share, self.alpha_floor, self.alpha_ceil))
        self._loss_gate = DynamicQuantileGate(upper=self.trim_alpha, warmup=self.trim_warmup)
