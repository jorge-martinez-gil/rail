"""Trace-capturing replay for the RAIL-H paper.

Replays a policy over a stream exactly as :func:`experiments.rail_paper.replay_once`
(legacy policies) or :func:`experiments.replay_integrated.replay_once_integrated`
(stateful policies) would, but records the per-event admission trace so that
*temporal* contamination metrics (AUCC, NCG, time-to-first-contamination,
contamination half-life, yield loss) can be computed from a single pass.

For the RAIL-H hybrid policies the per-event pass/fail of the two component
gates (vigilance, trimmed loss) is recorded separately, which allows

* empirical estimation of the conjunction's false-admission and
  clean-retention rates (alpha, rho) for the contamination contract,
* a direct test of the conditional-independence approximation
  ``alpha_conj ~= alpha_V * alpha_L`` via the phi correlation of the two
  gate indicators on contaminated and clean events.

The replay loops below mirror the reference implementations statement by
statement; any divergence would invalidate comparisons against published
run_metrics.csv rows, so please keep them in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

try:
    from .baselines_integrated import StatefulPolicy, _cross_entropy
    from .metrics import (
        admission_yield_loss,
        area_under_contamination_curve,
        contamination_half_life,
        normalised_contamination_gain,
        time_to_first_contamination,
    )
    from .rail_hybrid import RailHybridPolicy
    from .rail_paper import (
        EPS,
        PolicyConfig,
        ReplayEvent,
        SklearnOnlineClassifier,
        compute_policy_score,
        policy_admits,
        policy_weight,
    )
except ImportError:  # pragma: no cover - script-style fallback
    from baselines_integrated import StatefulPolicy, _cross_entropy  # type: ignore
    from metrics import (  # type: ignore
        admission_yield_loss,
        area_under_contamination_curve,
        contamination_half_life,
        normalised_contamination_gain,
        time_to_first_contamination,
    )
    from rail_hybrid import RailHybridPolicy  # type: ignore
    from rail_paper import (  # type: ignore
        EPS,
        PolicyConfig,
        ReplayEvent,
        SklearnOnlineClassifier,
        compute_policy_score,
        policy_admits,
        policy_weight,
    )


@dataclass
class TraceMetrics:
    """Per-run row: headline metrics + temporal metrics + gate diagnostics."""

    dataset: str
    method: str
    run_id: int
    final_macro_f1: float
    contaminated_admissions: int
    admitted_feedback: int
    total_feedback: int
    admitted_yield: float
    ae: float
    # Temporal admission-quality metrics (experiments.metrics).
    aucc: float
    ncg: float
    ttfc: int
    half_life: int
    yield_loss: float
    # Gate diagnostics -- populated only for RAIL-H policies (else NaN).
    alpha_v: float = float("nan")   # P(V-gate passes | contaminated)
    alpha_l: float = float("nan")   # P(loss-gate passes | contaminated)
    alpha_conj: float = float("nan")  # P(both pass | contaminated)
    rho_v: float = float("nan")     # P(V-gate passes | clean)
    rho_l: float = float("nan")     # P(loss-gate passes | clean)
    rho_conj: float = float("nan")  # P(both pass | clean)
    phi_contaminated: float = float("nan")  # gate-indicator phi corr | contaminated
    phi_clean: float = float("nan")         # gate-indicator phi corr | clean
    base_contamination: float = float("nan")


def _phi(a: np.ndarray, b: np.ndarray) -> float:
    """Phi (Matthews) correlation of two binary vectors; NaN if degenerate."""
    if len(a) == 0:
        return float("nan")
    a = a.astype(float)
    b = b.astype(float)
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _finalize(
    dataset_name: str,
    method: str,
    run_id: int,
    model: SklearnOnlineClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    admitted: List[bool],
    correct: List[bool],
    always_reference_counts: Optional[tuple],
    gate_v: Optional[List[bool]] = None,
    gate_l: Optional[List[bool]] = None,
) -> TraceMetrics:
    contaminated_admissions = sum(1 for a, ok in zip(admitted, correct) if a and not ok)
    admitted_feedback = sum(admitted)
    total_feedback = len(admitted)
    admitted_yield = admitted_feedback / max(total_feedback, 1)
    final_macro_f1 = model.evaluate_macro_f1(X_test, y_test)

    if always_reference_counts is None:
        c_always, y_always = contaminated_admissions, admitted_feedback
    else:
        c_always, y_always = always_reference_counts
    if y_always == admitted_feedback:
        ae = 0.0
    else:
        ae = float((c_always - contaminated_admissions) / (y_always - admitted_feedback + EPS))

    row = TraceMetrics(
        dataset=dataset_name,
        method=method,
        run_id=run_id,
        final_macro_f1=final_macro_f1,
        contaminated_admissions=contaminated_admissions,
        admitted_feedback=admitted_feedback,
        total_feedback=total_feedback,
        admitted_yield=admitted_yield,
        ae=ae,
        aucc=area_under_contamination_curve(correct, admitted),
        ncg=normalised_contamination_gain(correct, admitted),
        ttfc=time_to_first_contamination(correct, admitted),
        half_life=contamination_half_life(correct, admitted),
        yield_loss=admission_yield_loss(correct, admitted),
        base_contamination=(sum(1 for ok in correct if not ok) / max(total_feedback, 1)),
    )

    if gate_v is not None and gate_l is not None:
        gv = np.asarray(gate_v, dtype=bool)
        gl = np.asarray(gate_l, dtype=bool)
        contam = ~np.asarray(correct, dtype=bool)
        clean = ~contam
        if contam.any():
            row.alpha_v = float(gv[contam].mean())
            row.alpha_l = float(gl[contam].mean())
            row.alpha_conj = float((gv & gl)[contam].mean())
            row.phi_contaminated = _phi(gv[contam], gl[contam])
        if clean.any():
            row.rho_v = float(gv[clean].mean())
            row.rho_l = float(gl[clean].mean())
            row.rho_conj = float((gv & gl)[clean].mean())
            row.phi_clean = _phi(gv[clean], gl[clean])
    return row


def replay_once_traced(
    dataset_name: str,
    base_model: SklearnOnlineClassifier,
    events: Sequence[ReplayEvent],
    X_test: np.ndarray,
    y_test: np.ndarray,
    policy: PolicyConfig,
    run_id: int,
    always_reference_counts: Optional[tuple] = None,
) -> TraceMetrics:
    """Mirror of rail_paper.replay_once with per-event trace capture."""
    model = base_model.clone()
    admitted: List[bool] = []
    correct: List[bool] = []
    for ev in events:
        probs = model.predict_proba(ev.x)
        score = compute_policy_score(policy, probs, ev.y_human, ev.telemetry)
        admit = policy_admits(policy, score)
        weight = policy_weight(policy, score)
        admitted.append(bool(admit))
        correct.append(bool(ev.human_feedback_is_correct))
        if weight > 0.0:
            model.update(ev.x, ev.y_human, sample_weight=weight)
    return _finalize(
        dataset_name, policy.name, run_id, model, X_test, y_test,
        admitted, correct, always_reference_counts,
    )


def replay_once_integrated_traced(
    dataset_name: str,
    base_model: SklearnOnlineClassifier,
    events: Sequence[ReplayEvent],
    X_test: np.ndarray,
    y_test: np.ndarray,
    policy: StatefulPolicy,
    run_id: int,
    always_reference_counts: Optional[tuple] = None,
) -> TraceMetrics:
    """Mirror of replay_integrated.replay_once_integrated with trace capture.

    For :class:`RailHybridPolicy` (and subclasses) the two component gates are
    evaluated explicitly, replicating ``RailHybridPolicy.admits`` statement by
    statement (the loss gate must observe *every* event), so the recorded
    conjunction is bit-identical to the policy's own decision.
    """
    model = base_model.clone()
    is_hybrid = isinstance(policy, RailHybridPolicy)
    admitted: List[bool] = []
    correct: List[bool] = []
    gate_v: List[bool] = []
    gate_l: List[bool] = []

    for ev in events:
        probs = model.predict_proba(ev.x)
        if is_hybrid:
            # Replicates RailHybridPolicy.admits: loss gate first and
            # unconditionally, then the pure vigilance check.
            loss_ok = bool(policy._loss_gate.decide(-_cross_entropy(probs, ev.y_human)))
            vig_ok = policy.vigilance(ev.telemetry) >= policy.theta
            admit = loss_ok and vig_ok
            gate_l.append(loss_ok)
            gate_v.append(vig_ok)
        else:
            admit = policy.admits(probs, ev.y_human, ev.telemetry)
        if type(policy).weight is StatefulPolicy.weight:
            weight = 1.0 if admit else 0.0
        else:
            weight = policy.weight(probs, ev.y_human, ev.telemetry)
        admitted.append(bool(admit))
        correct.append(bool(ev.human_feedback_is_correct))
        if weight > 0.0:
            model.update(ev.x, ev.y_human, sample_weight=weight)

    return _finalize(
        dataset_name, policy.name, run_id, model, X_test, y_test,
        admitted, correct, always_reference_counts,
        gate_v=gate_v if is_hybrid else None,
        gate_l=gate_l if is_hybrid else None,
    )


TRACE_FIELDS = list(TraceMetrics.__dataclass_fields__)

__all__ = [
    "TraceMetrics",
    "TRACE_FIELDS",
    "replay_once_traced",
    "replay_once_integrated_traced",
]
