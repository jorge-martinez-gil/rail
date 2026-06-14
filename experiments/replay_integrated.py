"""Replay loop that drives integrated stateful policies for the headline tables.

The existing :func:`experiments.rail_paper.replay_once` consults
``compute_policy_score``/``policy_admits``/``policy_weight`` via the
``PolicyConfig`` dataclass. The integrated noise-robust baselines maintain
mutable state (running quantiles, age counters) and so cannot be expressed
as a frozen ``PolicyConfig``.

This module provides a thin sibling replay loop that accepts a
:class:`experiments.baselines_integrated.StatefulPolicy` directly, returning
a :class:`experiments.rail_paper.RunMetrics` row that is bit-identical in
schema to the legacy result so it can be concatenated and summarised with
``summarize_runs``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from .baselines_integrated import StatefulPolicy
    from .rail_paper import (
        ReplayEvent,
        RunMetrics,
        SklearnOnlineClassifier,
    )
except ImportError:  # pragma: no cover
    from baselines_integrated import StatefulPolicy  # type: ignore[no-redef]
    from rail_paper import (  # type: ignore[no-redef]
        ReplayEvent,
        RunMetrics,
        SklearnOnlineClassifier,
    )

EPS = 1e-12


def replay_once_integrated(
    dataset_name: str,
    base_model: SklearnOnlineClassifier,
    events: Sequence[ReplayEvent],
    X_test: np.ndarray,
    y_test: np.ndarray,
    policy: StatefulPolicy,
    run_id: int,
    always_reference_counts: tuple[int, int] | None = None,
) -> RunMetrics:
    """Run one full replay for a stateful integrated policy.

    Mirrors :func:`experiments.rail_paper.replay_once` exactly except that
    admission and weight decisions are obtained from ``policy.admits`` and
    ``policy.weight`` (driven by the policy's internal state).
    """

    model = base_model.clone()
    contaminated_admissions = 0
    admitted_feedback = 0
    total_feedback = 0

    for ev in events:
        probs = model.predict_proba(ev.x)
        admit = policy.admits(probs, ev.y_human, ev.telemetry)
        if type(policy).weight is StatefulPolicy.weight:
            weight = 1.0 if admit else 0.0
        else:
            weight = policy.weight(probs, ev.y_human, ev.telemetry)
        total_feedback += 1
        if admit:
            admitted_feedback += 1
            if not ev.human_feedback_is_correct:
                contaminated_admissions += 1
        if weight > 0.0:
            model.update(ev.x, ev.y_human, sample_weight=weight)

    final_macro_f1 = model.evaluate_macro_f1(X_test, y_test)
    admitted_yield = admitted_feedback / max(total_feedback, 1)

    if always_reference_counts is None:
        c_always, y_always = contaminated_admissions, admitted_feedback
    else:
        c_always, y_always = always_reference_counts

    if y_always == admitted_feedback:
        ae = 0.0
    else:
        ae = float((c_always - contaminated_admissions) / (y_always - admitted_feedback + EPS))

    return RunMetrics(
        dataset=dataset_name,
        method=policy.name,
        run_id=run_id,
        final_macro_f1=final_macro_f1,
        contaminated_admissions=contaminated_admissions,
        admitted_feedback=admitted_feedback,
        total_feedback=total_feedback,
        admitted_yield=admitted_yield,
        ae=ae,
    )


def run_all_integrated_policies(
    dataset_name: str,
    base_model: SklearnOnlineClassifier,
    events: Sequence[ReplayEvent],
    X_test: np.ndarray,
    y_test: np.ndarray,
    policies: Sequence[StatefulPolicy],
    always_reference_counts: tuple[int, int],
    run_id: int,
) -> list[RunMetrics]:
    """Run every integrated policy on the same event stream for one seed."""

    out: list[RunMetrics] = []
    for p in policies:
        out.append(
            replay_once_integrated(
                dataset_name=dataset_name,
                base_model=base_model,
                events=events,
                X_test=X_test,
                y_test=y_test,
                policy=p,
                run_id=run_id,
                always_reference_counts=always_reference_counts,
            )
        )
    return out


__all__ = ["replay_once_integrated", "run_all_integrated_policies"]
