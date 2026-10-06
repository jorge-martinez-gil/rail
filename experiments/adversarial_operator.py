"""Adversarial / misspecified operator models for the RAIL-H robustness study.

The self-contained benchmark generates telemetry with
:func:`experiments.rail_paper.simulate_events`, in which contamination is
*telemetry-correlated by construction*: overloaded operators are both more
error-prone and behaviourally distinguishable (hasty or over-long dwell, low
focus, few edits). Any vigilance-style gate benefits from that coupling.

This module provides operator models that deliberately *break* the coupling,
to measure how admission policies degrade when the behavioural signal is
absent or misleading:

``standard``
    The published generator, reproduced verbatim (control condition).

``blind``
    Telemetry-independent contamination. Correctness is an i.i.d. coin flip
    with the same *marginal* contamination rate as ``standard``, and telemetry
    is drawn from the same state mixture regardless of correctness. The
    vigilance score carries zero information about contamination, so a
    telemetry-only gate can at best subsample the stream at random.

``inverted``
    Fast-but-correct experts and deliberate-looking errors. A fraction of
    events comes from experienced operators who answer inside the *hasty*
    band yet are almost always correct; contaminated events are produced
    with *good-band* dwell, normal focus, and normal edit counts (plausible
    deliberation). The telemetry signal is anti-correlated with reliability:
    a vigilance gate now preferentially rejects clean labels and admits
    contaminated ones.

All three conditions share the marginal contamination rate (up to Monte Carlo
noise) so that differences in admitted-stream purity are attributable to the
telemetry-contamination coupling, not to the difficulty of the stream.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from .rail_paper import (
        DatasetBundle,
        ReplayEvent,
        Telemetry,
        make_multiclass_data,
        preprocess_splits,
        simulate_events,
    )
except ImportError:  # pragma: no cover - script-style fallback
    from rail_paper import (  # type: ignore
        DatasetBundle,
        ReplayEvent,
        Telemetry,
        make_multiclass_data,
        preprocess_splits,
        simulate_events,
    )

CONDITIONS = ("standard", "blind", "inverted")

# Shared band geometry (identical to the published synthetic benchmark).
FAST_BAND = (0.1, 0.8)
GOOD_BAND = (1.2, 5.5)
SLOW_BAND = (7.0, 12.0)
FOCUS_NORMAL = (1.5, 4.0)
FOCUS_OVERLOAD = (0.1, 1.3)
EDITS_NORMAL = (1, 3)
EDITS_OVERLOAD = (0, 1)

# Published synthetic replay parameters (rail_paper.make_synthetic_dataset).
OVERLOAD_PROB = 0.38
CORRECT_NORMAL = 0.89
CORRECT_OVERLOAD = 0.66


def marginal_contamination(
    overload_prob: float = OVERLOAD_PROB,
    correct_normal: float = CORRECT_NORMAL,
    correct_overload: float = CORRECT_OVERLOAD,
) -> float:
    """Marginal P(contaminated) of the standard generator."""
    p_correct = (1.0 - overload_prob) * correct_normal + overload_prob * correct_overload
    return 1.0 - p_correct


def _wrong_label(rng: np.random.Generator, y_true: int, n_classes: int) -> int:
    wrong = [c for c in range(n_classes) if c != y_true]
    return int(rng.choice(wrong))


def _telemetry_from_state(
    rng: np.random.Generator, overload: bool, n_features: int
) -> Telemetry:
    """Draw telemetry from the standard state mixture (no correctness link)."""
    anchor = float(rng.uniform(0.0, 1.0))
    if overload:
        if rng.random() < 0.5:
            delta = float(rng.uniform(*FAST_BAND))
        else:
            delta = float(rng.uniform(*SLOW_BAND))
        focus = float(rng.uniform(*FOCUS_OVERLOAD))
        edits = int(rng.integers(EDITS_OVERLOAD[0], EDITS_OVERLOAD[1] + 1))
    else:
        delta = float(rng.uniform(*GOOD_BAND))
        focus = float(rng.uniform(*FOCUS_NORMAL))
        edits = int(rng.integers(EDITS_NORMAL[0], EDITS_NORMAL[1] + 1))
    return Telemetry(
        anchor_time_s=anchor,
        decision_time_s=anchor + delta,
        focus_time_s=focus,
        edit_count=edits,
        num_features_shown=n_features,
    )


def simulate_events_blind(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
    contamination: float,
    overload_prob: float = OVERLOAD_PROB,
) -> List[ReplayEvent]:
    """Contamination i.i.d. and independent of telemetry.

    The workload state (and therefore the telemetry distribution) evolves as
    in the standard generator, but correctness is decoupled from it: telemetry
    conveys no information about label quality.
    """
    n_features = min(X.shape[1], 24)
    events: List[ReplayEvent] = []
    for i in range(len(X)):
        y_true = int(y[i])
        overload = rng.random() < overload_prob
        is_correct = bool(rng.random() >= contamination)
        telemetry = _telemetry_from_state(rng, overload, n_features)
        y_human = y_true if is_correct else _wrong_label(rng, y_true, n_classes)
        events.append(
            ReplayEvent(
                x=X[i],
                y_true=y_true,
                y_human=y_human,
                human_feedback_is_correct=is_correct,
                telemetry=telemetry,
            )
        )
    return events


def simulate_events_inverted(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
    contamination: float,
    expert_frac: float = 0.5,
    expert_correct: float = 0.97,
) -> List[ReplayEvent]:
    """Fast-but-correct experts + deliberate-looking errors.

    A fraction ``expert_frac`` of events comes from experts who answer inside
    the hasty band with correctness ``expert_correct``. The remaining events
    are calibrated so that the marginal contamination matches ``contamination``;
    crucially, *incorrect* events are emitted with good-band dwell, normal
    focus, and normal edit counts, i.e. they look like careful deliberation.
    """
    n_features = min(X.shape[1], 24)
    # Solve for the non-expert correctness that preserves the marginal rate:
    # (1-f)*(1-c_ne) + f*(1-c_e) = contamination.
    f = expert_frac
    c_ne = 1.0 - (contamination - f * (1.0 - expert_correct)) / (1.0 - f)
    c_ne = float(np.clip(c_ne, 0.0, 1.0))

    events: List[ReplayEvent] = []
    for i in range(len(X)):
        y_true = int(y[i])
        is_expert = rng.random() < f
        if is_expert:
            is_correct = bool(rng.random() < expert_correct)
            # Experts answer fast regardless of correctness.
            anchor = float(rng.uniform(0.0, 1.0))
            delta = float(rng.uniform(*FAST_BAND))
            focus = float(rng.uniform(*FOCUS_OVERLOAD))
            edits = int(rng.integers(EDITS_OVERLOAD[0], EDITS_OVERLOAD[1] + 1))
        else:
            is_correct = bool(rng.random() < c_ne)
            anchor = float(rng.uniform(0.0, 1.0))
            if is_correct:
                # Clean non-expert events: standard state mixture.
                if rng.random() < OVERLOAD_PROB:
                    delta = float(
                        rng.uniform(*FAST_BAND) if rng.random() < 0.5 else rng.uniform(*SLOW_BAND)
                    )
                    focus = float(rng.uniform(*FOCUS_OVERLOAD))
                    edits = int(rng.integers(EDITS_OVERLOAD[0], EDITS_OVERLOAD[1] + 1))
                else:
                    delta = float(rng.uniform(*GOOD_BAND))
                    focus = float(rng.uniform(*FOCUS_NORMAL))
                    edits = int(rng.integers(EDITS_NORMAL[0], EDITS_NORMAL[1] + 1))
            else:
                # Contaminated events look like careful deliberation.
                delta = float(rng.uniform(*GOOD_BAND))
                focus = float(rng.uniform(*FOCUS_NORMAL))
                edits = int(rng.integers(EDITS_NORMAL[0], EDITS_NORMAL[1] + 1))
        telemetry = Telemetry(
            anchor_time_s=anchor,
            decision_time_s=anchor + delta,
            focus_time_s=focus,
            edit_count=edits,
            num_features_shown=n_features,
        )
        y_human = y_true if is_correct else _wrong_label(rng, y_true, n_classes)
        events.append(
            ReplayEvent(
                x=X[i],
                y_true=y_true,
                y_human=y_human,
                human_feedback_is_correct=is_correct,
                telemetry=telemetry,
            )
        )
    return events


def build_adversarial_bundle(condition: str, seed: int) -> DatasetBundle:
    """Synthetic bundle (published geometry) under the given operator model.

    Data geometry matches ``rail_paper.make_synthetic_dataset``:
    12 features, 4 classes, 500 warmup / 250 validation / 2400 replay /
    600 test, drift levels 0.0 / 0.25 / 0.55 / 0.55.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    rng = np.random.default_rng(seed)
    n_features, n_classes = 12, 4
    bias = np.array([0.1, -0.1, 0.0, 0.0])
    X_warmup, y_warmup = make_multiclass_data(rng, 500, n_features, n_classes, 0.0, bias, 0.9, 1.0)
    X_val, y_val = make_multiclass_data(rng, 250, n_features, n_classes, 0.25, bias, 0.9, 1.0)
    X_rep, y_rep = make_multiclass_data(rng, 2400, n_features, n_classes, 0.55, bias, 0.9, 1.0)
    X_test, y_test = make_multiclass_data(rng, 600, n_features, n_classes, 0.55, bias, 0.9, 1.0)
    X_warmup, X_val, X_rep, X_test = preprocess_splits(X_warmup, X_val, X_rep, X_test)

    c = marginal_contamination()

    def _events(X, y):
        if condition == "standard":
            return simulate_events(
                X, y, n_classes, rng,
                overload_prob=OVERLOAD_PROB,
                correct_prob_normal=CORRECT_NORMAL,
                correct_prob_overload=CORRECT_OVERLOAD,
                fast_band=FAST_BAND,
                good_band=GOOD_BAND,
                slow_band=SLOW_BAND,
                focus_normal=FOCUS_NORMAL,
                focus_overload=FOCUS_OVERLOAD,
                edits_normal=EDITS_NORMAL,
                edits_overload=EDITS_OVERLOAD,
            )
        if condition == "blind":
            return simulate_events_blind(X, y, n_classes, rng, contamination=c)
        return simulate_events_inverted(X, y, n_classes, rng, contamination=c)

    return DatasetBundle(
        name=f"adv_{condition}",
        X_warmup=X_warmup,
        y_warmup=y_warmup,
        validation_events=_events(X_val, y_val),
        replay_events=_events(X_rep, y_rep),
        X_test=X_test,
        y_test=y_test,
    )


__all__ = [
    "CONDITIONS",
    "build_adversarial_bundle",
    "marginal_contamination",
    "simulate_events_blind",
    "simulate_events_inverted",
]
