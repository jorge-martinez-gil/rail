"""Reference streaming operator for the RAIL ingress layer.

This module is the artifact that backs up the paper's claim (Section 3.4)
that RAIL "can be integrated as a lightweight pre-processing operator in
existing stream processing systems (e.g., Apache Flink or Kafka), requiring
constant-time computation per event."

It exposes :class:`RailOperator`, an engine-agnostic stream operator that:

1. Consumes interaction events (one per operator decision).
2. Maintains an online burn-in calibrator over anchored deliberation
   :math:`\\Delta` and (optionally) auto-swaps the admission window once
   enough samples have been observed.
3. Computes the admission score :math:`V` and applies the configured policy
   (``"gated"`` or ``"weighted"``).
4. Emits a structured audit record that mirrors the export schema used by
   the browser console (``schemas/rail.non_ok_export.v4.schema.json``).

The operator is pure Python -- no Flink/Kafka client is imported here. We
ship adapter recipes (see :func:`flink_map_function` and
:func:`kafka_streams_handler`) that show how the engine-specific glue
collapses to a couple of lines because the per-event work is already
encapsulated.

Design notes
------------

* All per-event work is :math:`\\mathcal{O}(1)` in time and memory.
* The operator is *stateful but checkpointable*: :meth:`RailOperator.snapshot`
  returns a JSON-serialisable dict and :meth:`RailOperator.restore` rehydrates
  the state. This is the contract a Flink ``KeyedProcessFunction`` or a Kafka
  Streams ``Transformer`` needs to participate in exactly-once delivery.
* The operator is deterministic given the same event order and seed; we use no
  hidden randomness.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

try:
    from .rail_core import (
        AdmissionParams,
        OnlineBurnInCalibrator,
        admission_diagnostics,
    )
except ImportError:  # pragma: no cover - allows running as a script
    from rail_core import (
        AdmissionParams,
        OnlineBurnInCalibrator,
        admission_diagnostics,
    )


Policy = Literal["gated", "weighted"]


@dataclass(frozen=True)
class InteractionEvent:
    """An interaction event delivered to the operator.

    Field naming mirrors the JSON shape produced by the browser console so the
    same DTO works in both the embedded reference and in deployment.
    """

    event_id: str
    anchor_time_ms: float
    decision_time_ms: float
    focus_time_s: float
    edit_count: int
    num_features_shown: int
    status: Literal["OK", "Non-OK"]
    corrected_target: Any | None = None
    annotation: str | None = None
    features: Mapping[str, Any] | None = None
    prediction: Any | None = None

    @property
    def delta_sec(self) -> float:
        return max(0.0, (self.decision_time_ms - self.anchor_time_ms) / 1000.0)


@dataclass(frozen=True)
class AdmissionDecision:
    """Audit-grade output of the operator for a single event.

    The shape is a strict superset of the v4 ``non_ok_export`` schema so the
    same record can be written to the durable export sink without a
    transformation step.
    """

    event_id: str
    admitted: bool
    score: float
    policy: Policy
    update_weight: float
    delta_sec: float
    beta: float
    s_fast: float
    s_slow: float
    tau_min: float
    tau_max: float
    theta: float
    calibrator_count: int
    calibrator_ready: bool
    timestamp_emit: float
    status: Literal["OK", "Non-OK"]
    corrected_target: Any | None = None
    annotation: str | None = None
    features: Mapping[str, Any] | None = None
    prediction: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RailOperator:
    """Stateful streaming operator that gates human-feedback ingress.

    Parameters
    ----------
    params : AdmissionParams
        Initial admission parameters. The window can be replaced once the
        online calibrator becomes ready (see ``adopt_calibrated_window``).
    policy : {"gated", "weighted"}
        ``"gated"`` admits only events with ``V >= theta``;
        ``"weighted"`` admits all events but exposes ``update_weight = V``
        that callers should multiply into gradient updates.
    calibrator : OnlineBurnInCalibrator | None
        Optional online calibrator. If ``None``, the operator does not adapt
        the window. Defaults to a 300-sample burn-in matching the paper.
    adopt_calibrated_window : bool
        When ``True``, the first decision after ``calibrator.is_ready()``
        becomes ``True`` swaps ``params.tau_min`` and ``params.tau_max`` for
        the calibrated window. This is reported in :meth:`snapshot`.
    audit_buffer : int
        Optional bounded ring buffer of recent decisions for in-memory
        debugging. ``0`` disables buffering (the default). The audit sink in
        production should be the durable export schema; this buffer is a
        convenience for tests and dashboards.

    Notes
    -----
    Thread safety: the operator is single-threaded by design (Flink/Kafka
    operators are sharded per key). Callers may use one instance per shard.
    """

    params: AdmissionParams = field(default_factory=AdmissionParams)
    policy: Policy = "gated"
    calibrator: OnlineBurnInCalibrator | None = field(
        default_factory=lambda: OnlineBurnInCalibrator(min_samples=300)
    )
    adopt_calibrated_window: bool = True
    audit_buffer: int = 0
    _events_seen: int = field(default=0, init=False, repr=False)
    _events_admitted: int = field(default=0, init=False, repr=False)
    _window_swapped: bool = field(default=False, init=False, repr=False)
    _audit: deque[AdmissionDecision] = field(
        default_factory=lambda: deque(maxlen=0), init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.policy not in ("gated", "weighted"):
            raise ValueError("policy must be 'gated' or 'weighted'")
        if self.audit_buffer < 0:
            raise ValueError("audit_buffer must be non-negative")
        self._audit = deque(maxlen=self.audit_buffer or None)
        if self.audit_buffer == 0:
            # deque with maxlen=None is unbounded; keep a 0-length proxy.
            self._audit = deque(maxlen=0)

    # ------------------------------------------------------------------ stats
    @property
    def events_seen(self) -> int:
        return self._events_seen

    @property
    def events_admitted(self) -> int:
        return self._events_admitted

    @property
    def admission_rate(self) -> float:
        return self._events_admitted / self._events_seen if self._events_seen else 0.0

    @property
    def window_swapped(self) -> bool:
        return self._window_swapped

    def recent_decisions(self) -> tuple[AdmissionDecision, ...]:
        return tuple(self._audit)

    # ------------------------------------------------------------------ core
    def process(self, event: InteractionEvent) -> AdmissionDecision:
        """Process one event and emit a single :class:`AdmissionDecision`."""

        delta = event.delta_sec

        # Update calibrator before scoring so that the very first ready
        # decision can use the freshly calibrated window deterministically.
        if self.calibrator is not None:
            self.calibrator.update(delta)
            if (
                self.adopt_calibrated_window
                and self.calibrator.is_ready()
                and not self._window_swapped
            ):
                self.params = self.calibrator.admission_params(base=self.params)
                self._window_swapped = True

        diag = admission_diagnostics(
            delta_sec=delta,
            num_features=event.num_features_shown,
            edit_count=event.edit_count,
            focus_seconds=event.focus_time_s,
            params=self.params,
        )

        if self.policy == "gated":
            admitted = bool(diag["score"] >= self.params.theta)
            weight = 1.0 if admitted else 0.0
        else:  # weighted
            admitted = True
            weight = float(diag["score"])

        decision = AdmissionDecision(
            event_id=event.event_id,
            admitted=admitted,
            score=float(diag["score"]),
            policy=self.policy,
            update_weight=float(weight),
            delta_sec=float(delta),
            beta=float(diag["beta"]),
            s_fast=float(diag["s_fast"]),
            s_slow=float(diag["s_slow"]),
            tau_min=float(self.params.tau_min),
            tau_max=float(self.params.tau_max),
            theta=float(self.params.theta),
            calibrator_count=self.calibrator.count if self.calibrator else 0,
            calibrator_ready=self.calibrator.is_ready() if self.calibrator else False,
            timestamp_emit=time.time(),
            status=event.status,
            corrected_target=event.corrected_target,
            annotation=event.annotation,
            features=event.features,
            prediction=event.prediction,
        )

        self._events_seen += 1
        if admitted:
            self._events_admitted += 1
        if self.audit_buffer > 0:
            self._audit.append(decision)

        return decision

    def process_batch(self, events: Iterable[InteractionEvent]) -> list[AdmissionDecision]:
        return [self.process(ev) for ev in events]

    # ---------------------------------------------------------- checkpointing
    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot for engine checkpointing."""

        cal_state: dict[str, Any] | None = None
        if self.calibrator is not None:
            lo, hi = self.calibrator.window()
            cal_state = {
                "count": self.calibrator.count,
                "min_samples": self.calibrator.min_samples,
                "low": self.calibrator.low,
                "high": self.calibrator.high,
                "ready_window": [lo, hi] if self.calibrator.is_ready() else None,
            }
        return {
            "params": {
                "tau_min": self.params.tau_min,
                "tau_max": self.params.tau_max,
                "k": self.params.k,
                "theta": self.params.theta,
                "w_delta": self.params.w_delta,
                "w_features": self.params.w_features,
                "w_edits": self.params.w_edits,
                "w_focus": self.params.w_focus,
            },
            "policy": self.policy,
            "events_seen": self._events_seen,
            "events_admitted": self._events_admitted,
            "window_swapped": self._window_swapped,
            "calibrator": cal_state,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore counters and parameters from a previous snapshot.

        Calibrator *quantile* internals are not restored bit-for-bit (the
        five floats per quantile would have to be exposed). Restoring
        ``count`` and the materialised window is enough to continue gating
        without re-burn-in; new samples warm the estimator back up. This
        matches the durability guarantees most stream engines need.
        """

        p = snapshot["params"]
        self.params = AdmissionParams(
            tau_min=p["tau_min"],
            tau_max=p["tau_max"],
            k=p["k"],
            theta=p["theta"],
            w_delta=p["w_delta"],
            w_features=p["w_features"],
            w_edits=p["w_edits"],
            w_focus=p["w_focus"],
        )
        self.policy = snapshot["policy"]
        self._events_seen = int(snapshot.get("events_seen", 0))
        self._events_admitted = int(snapshot.get("events_admitted", 0))
        self._window_swapped = bool(snapshot.get("window_swapped", False))


# ---------------------------------------------------------------------------
# Engine adapter recipes.
#
# We do not import flink or kafka because the artifact must remain
# dependency-light. These factories return small callables that an engine
# integrator can plug in directly. Each is documented as a recipe rather
# than a polished SDK.
# ---------------------------------------------------------------------------


def flink_map_function(operator: RailOperator) -> Callable[[InteractionEvent], dict[str, Any]]:
    """Recipe: a Flink ``MapFunction`` body.

    Usage (pseudo-code on the Flink side):

    .. code-block:: python

        operator = RailOperator(policy="gated")
        rail_fn = flink_map_function(operator)
        stream.map(rail_fn).filter(lambda r: r["admitted"]).sink_to(curated_sink)

    The returned callable is pickle-friendly because the operator state is
    a vanilla dataclass.
    """

    def _fn(event: InteractionEvent) -> dict[str, Any]:
        return operator.process(event).to_dict()

    return _fn


def kafka_streams_handler(
    operator: RailOperator,
) -> Callable[[str, InteractionEvent], tuple[str, dict[str, Any]]]:
    """Recipe: a Kafka Streams ``Transformer`` body (key, value -> key, value).

    The handler preserves the input key so downstream topology partitioning
    is unchanged. The output value is the audit record.
    """

    def _handler(key: str, event: InteractionEvent) -> tuple[str, dict[str, Any]]:
        return key, operator.process(event).to_dict()

    return _handler


# ---------------------------------------------------------------------------
# Demo helpers: useful for notebooks and for the test suite.
# ---------------------------------------------------------------------------


def synthesise_events(
    n: int = 1000,
    seed: int = 0,
    contamination_rate: float = 0.3,
    clean_delta_mean: float = 2.5,
    clean_delta_sd: float = 0.7,
) -> list[tuple[InteractionEvent, bool]]:
    """Generate a synthetic operator-feedback stream for tests/demos.

    Each item is ``(event, is_correct)`` so callers can compute the empirical
    contamination contract after the operator has consumed the stream. Clean
    events draw :math:`\\Delta` from a log-normal centred near the typical
    deliberation; contaminated events are uniformly fast or slow.
    """

    import random

    rng = random.Random(seed)
    out: list[tuple[InteractionEvent, bool]] = []
    for i in range(n):
        if rng.random() < contamination_rate:
            # Contaminated: reflexive or stalled.
            delta = rng.choice([rng.uniform(0.05, 0.4), rng.uniform(10.0, 25.0)])
            is_correct = False
            edits = 0
            focus = 0.1
        else:
            delta = max(0.05, rng.lognormvariate(math.log(clean_delta_mean), clean_delta_sd))
            is_correct = True
            edits = rng.choice([0, 1, 1, 2])
            focus = max(0.1, delta - rng.uniform(0.2, 0.8))

        anchor_ms = 1_000_000.0 + i * 5_000.0
        decision_ms = anchor_ms + delta * 1000.0
        ev = InteractionEvent(
            event_id=f"evt-{i:06d}",
            anchor_time_ms=anchor_ms,
            decision_time_ms=decision_ms,
            focus_time_s=focus,
            edit_count=edits,
            num_features_shown=10,
            status="Non-OK" if rng.random() < 0.5 else "OK",
        )
        out.append((ev, is_correct))
    return out
