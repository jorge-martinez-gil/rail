"""Tests for the reference streaming operator."""

import pytest

from experiments.rail_core import (
    AdmissionParams,
    OnlineBurnInCalibrator,
    contamination_contract,
)
from experiments.rail_operator import (
    InteractionEvent,
    RailOperator,
    flink_map_function,
    kafka_streams_handler,
    synthesise_events,
)


def _ev(delta_sec: float, **kwargs) -> InteractionEvent:
    anchor = 0.0
    decision = delta_sec * 1000.0
    defaults = dict(
        event_id="e",
        anchor_time_ms=anchor,
        decision_time_ms=decision,
        focus_time_s=max(0.1, delta_sec - 0.5),
        edit_count=1,
        num_features_shown=8,
        status="Non-OK",
    )
    defaults.update(kwargs)
    return InteractionEvent(**defaults)


def test_gated_policy_admits_only_in_window():
    op = RailOperator(
        params=AdmissionParams(tau_min=1.0, tau_max=5.0, k=2.0, theta=0.6),
        policy="gated",
        calibrator=None,
    )
    hasty = op.process(_ev(0.1, event_id="hasty"))
    deliberate = op.process(_ev(3.0, event_id="ok"))
    stalled = op.process(_ev(20.0, event_id="stalled"))
    assert not hasty.admitted
    assert deliberate.admitted
    assert not stalled.admitted
    assert op.events_seen == 3
    assert op.events_admitted == 1


def test_weighted_policy_admits_all_and_returns_score_as_weight():
    op = RailOperator(
        params=AdmissionParams(tau_min=1.0, tau_max=5.0, k=2.0, theta=0.6),
        policy="weighted",
        calibrator=None,
    )
    decisions = op.process_batch(
        [_ev(0.1, event_id="a"), _ev(3.0, event_id="b"), _ev(20.0, event_id="c")]
    )
    for d in decisions:
        assert d.admitted
        assert d.update_weight == d.score


def test_calibrator_swaps_window_after_burn_in():
    cal = OnlineBurnInCalibrator(min_samples=20, low=0.1, high=0.9)
    op = RailOperator(
        params=AdmissionParams(tau_min=0.4, tau_max=30.0, k=1.2),
        policy="gated",
        calibrator=cal,
        adopt_calibrated_window=True,
    )
    # Feed deliberation values concentrated around 2.0-3.0.
    for i in range(40):
        op.process(_ev(2.0 + (i % 5) * 0.2, event_id=f"e{i}"))
    snap = op.snapshot()
    assert snap["window_swapped"] is True
    # New tau_max should have shrunk well below the original 30s ceiling.
    assert snap["params"]["tau_max"] < 10.0


def test_snapshot_restore_preserves_counts_and_params():
    op = RailOperator(policy="gated", calibrator=None)
    for i in range(10):
        op.process(_ev(2.0, event_id=str(i)))
    snap = op.snapshot()
    restored = RailOperator(calibrator=None)
    restored.restore(snap)
    assert restored.events_seen == 10
    assert restored.params.theta == op.params.theta
    assert restored.params.tau_min == op.params.tau_min
    assert restored.policy == op.policy


def test_audit_buffer_is_bounded():
    op = RailOperator(audit_buffer=3, calibrator=None)
    for i in range(10):
        op.process(_ev(2.0, event_id=str(i)))
    recent = op.recent_decisions()
    assert len(recent) == 3
    assert [d.event_id for d in recent] == ["7", "8", "9"]


def test_flink_recipe_returns_dict():
    op = RailOperator(calibrator=None)
    fn = flink_map_function(op)
    out = fn(_ev(2.0, event_id="x"))
    assert isinstance(out, dict)
    assert out["event_id"] == "x"
    assert "admitted" in out


def test_kafka_recipe_preserves_key():
    op = RailOperator(calibrator=None)
    handler = kafka_streams_handler(op)
    key, value = handler("k1", _ev(2.0, event_id="x"))
    assert key == "k1"
    assert isinstance(value, dict)


def test_synthesise_events_admission_rate_drops_on_contaminated_stream():
    events = synthesise_events(n=500, seed=42, contamination_rate=0.4)
    op = RailOperator(
        params=AdmissionParams(tau_min=1.0, tau_max=5.0, k=2.0, theta=0.6),
        policy="gated",
        calibrator=None,
    )
    decisions = op.process_batch([ev for ev, _ in events])
    admitted = [d.admitted for d in decisions]
    correct = [ok for _, ok in events]
    contract = contamination_contract(correct, admitted)
    # On this distribution the contaminated tail is far from the window,
    # so admitted-contamination should be well below the base rate.
    assert contract["admitted_contamination_rate"] < contract["base_contamination_rate"]


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        RailOperator(policy="bogus")
