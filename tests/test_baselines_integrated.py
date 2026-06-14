"""Tests for the integrated noise-robust baselines."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.baselines_integrated import (
    INTEGRATED_POLICY_NAMES,
    CoTeachingPolicy,
    DynamicQuantilePolicy,
    GCEWeightPolicy,
    ITLMPolicy,
    JointAgreementPolicy,
    SCEWeightPolicy,
    SelfPacedPolicy,
    StatefulPolicy,
    make_integrated_policies,
)
from experiments.replay_integrated import replay_once_integrated


class _DummyTelemetry:
    """Telemetry placeholder; integrated policies do not consult it."""


@pytest.fixture
def telemetry():
    return _DummyTelemetry()


@pytest.fixture
def confident_correct_probs():
    # 5-class predictor; correct class predicted with p=0.9.
    return np.array([0.025, 0.025, 0.9, 0.025, 0.025])


@pytest.fixture
def confident_wrong_probs():
    # 5-class predictor very confident in the wrong class.
    return np.array([0.025, 0.025, 0.025, 0.9, 0.025])


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------


def test_make_integrated_policies_has_all_names():
    policies = make_integrated_policies(seed=0)
    names = {p.name for p in policies}
    assert names == set(INTEGRATED_POLICY_NAMES)


def test_make_integrated_policies_unique_names():
    policies = make_integrated_policies(seed=0)
    names = [p.name for p in policies]
    assert len(names) == len(set(names)), "Integrated policy names must be unique"


# ---------------------------------------------------------------------------
# GCE weighting -- confident-wrong samples get small weight.
# ---------------------------------------------------------------------------


def test_gce_weight_small_for_confident_wrong(
    confident_correct_probs, confident_wrong_probs, telemetry
):
    """Confident-but-wrong labels must receive a smaller GCE weight."""
    gce = GCEWeightPolicy(q=0.7)
    w_correct = gce.weight(confident_correct_probs, y_human=2, telemetry=telemetry)
    w_wrong = gce.weight(confident_wrong_probs, y_human=2, telemetry=telemetry)
    assert 0.0 < w_wrong < w_correct
    # GCE always admits, weight is the robust mechanism.
    assert gce.admits(confident_correct_probs, 2, telemetry)
    assert gce.admits(confident_wrong_probs, 2, telemetry)


def test_gce_weight_recovers_mae_at_q_one():
    """At q -> 1, GCE weight should approach p_y (the MAE gradient ratio)."""
    gce = GCEWeightPolicy(q=1.0)
    probs = np.array([0.1, 0.2, 0.7])
    w = gce.weight(probs, y_human=2, telemetry=None)
    assert w == pytest.approx(0.7, rel=1e-6)


# ---------------------------------------------------------------------------
# SCE weighting -- bounded above; downweights confident-wrong samples.
# ---------------------------------------------------------------------------


def test_sce_weight_bounded(confident_correct_probs, telemetry):
    sce = SCEWeightPolicy(alpha=0.1, beta=1.0, clip_log=4.0)
    w = sce.weight(confident_correct_probs, y_human=2, telemetry=telemetry)
    # Upper bound: alpha + beta * clip_log * 1 = 0.1 + 4.0 = 4.1.
    assert 0.0 < w < 4.1


def test_sce_weight_decreases_for_confident_wrong(
    confident_correct_probs, confident_wrong_probs, telemetry
):
    sce = SCEWeightPolicy(alpha=0.1, beta=1.0)
    w_correct = sce.weight(confident_correct_probs, y_human=2, telemetry=telemetry)
    w_wrong = sce.weight(confident_wrong_probs, y_human=2, telemetry=telemetry)
    assert w_wrong < w_correct


# ---------------------------------------------------------------------------
# Self-paced -- curriculum lambda grows over time.
# ---------------------------------------------------------------------------


def test_self_paced_admits_more_over_time(confident_wrong_probs, telemetry):
    """A label that is initially too-loss-y should eventually be admitted."""
    sp = SelfPacedPolicy(lambda_0=0.5, growth=1.5, step=5, cap=50.0)
    early = [sp.admits(confident_wrong_probs, 2, telemetry) for _ in range(3)]
    for _ in range(200):
        sp.admits(confident_wrong_probs, 2, telemetry)
    later = [sp.admits(confident_wrong_probs, 2, telemetry) for _ in range(3)]
    # Curriculum should have grown by the time we hit the cap.
    assert any(later) and not all(early), (
        "Self-paced gate should become more permissive once curriculum grows"
    )


# ---------------------------------------------------------------------------
# Dynamic quantile -- admits ~upper fraction of recent scores after warmup.
# ---------------------------------------------------------------------------


def test_dynamic_quantile_calibrates_to_upper_fraction(telemetry):
    """Across 1000 i.i.d. scores, admission rate should approach ``upper``."""
    rng = np.random.default_rng(0)
    upper = 0.6
    dq = DynamicQuantilePolicy(
        upper=upper,
        warmup=50,
        score_fn=lambda probs, y, t: float(probs[0]),
    )
    admits = 0
    n = 1000
    for _ in range(n):
        # Synthesize an arbitrary 3-class probability vector.
        s = rng.beta(2, 2)
        probs = np.array([s, (1 - s) / 2, (1 - s) / 2])
        if dq.admits(probs, y_human=0, telemetry=telemetry):
            admits += 1
    rate = admits / n
    # Generous tolerance: P^2 quantile and warmup distort the empirical rate.
    assert abs(rate - upper) < 0.10


# ---------------------------------------------------------------------------
# ITLM -- admits roughly the smallest-loss alpha fraction.
# ---------------------------------------------------------------------------


def test_itlm_admits_low_loss_after_warmup(telemetry):
    rng = np.random.default_rng(0)
    itlm = ITLMPolicy(alpha=0.7, warmup=50)
    # Warm up with mixed losses.
    for _ in range(200):
        p_correct = float(rng.uniform(0.05, 0.95))
        probs = np.array([1 - p_correct, p_correct])
        itlm.admits(probs, y_human=1, telemetry=telemetry)
    # High-prob correct sample: should admit.
    admit_easy = itlm.admits(np.array([0.05, 0.95]), 1, telemetry)
    # Low-prob correct sample (high loss): typically should not admit.
    admit_hard = itlm.admits(np.array([0.99, 0.01]), 1, telemetry)
    assert admit_easy is True
    assert admit_hard is False


def test_replay_drives_stateful_gate_once_per_event(telemetry):
    """Default gate weights must reuse the admission decision, not mutate twice."""

    class CountingGate(StatefulPolicy):
        def __init__(self):
            super().__init__(name="counting_gate")
            self.calls = 0

        def score(self, probs, y_human, telemetry):
            return 1.0

        def admits(self, probs, y_human, telemetry):
            self.calls += 1
            return self.calls % 2 == 1

    class DummyModel:
        def __init__(self):
            self.updates = 0

        def clone(self):
            return self

        def predict_proba(self, x):
            return np.array([0.1, 0.9])

        def update(self, x, y, sample_weight=1.0):
            self.updates += 1

        def evaluate_macro_f1(self, X_test, y_test):
            return 0.0

    events = [
        SimpleNamespace(
            x=np.array([float(i)]),
            y_human=1,
            human_feedback_is_correct=True,
            telemetry=telemetry,
        )
        for i in range(5)
    ]
    policy = CountingGate()
    model = DummyModel()

    row = replay_once_integrated(
        dataset_name="dummy",
        base_model=model,
        events=events,
        X_test=np.zeros((1, 1)),
        y_test=np.array([1]),
        policy=policy,
        run_id=0,
        always_reference_counts=(0, len(events)),
    )

    assert policy.calls == len(events)
    assert row.admitted_feedback == 3
    assert model.updates == 3


# ---------------------------------------------------------------------------
# CoTeaching / JointAgreement -- decisions stable across enough events.
# ---------------------------------------------------------------------------


def test_co_teaching_decisions_settle(telemetry):
    rng = np.random.default_rng(123)
    ct = CoTeachingPolicy(forget_rate=0.3, ramp_steps=100, window=200, seed=42)
    admits = []
    for _ in range(500):
        p = float(rng.uniform(0.3, 0.95))
        probs = np.array([1 - p, p])
        admits.append(ct.admits(probs, y_human=1, telemetry=telemetry))
    # After ramp-up, expected admit rate should fall to roughly (1 - 0.3) = 0.7.
    tail_rate = sum(admits[-200:]) / 200
    assert 0.5 < tail_rate < 0.9


def test_joint_agreement_decisions_change_under_random_state(telemetry):
    """Joint-agreement gate decisions should depend on RNG state."""
    rng = np.random.default_rng(0)
    ja_a = JointAgreementPolicy(keep_fraction=0.6, seed=1)
    ja_b = JointAgreementPolicy(keep_fraction=0.6, seed=2)
    diffs = 0
    for _ in range(200):
        p = float(rng.uniform(0.3, 0.9))
        probs = np.array([1 - p, p])
        if ja_a.admits(probs, 1, telemetry) != ja_b.admits(probs, 1, telemetry):
            diffs += 1
    assert diffs > 0
