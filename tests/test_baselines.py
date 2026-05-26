"""Tests for the new noise-robust baselines."""

import random

import pytest

from experiments.baselines import (
    CoTeachingGate,
    DynamicQuantileGate,
    JointAgreementGate,
    SelfPacedGate,
)


def test_self_paced_gate_admits_small_losses():
    gate = SelfPacedGate(lambda_0=1.0, growth=1.0, step=1)
    assert gate.decide(0.2) is True
    assert gate.decide(2.0) is False


def test_self_paced_gate_relaxes_over_time():
    gate = SelfPacedGate(lambda_0=0.5, growth=2.0, step=1)
    # After many epochs lambda exceeds the loss.
    for _ in range(10):
        gate.decide(0.1)
    assert gate.current_lambda() > 0.5
    assert gate.decide(5.0) is True


def test_co_teaching_rejects_top_loss_percentile():
    gate = CoTeachingGate(forget_rate=0.3, ramp_steps=0, window=100)
    rng = random.Random(0)
    losses = [(rng.random(), rng.random()) for _ in range(200)]
    decisions = [gate.decide(a, b) for a, b in losses]
    # After warmup the gate should reject roughly 30% of events.
    keep_rate = sum(decisions[-100:]) / 100
    assert 0.4 <= keep_rate <= 0.8  # 30% rejection per side but joint AND reduces it


def test_joint_agreement_requires_agreement_to_admit():
    gate = JointAgreementGate(keep_fraction=0.8, window=10)
    # Predictions disagree -> always reject.
    for _ in range(5):
        assert gate.decide(0, 1, 0.1, 0.1) is False


def test_joint_agreement_admits_on_low_combined_loss():
    gate = JointAgreementGate(keep_fraction=0.8, window=20)
    # Build history of high losses, then test a low one.
    for _ in range(20):
        gate.decide(0, 0, 1.0, 1.0)
    assert gate.decide(0, 0, 0.05, 0.05) is True


def test_dynamic_quantile_gate_admits_top_fraction():
    gate = DynamicQuantileGate(upper=0.5, warmup=10)
    rng = random.Random(0)
    scores = [rng.random() for _ in range(500)]
    decisions = [gate.decide(s) for s in scores]
    keep_rate = sum(decisions[-200:]) / 200
    assert 0.3 <= keep_rate <= 0.7


def test_validation_errors():
    with pytest.raises(ValueError):
        SelfPacedGate(lambda_0=-1.0)
    with pytest.raises(ValueError):
        SelfPacedGate(growth=0.0)
    with pytest.raises(ValueError):
        SelfPacedGate(step=0)
    with pytest.raises(ValueError):
        CoTeachingGate(forget_rate=1.0)
    with pytest.raises(ValueError):
        JointAgreementGate(keep_fraction=0.0)
    with pytest.raises(ValueError):
        DynamicQuantileGate(upper=0.0)
