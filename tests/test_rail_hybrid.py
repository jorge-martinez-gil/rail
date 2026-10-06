"""Tests for the RAIL-H hybrid admission policy."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.baselines_integrated import ITLMPolicy, make_integrated_policies
from experiments.rail_hybrid import RailHybridPolicy, vigilance_score
from experiments.rail_paper import PolicyConfig, Telemetry, rail_components


def _telemetry(delta: float, focus: float = 1.0, edits: int = 1) -> Telemetry:
    return Telemetry(
        anchor_time_s=0.0,
        decision_time_s=delta,
        focus_time_s=focus,
        edit_count=edits,
        num_features_shown=12,
    )


def _probs(p_correct: float, y: int = 0, n_classes: int = 4) -> np.ndarray:
    p = np.full(n_classes, (1.0 - p_correct) / (n_classes - 1))
    p[y] = p_correct
    return p


@pytest.fixture
def goldilocks_telemetry():
    return _telemetry(delta=3.0)  # well inside [tau_min=1.2, tau_max=5.0]


@pytest.fixture
def hasty_telemetry():
    return _telemetry(delta=0.1, focus=0.0, edits=0)  # far below tau_min


class TestVigilanceComponent:
    def test_vigilance_matches_rail_gated_score(self, goldilocks_telemetry):
        """RAIL-H's V must be bit-identical to the rail_gated policy score."""
        policy = RailHybridPolicy()
        cfg = PolicyConfig(name="rail_gated")
        expected = rail_components(goldilocks_telemetry, cfg)["score"]
        assert vigilance_score(goldilocks_telemetry, policy._params) == pytest.approx(expected)

    def test_defaults_pin_rail_gated_and_itlm(self):
        """No new tuned hyperparameters: defaults mirror the parents exactly."""
        policy = RailHybridPolicy()
        cfg = PolicyConfig(name="rail_gated")
        assert (policy.tau_min, policy.tau_max, policy.k, policy.theta) == (
            cfg.tau_min,
            cfg.tau_max,
            cfg.k,
            cfg.theta,
        )
        assert (policy.w_delta, policy.w_f, policy.w_e, policy.w_s) == (
            cfg.w_delta,
            cfg.w_f,
            cfg.w_e,
            cfg.w_s,
        )
        itlm = ITLMPolicy()
        assert policy.trim_alpha == itlm.alpha
        assert policy.trim_warmup == itlm.warmup


class TestConjunction:
    def test_admits_subset_of_both_parents(self):
        """On an identical stream, rail_h admissions = ITLM ∩ vigilance gate."""
        rng = np.random.default_rng(7)
        hybrid = RailHybridPolicy(trim_warmup=10)
        itlm = ITLMPolicy(warmup=10)
        telem_stream = []
        prob_stream = []
        for _ in range(400):
            delta = float(rng.uniform(0.05, 8.0))
            telem_stream.append(_telemetry(delta))
            prob_stream.append(_probs(float(rng.uniform(0.05, 0.95))))

        for probs, telem in zip(prob_stream, telem_stream, strict=True):
            v_ok = hybrid.vigilance(telem) >= hybrid.theta
            loss_ok_ref = itlm.admits(probs, 0, telem)
            admitted = hybrid.admits(probs, 0, telem)
            assert admitted == (v_ok and loss_ok_ref)

    def test_hasty_decision_rejected_despite_low_loss(self, hasty_telemetry):
        policy = RailHybridPolicy(trim_warmup=1)
        # Warm the loss gate with high losses so a confident label passes it.
        for _ in range(50):
            policy._loss_gate.decide(-5.0)
        assert not policy.admits(_probs(0.99), 0, hasty_telemetry)

    def test_high_loss_rejected_despite_goldilocks_timing(self, goldilocks_telemetry):
        policy = RailHybridPolicy(trim_warmup=1)
        # Warm the loss gate with low losses so a high-loss label fails it.
        for _ in range(200):
            policy._loss_gate.decide(-0.01)
        assert not policy.admits(_probs(0.01), 0, goldilocks_telemetry)

    def test_admits_when_both_gates_pass(self, goldilocks_telemetry):
        policy = RailHybridPolicy(trim_warmup=200)
        # During warmup the loss gate admits everything, so the decision
        # reduces to the vigilance gate.
        assert policy.admits(_probs(0.9), 0, goldilocks_telemetry)


class TestLossGateSeesAllEvents:
    def test_quantile_state_updates_on_vigilance_rejected_events(self, hasty_telemetry):
        """The loss gate must track the FULL stream (ITLM-comparable state)."""
        policy = RailHybridPolicy(trim_warmup=5)
        seen_before = policy._loss_gate._seen
        for _ in range(20):
            policy.admits(_probs(0.5), 0, hasty_telemetry)  # vigilance fails
        assert policy._loss_gate._seen == seen_before + 20

    def test_loss_state_matches_standalone_itlm(self):
        rng = np.random.default_rng(11)
        policy = RailHybridPolicy()
        itlm = ITLMPolicy()
        telem = SimpleNamespace(
            anchor_time_s=0.0,
            decision_time_s=0.01,  # vigilance always fails
            focus_time_s=0.0,
            edit_count=0,
            num_features_shown=12,
        )
        for _ in range(300):
            probs = _probs(float(rng.uniform(0.05, 0.95)))
            policy.admits(probs, 0, telem)
            itlm.admits(probs, 0, telem)
        assert policy._loss_gate._seen == itlm._gate._seen
        assert policy._loss_gate.current_threshold == pytest.approx(itlm._gate.current_threshold)


class TestFactoryIntegration:
    def test_factory_includes_rail_h(self):
        names = [p.name for p in make_integrated_policies(seed=0)]
        assert "rail_h" in names

    def test_fresh_state_per_factory_call(self):
        a = next(p for p in make_integrated_policies(0) if p.name == "rail_h")
        b = next(p for p in make_integrated_policies(0) if p.name == "rail_h")
        assert a._loss_gate is not b._loss_gate


class TestCalibratedVariant:
    def test_calibration_matches_rail_gated_yield_in_expectation(self):
        from experiments.rail_hybrid import RailHybridCalibratedPolicy

        rng = np.random.default_rng(3)
        events = [
            SimpleNamespace(telemetry=_telemetry(float(rng.uniform(0.05, 8.0))))
            for _ in range(2000)
        ]
        policy = RailHybridCalibratedPolicy()
        base = RailHybridPolicy()
        target = np.mean([base.vigilance(e.telemetry) >= base.theta for e in events])
        policy.calibrate_from_validation(events)
        # After calibration each gate passes ~sqrt(target); under independence
        # the conjunction admits ~target.
        v_pass = np.mean([policy.vigilance(e.telemetry) >= policy.theta for e in events])
        assert v_pass == pytest.approx(np.sqrt(target), abs=0.03)
        assert policy.trim_alpha == pytest.approx(np.clip(np.sqrt(target), 0.05, 0.99), abs=1e-9)

    def test_calibration_resets_loss_gate(self):
        from experiments.rail_hybrid import RailHybridCalibratedPolicy

        policy = RailHybridCalibratedPolicy()
        policy._loss_gate.decide(-1.0)
        events = [SimpleNamespace(telemetry=_telemetry(3.0)) for _ in range(50)]
        policy.calibrate_from_validation(events)
        assert policy._loss_gate._seen == 0

    def test_factory_includes_rail_h_cal(self):
        names = [p.name for p in make_integrated_policies(seed=0)]
        assert "rail_h_cal" in names
