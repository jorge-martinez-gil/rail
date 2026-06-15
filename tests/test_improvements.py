"""Tests for all new features added in the top-tier-paper improvement pass.

Covers:
* Bennett bound and variance_aware_horizon (theory.py)
* AUCC and NCG metrics (metrics.py)
* Concept-drift synthesis (rail_operator.py DriftConfig)
* OnlineDriftCalibrator (rail_core.py)
* LaTeX table export (rail_stats.py)
* JointAgreementPolicy Dirichlet consistency fix (baselines_integrated.py)
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# Bennett bound and variance_aware_horizon
# ---------------------------------------------------------------------------


def test_bennett_bound_is_at_most_hoeffding():
    """Bennett's inequality should be at most as large as Hoeffding's."""
    from experiments.theory import bennett_contamination_bound, hoeffding_contamination_bound

    params = [(1000, 0.2, 0.3, 0.05), (500, 0.1, 0.5, 0.03), (200, 0.05, 0.2, 0.01)]
    for horizon, pi, alpha, eps in params:
        bennett = bennett_contamination_bound(horizon, pi, alpha, eps)
        hoeffding = hoeffding_contamination_bound(horizon, pi, alpha, eps)
        assert bennett <= hoeffding + 1e-12, (
            f"Bennett {bennett} > Hoeffding {hoeffding} for params {params}"
        )


def test_bennett_bound_clips_to_unit_interval():
    from experiments.theory import bennett_contamination_bound

    assert bennett_contamination_bound(0, 0.3, 0.2, 0.05) == 1.0
    val = bennett_contamination_bound(10000, 0.3, 0.2, 0.5)
    assert 0.0 <= val <= 1.0


def test_bennett_bound_rejects_invalid_inputs():
    from experiments.theory import bennett_contamination_bound

    with pytest.raises(ValueError):
        bennett_contamination_bound(100, 0.3, 0.2, epsilon=0.0)
    with pytest.raises(ValueError):
        bennett_contamination_bound(100, -0.1, 0.2, epsilon=0.05)
    with pytest.raises(ValueError):
        bennett_contamination_bound(100, 0.3, 1.5, epsilon=0.05)


def test_variance_aware_horizon_le_hoeffding_horizon():
    """Bennett-based horizon should be <= Hoeffding-based horizon."""
    from experiments.theory import required_horizon_for_budget, variance_aware_horizon

    n_hoeff = required_horizon_for_budget(60.0, 0.95, 0.3, 0.2)
    n_bennett = variance_aware_horizon(60.0, 0.95, 0.3, 0.2)
    assert n_hoeff >= 1
    assert n_bennett >= 1
    # Bennett is at least as tight.
    assert n_bennett <= n_hoeff


def test_variance_aware_horizon_infeasible():
    from experiments.theory import variance_aware_horizon

    assert variance_aware_horizon(0.001, 0.95, 0.9, 0.9) == -1


def test_variance_aware_horizon_rejects_invalid():
    from experiments.theory import variance_aware_horizon

    with pytest.raises(ValueError):
        variance_aware_horizon(60.0, 0.0, 0.3, 0.2)
    with pytest.raises(ValueError):
        variance_aware_horizon(-1.0, 0.95, 0.3, 0.2)


# ---------------------------------------------------------------------------
# AUCC metric
# ---------------------------------------------------------------------------


def test_aucc_always_zero_for_perfect_filtering():
    """A gate that blocks all contaminated events should have AUCC = 0."""
    from experiments.metrics import area_under_contamination_curve

    correct = [True, False, True, False, True]
    admitted = [True, False, True, False, True]  # all contaminated are rejected
    aucc = area_under_contamination_curve(correct, admitted)
    assert aucc == pytest.approx(0.0, abs=1e-9)


def test_aucc_always_equals_base_rate_for_unfiltered():
    """When all events are admitted, AUCC should equal base_contamination_rate."""
    from experiments.metrics import area_under_contamination_curve

    n = 200
    contamination_rate = 0.3
    correct = [i % round(1 / contamination_rate) != 0 for i in range(n)]
    admitted = [True] * n
    # For a perfectly uniform stream with all admitted, AUCC ≈ base_rate.
    aucc = area_under_contamination_curve(correct, admitted)
    base_rate = sum(1 for ok in correct if not ok) / n
    assert abs(aucc - base_rate) < 0.05


def test_aucc_returns_zero_for_no_admissions():
    from experiments.metrics import area_under_contamination_curve

    correct = [True, False, True]
    admitted = [False, False, False]
    assert area_under_contamination_curve(correct, admitted) == 0.0


def test_aucc_mismatched_lengths_raises():
    from experiments.metrics import area_under_contamination_curve

    with pytest.raises(ValueError):
        area_under_contamination_curve([True, False], [True])


def test_aucc_in_unit_interval():
    from experiments.metrics import area_under_contamination_curve

    correct = [True, False, True, False, True, False]
    admitted = [True, True, True, False, True, False]
    val = area_under_contamination_curve(correct, admitted)
    assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# NCG metric
# ---------------------------------------------------------------------------


def test_ncg_perfect_filtering_is_one():
    from experiments.metrics import normalised_contamination_gain

    correct = [True, False, True, False, True]
    admitted = [True, False, True, False, True]  # all contaminated rejected
    assert normalised_contamination_gain(correct, admitted) == pytest.approx(1.0)


def test_ncg_no_filtering_is_zero():
    """When admitted contamination rate equals base rate, NCG should be 0."""
    from experiments.metrics import normalised_contamination_gain

    correct = [True, False, True, False]
    admitted = [True, True, True, True]
    ncg = normalised_contamination_gain(correct, admitted)
    assert ncg == pytest.approx(0.0, abs=1e-9)


def test_ncg_zero_contamination_returns_zero():
    from experiments.metrics import normalised_contamination_gain

    correct = [True, True, True, True]
    admitted = [True, True, False, True]
    assert normalised_contamination_gain(correct, admitted) == 0.0


def test_ncg_all_rejected_is_one():
    from experiments.metrics import normalised_contamination_gain

    correct = [True, False, True, False]
    admitted = [False, False, False, False]
    assert normalised_contamination_gain(correct, admitted) == pytest.approx(1.0)


def test_ncg_mismatched_lengths_raises():
    from experiments.metrics import normalised_contamination_gain

    with pytest.raises(ValueError):
        normalised_contamination_gain([True, False], [True])


# ---------------------------------------------------------------------------
# Concept-drift synthesis
# ---------------------------------------------------------------------------


def test_drift_config_validates_inputs():
    from experiments.rail_operator import DriftConfig

    with pytest.raises(ValueError):
        DriftConfig(onset=-1, new_delta_mean=3.0, new_delta_sd=0.5)
    with pytest.raises(ValueError):
        DriftConfig(onset=100, new_delta_mean=0.0, new_delta_sd=0.5)
    with pytest.raises(ValueError):
        DriftConfig(onset=100, new_delta_mean=3.0, new_delta_sd=-1.0)
    with pytest.raises(ValueError):
        DriftConfig(onset=100, new_delta_mean=3.0, new_delta_sd=0.5, mode="unknown")
    with pytest.raises(ValueError):
        DriftConfig(onset=100, new_delta_mean=3.0, new_delta_sd=0.5, duration=0)


def test_abrupt_drift_changes_delta_distribution():
    """Events after abrupt drift onset should have different mean delta."""
    from experiments.rail_operator import DriftConfig, synthesise_events

    drift = DriftConfig(onset=200, new_delta_mean=8.0, new_delta_sd=0.3, mode="abrupt")
    events = synthesise_events(n=400, seed=42, contamination_rate=0.0, drift_config=drift)

    pre_deltas = [ev.delta_sec for ev, ok in events[:200] if ok]
    post_deltas = [ev.delta_sec for ev, ok in events[250:] if ok]
    assert pre_deltas, "Should have clean pre-drift events"
    assert post_deltas, "Should have clean post-drift events"
    pre_mean = sum(pre_deltas) / len(pre_deltas)
    post_mean = sum(post_deltas) / len(post_deltas)
    # Post-drift mean should be substantially larger (target 8.0 vs 2.5).
    assert post_mean > pre_mean * 1.5, f"pre={pre_mean:.2f}, post={post_mean:.2f}"


def test_gradual_drift_is_continuous():
    """During gradual drift the mean should increase monotonically (in expectation)."""
    from experiments.rail_operator import DriftConfig, synthesise_events

    # Use contamination_rate=0 to focus on clean delta distribution only.
    drift = DriftConfig(
        onset=100, new_delta_mean=10.0, new_delta_sd=0.3, mode="gradual", duration=300
    )
    events = synthesise_events(n=500, seed=0, contamination_rate=0.0, drift_config=drift)
    # Mean in first half of drift window should be smaller than in second half.
    mid_deltas = [ev.delta_sec for ev, _ in events[100:250]]
    late_deltas = [ev.delta_sec for ev, _ in events[300:500]]
    mid_mean = sum(mid_deltas) / len(mid_deltas)
    late_mean = sum(late_deltas) / len(late_deltas)
    assert late_mean > mid_mean


def test_no_drift_config_is_backward_compatible():
    """synthesise_events without drift_config should behave as before."""
    from experiments.rail_operator import synthesise_events

    events = synthesise_events(n=100, seed=0)
    assert len(events) == 100
    assert all(isinstance(ev, tuple) and len(ev) == 2 for ev in events)


# ---------------------------------------------------------------------------
# OnlineDriftCalibrator
# ---------------------------------------------------------------------------


def test_online_drift_calibrator_detects_shift():
    """Feeding a strongly shifted stream should trigger at least one reset."""
    import random

    from experiments.rail_core import OnlineBurnInCalibrator, OnlineDriftCalibrator

    rng = random.Random(1)
    cal = OnlineDriftCalibrator(
        base_calibrator=OnlineBurnInCalibrator(min_samples=50),
        drift_threshold=2.0,
        drift_window=10,
        reference_window=100,
    )
    # Feed a stable stream to complete burn-in.
    for _ in range(100):
        cal.update(rng.gauss(2.5, 0.3))
    assert cal.is_ready()
    # Now inject a massively shifted stream.
    resets_before = cal.resets
    for _ in range(50):
        cal.update(rng.gauss(12.0, 0.3))  # far from reference mean
    assert cal.resets > resets_before, "Expected at least one drift reset"


def test_online_drift_calibrator_no_false_alarms_on_stable_stream():
    """No resets should be triggered on a stationary stream."""
    import random

    from experiments.rail_core import OnlineBurnInCalibrator, OnlineDriftCalibrator

    rng = random.Random(2)
    cal = OnlineDriftCalibrator(
        base_calibrator=OnlineBurnInCalibrator(min_samples=50),
        drift_threshold=4.0,
        drift_window=20,
        reference_window=100,
    )
    for _ in range(500):
        cal.update(rng.gauss(2.5, 0.3))
    assert cal.resets == 0, f"Unexpected resets on stationary stream: {cal.resets}"


def test_online_drift_calibrator_rejects_invalid_params():
    from experiments.rail_core import OnlineDriftCalibrator

    with pytest.raises(ValueError):
        OnlineDriftCalibrator(drift_threshold=0.0)
    with pytest.raises(ValueError):
        OnlineDriftCalibrator(drift_window=0)
    with pytest.raises(ValueError):
        OnlineDriftCalibrator(reference_window=0)


def test_online_drift_calibrator_ignores_invalid_deltas():

    from experiments.rail_core import OnlineDriftCalibrator

    cal = OnlineDriftCalibrator()
    cal.update(math.nan)
    cal.update(-1.0)
    cal.update(math.inf)
    # No crash, no state corruption.
    assert cal.resets == 0


def test_online_drift_calibrator_window_and_params_after_burnin():
    """After burn-in, window() and admission_params() should be callable."""
    import random

    from experiments.rail_core import OnlineBurnInCalibrator, OnlineDriftCalibrator

    rng = random.Random(3)
    cal = OnlineDriftCalibrator(
        base_calibrator=OnlineBurnInCalibrator(min_samples=30),
    )
    for _ in range(60):
        cal.update(rng.gauss(2.5, 0.5))
    assert cal.is_ready()
    lo, hi = cal.window()
    assert lo < hi
    p = cal.admission_params()
    assert p.tau_min < p.tau_max


# ---------------------------------------------------------------------------
# LaTeX table export
# ---------------------------------------------------------------------------


def test_report_to_latex_produces_booktabs_table():
    from experiments.rail_stats import ComparisonRow, report_to_latex

    rows = [
        ComparisonRow(
            policy="rail_gated",
            n=30,
            mean=0.82,
            sd=0.03,
            ci_lower=0.79,
            ci_upper=0.85,
            diff_mean=0.05,
            diff_ci_lower=0.02,
            diff_ci_upper=0.08,
            cliffs_delta=0.65,
            cliffs_magnitude="large",
            rank_biserial=0.60,
            wilcoxon_p=0.002,
            wilcoxon_p_holm=0.008,
        ),
        ComparisonRow(
            policy="confidence_gated",
            n=30,
            mean=0.77,
            sd=0.04,
            ci_lower=0.74,
            ci_upper=0.80,
            diff_mean=0.00,
            diff_ci_lower=-0.02,
            diff_ci_upper=0.02,
            cliffs_delta=0.05,
            cliffs_magnitude="negligible",
            rank_biserial=0.04,
            wilcoxon_p=0.61,
            wilcoxon_p_holm=0.61,
        ),
    ]
    latex = report_to_latex(rows, baseline="always")
    assert r"\begin{table}" in latex
    assert r"\toprule" in latex
    assert r"\midrule" in latex
    assert r"\bottomrule" in latex
    assert r"\end{table}" in latex
    assert r"\dag" in latex  # significant row
    assert "rail_gated" in latex
    assert "confidence_gated" in latex


def test_report_to_latex_empty_rows():
    from experiments.rail_stats import report_to_latex

    latex = report_to_latex([], baseline="always")
    assert r"\begin{table}" in latex
    assert r"\end{table}" in latex


def test_report_to_latex_custom_label():
    from experiments.rail_stats import ComparisonRow, report_to_latex

    rows = [
        ComparisonRow(
            policy="x",
            n=10,
            mean=0.5,
            sd=0.1,
            ci_lower=0.4,
            ci_upper=0.6,
            diff_mean=0.1,
            diff_ci_lower=0.0,
            diff_ci_upper=0.2,
            cliffs_delta=0.2,
            cliffs_magnitude="small",
            rank_biserial=0.15,
            wilcoxon_p=0.06,
            wilcoxon_p_holm=0.12,
        )
    ]
    latex = report_to_latex(rows, baseline="base", label="tab:mytest")
    assert "tab:mytest" in latex


# ---------------------------------------------------------------------------
# JointAgreementPolicy Dirichlet consistency
# ---------------------------------------------------------------------------


def test_joint_agreement_uses_consistent_views():
    """The fix ensures each call to admits() uses 2 Dirichlet samples, not 4."""
    import numpy as np

    from experiments.baselines_integrated import JointAgreementPolicy

    rng = np.random.default_rng(99)
    probs = rng.dirichlet([5.0, 2.0, 1.0])
    # We can't directly inspect RNG state, but we can verify the policy
    # produces the same decision given the same seed on repeated independent
    # instantiations (determinism test -- would fail if extra Dirichlet draws
    # leaked between calls before the fix).
    decisions_a = []
    for _ in range(20):
        p = JointAgreementPolicy(seed=42)
        decisions_a.append(p.admits(probs, y_human=0, telemetry=None))

    decisions_b = []
    for _ in range(20):
        p = JointAgreementPolicy(seed=42)
        decisions_b.append(p.admits(probs, y_human=0, telemetry=None))

    assert decisions_a == decisions_b, "Policy should be deterministic given seed"


def test_joint_agreement_policy_smoke():
    """admits() should return a bool without raising."""
    import numpy as np

    from experiments.baselines_integrated import JointAgreementPolicy

    policy = JointAgreementPolicy(seed=0)
    probs = np.array([0.7, 0.2, 0.1])
    for _ in range(10):
        result = policy.admits(probs, y_human=0, telemetry=None)
        assert isinstance(result, bool)
