"""Tests for the rail_core extensions (validation + online calibrator)."""

import math
import random

import pytest

from experiments.rail_core import (
    AdmissionParams,
    OnlineBurnInCalibrator,
    _P2Quantile,
    burn_in_window_from_samples,
)


def test_admission_params_rejects_invalid_window():
    with pytest.raises(ValueError):
        AdmissionParams(tau_min=5.0, tau_max=3.0)


def test_admission_params_rejects_negative_weights():
    with pytest.raises(ValueError):
        AdmissionParams(w_features=-0.1)
    with pytest.raises(ValueError):
        AdmissionParams(w_delta=0.0)


def test_admission_params_rejects_theta_out_of_range():
    with pytest.raises(ValueError):
        AdmissionParams(theta=1.5)
    with pytest.raises(ValueError):
        AdmissionParams(theta=-0.1)


def test_admission_params_accepts_paper_defaults():
    # Should not raise.
    AdmissionParams()
    AdmissionParams(tau_min=1.5, tau_max=4.8, k=2.8, theta=0.38)
    AdmissionParams(tau_min=0.9, tau_max=5.8, k=1.4, theta=0.60)


def test_p2_quantile_matches_offline_median():
    rng = random.Random(1)
    xs = [rng.gauss(0.0, 1.0) for _ in range(10_000)]
    est = _P2Quantile(0.5)
    for x in xs:
        est.update(x)
    truth = sorted(xs)[5_000]
    assert abs(est.value() - truth) < 0.1


def test_p2_quantile_handles_small_samples():
    est = _P2Quantile(0.5)
    for x in [3.0, 1.0, 2.0]:
        est.update(x)
    # Median of [1, 2, 3] is 2.
    assert est.value() == pytest.approx(2.0)


def test_online_calibrator_recovers_lognormal_window():
    rng = random.Random(7)
    xs = [rng.lognormvariate(0.9, 0.7) for _ in range(3_000)]
    cal = OnlineBurnInCalibrator(min_samples=300)
    cal.extend(xs)
    assert cal.is_ready()
    online_lo, online_hi = cal.window()
    offline_lo, offline_hi = burn_in_window_from_samples(xs)
    # P^2 is approximate but should be within reasonable tolerance.
    assert abs(online_lo - offline_lo) / max(offline_lo, 1e-6) < 0.25
    assert abs(online_hi - offline_hi) / max(offline_hi, 1e-6) < 0.25


def test_online_calibrator_materialises_admission_params():
    cal = OnlineBurnInCalibrator(min_samples=5)
    cal.extend([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    base = AdmissionParams(k=1.7, theta=0.62)
    p = cal.admission_params(base=base)
    assert isinstance(p, AdmissionParams)
    assert p.k == 1.7
    assert p.theta == 0.62
    lo, hi = cal.window()
    assert p.tau_min == lo
    assert p.tau_max == hi


def test_online_calibrator_ignores_non_finite_and_negative_inputs():
    cal = OnlineBurnInCalibrator(min_samples=3)
    cal.update(float("nan"))
    cal.update(-1.0)
    cal.update(2.0)
    cal.update(3.0)
    cal.update(4.0)
    assert cal.count == 3
    assert cal.is_ready()


def test_burn_in_window_from_samples_edge_cases():
    # Empty -> defaults.
    lo, hi = burn_in_window_from_samples([])
    assert lo < hi
    # All-NaN -> defaults.
    lo, hi = burn_in_window_from_samples([math.nan, math.inf])
    assert lo < hi
