"""Tests for the publication-grade statistics module."""

import random

import pytest

from experiments.rail_stats import (
    _phi,
    _phi_inv,
    bootstrap_ci,
    cliffs_delta,
    cliffs_delta_magnitude,
    compute_statistical_report,
    holm_bonferroni,
    paired_difference_ci,
    paired_wilcoxon,
    rank_biserial,
    report_to_markdown,
)


def test_phi_inv_is_inverse_of_phi():
    for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
        assert abs(_phi(_phi_inv(p)) - p) < 1e-6


def test_bootstrap_percentile_contains_true_mean():
    rng = random.Random(0)
    vals = [rng.gauss(0.5, 0.2) for _ in range(200)]
    ci = bootstrap_ci(vals, statistic="mean", iterations=2000, seed=0)
    assert ci.lower <= ci.point <= ci.upper
    assert ci.lower <= 0.5 <= ci.upper


def test_bootstrap_bca_runs_and_returns_valid_interval():
    rng = random.Random(1)
    vals = [rng.lognormvariate(0.0, 0.5) for _ in range(100)]
    ci = bootstrap_ci(vals, iterations=1000, method="bca", seed=2)
    assert ci.method == "bca"
    assert ci.lower <= ci.point <= ci.upper


def test_paired_difference_ci_zero_when_inputs_identical():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = paired_difference_ci(vals, vals, iterations=500, seed=0)
    assert ci.point == pytest.approx(0.0)
    assert ci.lower <= 0.0 <= ci.upper


def test_cliffs_delta_extremes():
    a = [1, 2, 3]
    b = [4, 5, 6]
    assert cliffs_delta(a, b) == -1.0
    assert cliffs_delta(b, a) == 1.0
    assert cliffs_delta(a, a) == 0.0


def test_cliffs_magnitude_bands():
    assert cliffs_delta_magnitude(0.1) == "negligible"
    assert cliffs_delta_magnitude(0.2) == "small"
    assert cliffs_delta_magnitude(0.4) == "medium"
    assert cliffs_delta_magnitude(0.6) == "large"


def test_paired_wilcoxon_detects_systematic_shift():
    rng = random.Random(0)
    a = [rng.gauss(0.0, 1.0) for _ in range(40)]
    b = [x + 1.0 for x in a]
    out = paired_wilcoxon(a, b)
    assert out["p_value"] < 0.001
    assert out["r_rb"] < -0.9  # a is consistently smaller than b


def test_paired_wilcoxon_no_diff_returns_p_one():
    out = paired_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert out["p_value"] == 1.0
    assert out["W"] == 0
    assert out["n_nonzero"] == 0


def test_rank_biserial_extremes():
    a = [1.0, 2.0, 3.0]
    b = [2.0, 3.0, 4.0]  # a < b always
    assert rank_biserial(a, b) == -1.0


def test_holm_bonferroni_monotone_and_clamped():
    ps = [0.01, 0.04, 0.03, 0.20]
    adj = holm_bonferroni(ps)
    assert all(0.0 <= a <= 1.0 for a in adj)
    # Holm: smallest p multiplied by m=4, next by 3, etc.
    sorted_pairs = sorted(zip(ps, adj))
    sorted_adj = [a for _, a in sorted_pairs]
    assert all(sorted_adj[i] <= sorted_adj[i + 1] for i in range(len(sorted_adj) - 1))


def test_compute_statistical_report_full_flow():
    rng = random.Random(42)
    baseline = [rng.gauss(0.40, 0.05) for _ in range(20)]
    better = [x + 0.10 for x in baseline]
    similar = [x + 0.005 for x in baseline]
    rows = compute_statistical_report(
        {"baseline": baseline, "better": better, "similar": similar},
        baseline="baseline",
        iterations=500,
        seed=0,
    )
    rows_by_name = {r.policy: r for r in rows}
    assert rows_by_name["better"].wilcoxon_p < 0.01
    assert rows_by_name["better"].cliffs_delta > 0.7
    md = report_to_markdown(rows, baseline="baseline")
    assert "better" in md and "similar" in md


def test_compute_statistical_report_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_statistical_report(
            {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0]}, baseline="a", iterations=100
        )
