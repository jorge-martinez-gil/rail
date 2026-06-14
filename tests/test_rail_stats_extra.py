"""Tests for the multi-dataset statistical pipeline (Friedman + Nemenyi)."""

from __future__ import annotations

import pytest

from experiments.rail_stats_extra import (
    average_ranks,
    critical_difference_value,
    friedman_test,
    multi_dataset_report,
    nemenyi_posthoc,
)

# ---------------------------------------------------------------------------
# Average ranks.
# ---------------------------------------------------------------------------


def test_average_ranks_basic():
    """A method that strictly dominates on every dataset has rank 1.0."""
    scores = {
        "ds1": {"rail": [0.95], "loss": [0.80], "conf": [0.60]},
        "ds2": {"rail": [0.92], "loss": [0.81], "conf": [0.55]},
        "ds3": {"rail": [0.90], "loss": [0.79], "conf": [0.50]},
    }
    ranks = average_ranks(scores, higher_is_better=True)
    assert ranks["rail"] == pytest.approx(1.0)
    assert ranks["loss"] == pytest.approx(2.0)
    assert ranks["conf"] == pytest.approx(3.0)


def test_average_ranks_handles_ties():
    """Equal means should yield the average of the tied ranks."""
    scores = {
        "ds1": {"a": [0.5], "b": [0.5], "c": [0.1]},
    }
    ranks = average_ranks(scores)
    # a and b tied for ranks 1 and 2 -> average 1.5; c is rank 3.
    assert ranks["a"] == pytest.approx(1.5)
    assert ranks["b"] == pytest.approx(1.5)
    assert ranks["c"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Friedman test.
# ---------------------------------------------------------------------------


def test_friedman_rejects_when_clear_separation():
    """A clear winner across many datasets should yield a low p-value."""
    scores = {
        f"ds{i}": {
            "winner": [0.90 + 0.001 * i],
            "middle": [0.70 + 0.001 * i],
            "loser": [0.50 + 0.001 * i],
        }
        for i in range(8)
    }
    res = friedman_test(scores, higher_is_better=True)
    assert res.p_value < 0.05
    assert res.n_datasets == 8
    assert res.n_methods == 3
    # 'winner' must have the smallest average rank.
    assert min(res.average_ranks, key=res.average_ranks.get) == "winner"


def test_friedman_does_not_reject_when_methods_tied():
    scores = {f"ds{i}": {"a": [0.5], "b": [0.5], "c": [0.5]} for i in range(5)}
    res = friedman_test(scores)
    # All ties -> chi2 = 0 -> p = 1.0.
    assert res.p_value >= 0.999


# ---------------------------------------------------------------------------
# Nemenyi critical difference.
# ---------------------------------------------------------------------------


def test_critical_difference_decreases_with_more_datasets():
    cd_few = critical_difference_value(k=5, n_datasets=5, alpha=0.05)
    cd_many = critical_difference_value(k=5, n_datasets=50, alpha=0.05)
    assert cd_many < cd_few


def test_critical_difference_increases_with_more_methods():
    cd_few = critical_difference_value(k=3, n_datasets=10, alpha=0.05)
    cd_many = critical_difference_value(k=10, n_datasets=10, alpha=0.05)
    assert cd_many > cd_few


def test_nemenyi_marks_clear_winner_significant():
    # Alternate the two losers so their average rank across datasets is the
    # same -- otherwise even a tiny per-dataset ordering produces a 1.0
    # average-rank gap, which is wider than the tabulated CD for k=3, N=12.
    scores = {}
    for i in range(12):
        if i % 2 == 0:
            a, b = 0.30, 0.28
        else:
            a, b = 0.28, 0.30
        scores[f"ds{i}"] = {
            "winner": [0.95 + 0.001 * i],
            "loser_a": [a + 0.001 * i],
            "loser_b": [b + 0.001 * i],
        }
    res = nemenyi_posthoc(scores, alpha=0.05)
    # Pull values out before asserting -- pytest's assertion rewriter has a
    # known quirk with chained dict-of-dict subscripts inside ``assert not``.
    sig = res.significant
    winner_vs_a = bool(sig["winner"]["loser_a"])
    winner_vs_b = bool(sig["winner"]["loser_b"])
    losers_diff = bool(sig["loser_a"]["loser_b"])
    # Winner vs each loser should be significantly different at the
    # tabulated CD for (k=3, N=12).
    assert winner_vs_a is True
    assert winner_vs_b is True
    # The two losers alternate per dataset -> tied average rank -> not sig.
    assert losers_diff is False


# ---------------------------------------------------------------------------
# End-to-end multi-dataset report.
# ---------------------------------------------------------------------------


def test_multi_dataset_report_smoke():
    scores = {
        f"ds{i}": {
            "rail_gated": [0.80 + 0.01 * j for j in range(10)],
            "conf": [0.65 + 0.01 * j for j in range(10)],
            "loss": [0.62 + 0.01 * j for j in range(10)],
        }
        for i in range(5)
    }
    report = multi_dataset_report(
        per_dataset_scores=scores,
        baseline="rail_gated",
        higher_is_better=True,
    )
    assert report.friedman.n_datasets == 5
    assert report.friedman.n_methods == 3
    pvs = report.pairwise_wilcoxon_holm["rail_gated"]
    assert 0.0 <= pvs["conf"] <= 1.0
    assert 0.0 <= pvs["loss"] <= 1.0


def test_multi_dataset_report_rejects_missing_baseline():
    scores = {"ds1": {"a": [0.5], "b": [0.5]}}
    with pytest.raises(ValueError):
        multi_dataset_report(
            per_dataset_scores=scores,
            baseline="not_present",
        )
