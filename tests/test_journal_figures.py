from __future__ import annotations

import pandas as pd
import pytest

from experiments.make_journal_figures import (
    bootstrap_mean_ci,
    pareto_frontier,
    regime_contrasts,
)


def test_bootstrap_interval_is_reproducible():
    first = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.5], n_boot=2_000, seed=9)
    second = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.5], n_boot=2_000, seed=9)
    assert first == second
    assert first.low <= first.mean <= first.high


def test_pareto_frontier_uses_high_yield_and_low_contamination():
    points = [
        ("trim", 0.5, 0.04),
        ("yield", 0.9, 0.12),
        ("dominated", 0.4, 0.18),
    ]
    assert [row[0] for row in pareto_frontier(points)] == ["trim", "yield"]


def test_regime_contrasts_select_best_gate_per_metric():
    rows = [
        {
            "cell": "c1",
            "correct_prob_normal": 0.9,
            "overload_prob": 0.2,
            "method": "rail_gated",
            "ae_mean": 0.5,
            "macro_f1_mean": 0.7,
            "n_seeds": 30,
        },
        {
            "cell": "c1",
            "correct_prob_normal": 0.9,
            "overload_prob": 0.2,
            "method": "confidence_gated",
            "ae_mean": 0.4,
            "macro_f1_mean": 0.72,
            "n_seeds": 30,
        },
        {
            "cell": "c1",
            "correct_prob_normal": 0.9,
            "overload_prob": 0.2,
            "method": "loss_gated",
            "ae_mean": 0.45,
            "macro_f1_mean": 0.68,
            "n_seeds": 30,
        },
        {
            "cell": "c1",
            "correct_prob_normal": 0.9,
            "overload_prob": 0.2,
            "method": "margin_gated",
            "ae_mean": 0.35,
            "macro_f1_mean": 0.69,
            "n_seeds": 30,
        },
        {
            "cell": "c1",
            "correct_prob_normal": 0.9,
            "overload_prob": 0.2,
            "method": "dynamic_quantile",
            "ae_mean": 0.42,
            "macro_f1_mean": 0.71,
            "n_seeds": 30,
        },
    ]
    result = regime_contrasts(pd.DataFrame(rows)).iloc[0]
    assert result.ae_comparator == "loss_gated"
    assert result.f1_comparator == "confidence_gated"
    assert result.ae_difference == pytest.approx(0.05)
    assert result.f1_difference == pytest.approx(-0.02)
