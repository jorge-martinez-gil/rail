"""Tests for the sensitivity sweep and operational metrics modules."""

import json
from pathlib import Path

from experiments.metrics import (
    admission_yield_loss,
    burn_in_estimator_from_paper,
    contamination_half_life,
    cumulative_contamination_curve,
    time_to_first_contamination,
)
from experiments.rail_operator import synthesise_events
from experiments.sensitivity import (
    sweep_grid,
    synthetic_sweep_demo,
    theta_only_pareto,
    write_pareto_json,
    write_sweep_csv,
)


def test_time_to_first_contamination_basic():
    correct = [True, True, False, True, False]
    admitted = [True, True, True, True, True]
    assert time_to_first_contamination(correct, admitted) == 2


def test_time_to_first_contamination_none():
    assert time_to_first_contamination([True, True], [True, True]) == -1


def test_contamination_half_life_basic():
    correct = [False, True, False, True, False]
    admitted = [True, True, True, True, True]
    # Contaminated admissions are at indices 0, 2, 4 (total 3, half = 2).
    # First index where running count >= 2 is index 2.
    assert contamination_half_life(correct, admitted) == 2


def test_cumulative_contamination_curve_monotone():
    correct = [True, False, True, False, True]
    admitted = [True, True, True, True, True]
    curve = cumulative_contamination_curve(correct, admitted)
    assert len(curve) == 5
    # Cumulative contaminated should be monotone non-decreasing.
    contam = [p.cumulative_contaminated for p in curve]
    assert all(contam[i] <= contam[i + 1] for i in range(len(contam) - 1))


def test_admission_yield_loss_full_rejection():
    correct = [True, True, True]
    admitted = [False, False, False]
    assert admission_yield_loss(correct, admitted) == 1.0


def test_admission_yield_loss_full_acceptance():
    correct = [True, True, True]
    admitted = [True, True, True]
    assert admission_yield_loss(correct, admitted) == 0.0


def test_burn_in_estimator_returns_widening_window():
    deltas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    lo, hi = burn_in_estimator_from_paper(deltas)
    assert lo < hi
    assert lo >= 0.3
    assert hi <= 30.0


def test_burn_in_estimator_handles_short_input():
    lo, hi = burn_in_estimator_from_paper([0.5, 1.0])
    assert lo < hi
    assert lo >= 0.3


def test_sweep_grid_produces_one_row_per_param_combo():
    events = synthesise_events(n=200, seed=0)
    rows = sweep_grid(
        events,
        tau_min_grid=(0.8, 1.2),
        tau_max_grid=(5.0, 7.0),
        k_grid=(1.4,),
        theta_grid=(0.5, 0.6),
    )
    # 2 tau_min * 2 tau_max * 1 k * 2 theta = 8 rows (no invalid windows)
    assert len(rows) == 8
    for r in rows:
        assert 0.0 <= r.yield_rate <= 1.0
        assert 0.0 <= r.contamination_rate <= 1.0


def test_sweep_grid_skips_invalid_windows():
    events = synthesise_events(n=100, seed=0)
    rows = sweep_grid(
        events,
        tau_min_grid=(5.0,),
        tau_max_grid=(3.0,),  # invalid: tau_min > tau_max
        k_grid=(1.4,),
        theta_grid=(0.5,),
    )
    assert rows == []


def test_theta_pareto_includes_extremes(tmp_path: Path):
    events = synthesise_events(n=300, seed=1)
    pts = theta_only_pareto(events, thetas=(0.0, 0.5, 1.0))
    assert len(pts) == 3
    # At theta=0 nothing is rejected (yield = 1.0 minus the floor case).
    assert pts[0].yield_rate >= 0.99
    # At theta=1 almost nothing is admitted.
    assert pts[-1].yield_rate <= 0.05


def test_synthetic_sweep_demo_writes_files(tmp_path: Path):
    out = synthetic_sweep_demo(tmp_path, n_events=200, seed=3)
    sweep_path = Path(out["sweep_csv"])
    pareto_path = Path(out["pareto_json"])
    assert sweep_path.exists()
    assert pareto_path.exists()
    data = json.loads(pareto_path.read_text())
    assert "frontier" in data and "envelope" in data


def test_write_helpers_round_trip(tmp_path: Path):
    events = synthesise_events(n=100, seed=2)
    rows = sweep_grid(
        events,
        tau_min_grid=(0.9,),
        tau_max_grid=(5.8,),
        k_grid=(1.4,),
        theta_grid=(0.6,),
    )
    csv_path = write_sweep_csv(rows, tmp_path / "sweep.csv")
    assert Path(csv_path).exists()

    pts = theta_only_pareto(events, thetas=(0.4, 0.6, 0.8))
    pareto_path = write_pareto_json(pts, tmp_path / "pareto.json")
    assert Path(pareto_path).exists()
