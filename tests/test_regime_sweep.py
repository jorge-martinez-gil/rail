"""Smoke tests for the regime-sweep harness.

These tests exercise the full pipeline at a tiny scale (1 seed, 2 cells) so
that integration breakage shows up in CI without burning minutes on a real
sweep.
"""

from __future__ import annotations

import csv
from pathlib import Path

from experiments.regime_sweep import (
    REGIMES_2D_DEFAULT,
    REGIMES_IMBALANCE,
    RegimeCell,
    run_regime_sweep,
)


def test_regime_cells_are_well_formed():
    for cell in REGIMES_2D_DEFAULT + REGIMES_IMBALANCE:
        assert 0.0 < cell.correct_prob_normal <= 1.0
        assert 0.0 < cell.correct_prob_overload <= 1.0
        assert 0.0 <= cell.overload_prob <= 1.0
        assert cell.key() != ""


def test_run_regime_sweep_smoke(tmp_path):
    """Tiny sweep: 2 cells x 1 seed. Verifies the harness wires up correctly
    and produces the expected CSV artefacts."""
    cells = [
        RegimeCell(
            correct_prob_normal=0.90,
            correct_prob_overload=0.65,
            overload_prob=0.30,
            class_imbalance=0.0,
            label="cell_easy",
        ),
        RegimeCell(
            correct_prob_normal=0.70,
            correct_prob_overload=0.45,
            overload_prob=0.45,
            class_imbalance=0.0,
            label="cell_hard",
        ),
    ]
    results, out_dir = run_regime_sweep(
        regimes=cells,
        seeds=[7],
        output_dir=tmp_path / "sweep",
    )
    assert len(results) == 2
    # Both CSV artefacts written.
    long_csv = Path(out_dir) / "regime_long.csv"
    winners_csv = Path(out_dir) / "regime_winners.csv"
    assert long_csv.exists() and long_csv.stat().st_size > 0
    assert winners_csv.exists() and winners_csv.stat().st_size > 0
    # Long CSV has every (cell, method) pair.
    with long_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    cells_seen = {r["cell"] for r in rows}
    methods_seen = {r["method"] for r in rows}
    assert cells_seen == {"cell_easy", "cell_hard"}
    # At least the legacy + integrated policy families show up.
    assert "rail_gated" in methods_seen
    assert "rail_weighted" in methods_seen
    assert "gce_weight" in methods_seen
    # Winner CSV has 2 rows.
    with winners_csv.open() as fh:
        winner_rows = list(csv.DictReader(fh))
    assert len(winner_rows) == 2
    for row in winner_rows:
        assert row["winner"] in methods_seen
        assert float(row["margin"]) >= 0.0 - 1e-9
