"""Sensitivity sweeps and Pareto utilities for the RAIL admission gate.

RQ4 in the paper asks how performance varies with the threshold and the base
admission window. We provide a small, reproducible harness that:

* Builds a synthetic interaction stream with controllable contamination
  (re-using :func:`experiments.rail_operator.synthesise_events`).
* Sweeps any subset of ``(theta, tau_min, tau_max, k)`` and records yield,
  contamination, admission efficiency, and admission-window-relative
  diagnostics per grid point.
* Exports results to CSV/JSON ready for plotting (or for the markdown
  rendering helpers in :mod:`experiments.publication_artifacts`).
* Builds the yield-vs-contamination Pareto frontier (re-using
  :func:`experiments.theory.pareto_frontier`).

The harness has no NumPy/Pandas dependency so it can run inside the test
suite. Heavy plots remain the responsibility of the existing publication
artifact scripts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from .rail_core import (
        AdmissionParams,
        admission_diagnostics,
        contamination_contract,
    )
    from .rail_operator import InteractionEvent, synthesise_events
    from .theory import ParetoPoint, pareto_frontier, pareto_lower_envelope
except ImportError:  # pragma: no cover
    from rail_core import (
        AdmissionParams,
        admission_diagnostics,
        contamination_contract,
    )
    from rail_operator import InteractionEvent, synthesise_events
    from theory import ParetoPoint, pareto_frontier, pareto_lower_envelope


@dataclass(frozen=True)
class SweepRow:
    tau_min: float
    tau_max: float
    k: float
    theta: float
    admitted: int
    contaminated_admitted: int
    yield_rate: float
    contamination_rate: float
    admission_efficiency: float
    contamination_bound: float


def _score_events(
    events: Sequence[tuple[InteractionEvent, bool]],
    params: AdmissionParams,
) -> tuple[list[float], list[bool]]:
    scores: list[float] = []
    correct: list[bool] = []
    for ev, ok in events:
        diag = admission_diagnostics(
            delta_sec=ev.delta_sec,
            num_features=ev.num_features_shown,
            edit_count=ev.edit_count,
            focus_seconds=ev.focus_time_s,
            params=params,
        )
        scores.append(diag["score"])
        correct.append(ok)
    return scores, correct


def sweep_grid(
    events: Sequence[tuple[InteractionEvent, bool]],
    tau_min_grid: Sequence[float] = (0.6, 0.9, 1.2),
    tau_max_grid: Sequence[float] = (4.5, 5.8, 8.0),
    k_grid: Sequence[float] = (0.9, 1.4, 1.8),
    theta_grid: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70),
    base_params: AdmissionParams | None = None,
) -> list[SweepRow]:
    """Cartesian-product sweep over the four main admission parameters."""

    base = base_params or AdmissionParams()
    out: list[SweepRow] = []
    for tmin in tau_min_grid:
        for tmax in tau_max_grid:
            if not tmin < tmax:
                continue
            for k in k_grid:
                params = AdmissionParams(
                    tau_min=tmin,
                    tau_max=tmax,
                    k=k,
                    theta=base.theta,
                    w_delta=base.w_delta,
                    w_features=base.w_features,
                    w_edits=base.w_edits,
                    w_focus=base.w_focus,
                )
                scores, correct = _score_events(events, params)
                for theta in theta_grid:
                    admitted = [s >= theta for s in scores]
                    contract = contamination_contract(correct, admitted)
                    out.append(
                        SweepRow(
                            tau_min=tmin,
                            tau_max=tmax,
                            k=k,
                            theta=theta,
                            admitted=int(contract["admitted_feedback"]),
                            contaminated_admitted=int(contract["contaminated_admissions"]),
                            yield_rate=contract["admitted_feedback"] / max(1, len(events)),
                            contamination_rate=contract["admitted_contamination_rate"],
                            admission_efficiency=contract["admission_efficiency"],
                            contamination_bound=contract["contamination_bound"],
                        )
                    )
    return out


def theta_only_pareto(
    events: Sequence[tuple[InteractionEvent, bool]],
    base_params: AdmissionParams | None = None,
    thetas: Sequence[float] | None = None,
) -> list[ParetoPoint]:
    """Return the Pareto frontier obtained by sweeping ``theta`` alone."""

    base = base_params or AdmissionParams()
    scores, correct = _score_events(events, base)
    return pareto_frontier(scores, correct, thetas=thetas)


def write_sweep_csv(rows: Iterable[SweepRow], path: str | Path) -> str:
    """Persist a sweep table to CSV. Returns the path written."""

    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tau_min",
        "tau_max",
        "k",
        "theta",
        "admitted",
        "contaminated_admitted",
        "yield_rate",
        "contamination_rate",
        "admission_efficiency",
        "contamination_bound",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in fields})
    return str(path)


def write_pareto_json(
    frontier: Sequence[ParetoPoint],
    path: str | Path,
    include_envelope: bool = True,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[Mapping[str, float]]] = {
        "frontier": [
            {
                "theta": p.theta,
                "yield_rate": p.yield_rate,
                "contamination_rate": p.contamination_rate,
                "admission_efficiency": p.admission_efficiency,
            }
            for p in frontier
        ]
    }
    if include_envelope:
        env = pareto_lower_envelope(frontier)
        payload["envelope"] = [
            {
                "theta": p.theta,
                "yield_rate": p.yield_rate,
                "contamination_rate": p.contamination_rate,
                "admission_efficiency": p.admission_efficiency,
            }
            for p in env
        ]
    path.write_text(json.dumps(payload, indent=2))
    return str(path)


def synthetic_sweep_demo(
    output_dir: str | Path,
    n_events: int = 2_000,
    contamination_rate: float = 0.3,
    seed: int = 0,
) -> dict[str, str]:
    """One-call demo that writes a sweep CSV and a Pareto JSON for ``output_dir``.

    Used by the README, by reviewers who want a quick visual, and by the test
    suite to guard against silent regressions in the grid logic.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = synthesise_events(
        n=n_events, seed=seed, contamination_rate=contamination_rate
    )
    sweep = sweep_grid(events)
    sweep_path = write_sweep_csv(sweep, output_dir / "rail_sensitivity_sweep.csv")
    frontier = theta_only_pareto(events)
    pareto_path = write_pareto_json(frontier, output_dir / "rail_pareto.json")
    return {"sweep_csv": sweep_path, "pareto_json": pareto_path}


__all__ = [
    "SweepRow",
    "sweep_grid",
    "theta_only_pareto",
    "write_sweep_csv",
    "write_pareto_json",
    "synthetic_sweep_demo",
]
