# RAIL Research Artifact Guide

This repository contains the RAIL console and the experiment code used to
evaluate reliability-aware admission of human feedback for stream
recalibration. Everything in the published paper is reproducible from a
single driver, `experiments/reproduce_paper.py`, over the frozen seeds in
`SEED_MANIFEST.json`.

## Scope of the evidence

The four headline streams (Synthetic, SECOM-like, APS-like, ATC-like) are
generated **self-contained** by the reproduction driver. They preserve the
dimensionality, class imbalance, and workload structure of their real-world
references without requiring any external download, so a reviewer can
execute the full artifact offline. Claims based on these streams are
described in the paper as controlled stress tests, not as direct
real-world measurements.

## Recommended setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[experiments,dev]"
```

## Canonical commands

Reproduce the paper (headline tier, 30 seeds, all stages):

```bash
python -m experiments.reproduce_paper --tier headline --workers 8 --slow
```

Quick sanity check (~30 s) and iteration tier are documented in `RUN.md`:

```bash
python -m experiments.reproduce_paper --tier smoke --workers 2 --fast
python -m experiments.reproduce_paper --tier medium --workers 8
```

Regenerate the two main-text figures from a completed headline run:

```bash
python -m experiments.make_main_figures \
    --data publication_outputs/self_contained_v3 \
    --out publication_outputs/main_figures
```

Cross-check the two online-learner backends (see the backend note in
`RUN.md`):

```bash
python -m experiments.diagnose_backend --seeds 5 --datasets Synthetic
```

Run the test suite for the shared scoring and statistics logic:

```bash
pytest
```

Aggregate browser activity reports exported by the RAIL console:

```bash
python -m experiments.activity_report path/to/activity-reports \
    --output-dir publication_outputs/activity
```

## Outputs

Each stage of the driver writes below `publication_outputs/`:

- `self_contained_v3/`: `summary_metrics.csv`, `run_metrics.csv`,
  `stats_report.md`, `cd_diagram_ae.pdf` — the canonical headline data.
- `main_figures/`: the two main-text figures (`fig_contamination_prevention`,
  `fig_pareto_yield_vs_contamination`) as 600 DPI PNG and vector PDF.
- `regime/`: `regime_long.csv`, `regime_winners.csv`, `phase_diagram_ae.pdf`.
- `theory/`: `theory_grid.json` (closed-form contract/risk grid).
- `run_manifest.json`: tier, seeds, environment fingerprint, and git SHA.

Generated outputs and any downloaded caches are intentionally ignored by
Git. For submission, archive `publication_outputs/` and deposit it with the
accepted artifact under a persistent DOI.

## Reviewer notes

- The browser console (`app/`) uses MQTT over WebSockets and runs without a
  build step; the experiments do not depend on any broker.
- JSON Schemas for export, recalibration, and activity-report payloads live
  in `schemas/`.
- The contamination contract is a Bayes-rule bound over validation rates;
  `experiments/rail_core.py::contamination_contract` exposes the same
  calculation used in the paper.
- Activity reporting is opt-in, local to the browser, and exports
  pseudonymous study metadata plus aggregate timing/focus/edit/vigilance
  summaries; raw row values and free-text notes are excluded unless the
  operator explicitly enables them before export.
- Backend sensitivity: the NumPy and sklearn online-learner backends are
  different optimisers. Confirm the headline ranking under `--slow` at full
  scale and audit it with `experiments/diagnose_backend.py` before
  camera-ready.
