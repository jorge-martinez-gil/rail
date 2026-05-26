# RAIL Research Artifact Guide

This repository contains the RAIL console and the experiment code used to evaluate reliability-aware admission of human feedback for stream recalibration.

## Evidence Tiers

The codebase separates two kinds of evidence.

1. Real-data decay experiments: SECOM, APS Failure, synthetic drift stress test, and ATC corpora are orchestrated through `experiments/experiments_ae.py`.
2. Self-contained benchmark-like experiments: `experiments/rail_paper.py` uses generated datasets that mimic the stressors in the paper and compares RAIL against stronger gating baselines.

Use the real-data pipeline for the primary empirical claims. Use the self-contained pipeline for fast review, CI smoke checks, and ablation-style stress testing when external datasets are unavailable.

## Recommended Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[experiments,dev]"
```

On Unix-like systems, replace the activation command with `source .venv/bin/activate`.

## Canonical Commands

Run the full publication pipeline:

```bash
python experiments/run_publication.py --mode both --runs 20 --seed 7
```

On Windows, the same full artifact run is wrapped by:

```bat
run_all_results.bat
```

Run only the real-data experiments:

```bash
python experiments/run_publication.py --mode real --seed 7
```

Run only the self-contained benchmark-like experiments:

```bash
python experiments/run_publication.py --mode self-contained --runs 20 --seed 7
```

Run tests for shared scoring logic:

```bash
pytest
```

Aggregate browser activity reports exported by the RAIL console:

```bash
python experiments/activity_report.py path/to/activity-reports --output-dir publication_outputs/activity
```

## Outputs

By default, generated outputs are written below `publication_outputs/`.

- `publication_outputs/real_data/`: real-data plots and admission-efficiency tables.
- `publication_outputs/self_contained/`: generated-dataset summaries, LaTeX tables, and JSON/CSV metrics.
- `publication_outputs/activity/`: session-level and condition-level summaries from exported browser activity reports.
- `publication_outputs/run_manifest.json`: command-line settings, Python version, platform, and timestamps for the run.
- `publication_outputs/JOURNAL_ARTIFACTS.md`: index of generated figures, tables, metadata, and recommended manuscript assets.

Figures are written as 600 DPI PNG files plus PDF and SVG vector masters when generated through the canonical publication runner. Tables are written as LaTeX masters with CSV and Markdown companions where possible.

Generated outputs and downloaded dataset caches are intentionally ignored by Git. For submission, archive the output directory and deposit it with the accepted artifact or a data repository.

## Data Availability Statement Draft

The SECOM and APS datasets are downloaded from the UCI Machine Learning Repository. ATC corpora are downloaded through Hugging Face Datasets where licensing permits. The repository also includes self-contained benchmark-like generators for reviewers who need to execute the artifact without external data access. Generated results, run manifests, and source code should be deposited with a persistent DOI before submission.

## Reviewer Notes

- The browser console uses MQTT over WebSockets and can be run without a build step.
- The default app configuration targets a local WebSocket-enabled MQTT broker; experiments do not depend on any broker.
- JSON Schemas for export, recalibration, and activity-report payloads live in `schemas/`.
- The README states the contamination contract as a Bayes-rule bound over validation rates, and `experiments/rail_core.py::contamination_contract` exposes the same calculation for reproducibility.
- Activity reporting is opt-in, local to the browser, and exports pseudonymous study metadata (`participant_id`, `condition`, `task_batch`) plus aggregate timing/focus/edit/vigilance summaries. Raw row values and free-text notes are excluded unless the operator explicitly enables them before export.
- Claims based on generated datasets should be described as stress tests, not as direct real-world evidence.
