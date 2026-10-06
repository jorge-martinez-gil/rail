# Reproducing the RAIL paper

The single command for every run tier. Estimates assume an 8-core machine
with the NumPy fast classifier (`--fast`, on by default for `medium` and
`headline`).

## Quick sanity (~30 seconds)
```bash
python -m experiments.reproduce_paper --tier smoke --workers 2 --fast
```
1 seed, all stages. Use this to verify the install works.

## Medium tier (recommended for iteration: ~2-3 minutes)
```bash
python -m experiments.reproduce_paper --tier medium \
    --workers 8 --max-cells 12 --n-replay-events 1200
```
10 seeds, 12 regime cells, 1200 replay events per cell. Auto-uses the
NumPy fast classifier. The winner pattern in the regime sweep stabilises
well before 30 seeds; use this tier to draft the paper and iterate.

## Headline (paper numbers)
```bash
python -m experiments.reproduce_paper --tier headline --workers 8 --slow
```
30 seeds (`headline_30`), all 24 regime cells, 2400 replay events. The
released headline artifacts are frozen in `publication_outputs/self_contained_v3/`
and are the numbers reported in the manuscript tables and figures. Before
camera-ready, confirm the headline method ranking under `--slow` at full
scale and cross-check the two online-learner backends with
`python -m experiments.diagnose_backend` (see the backend note below).

## Robustness (appendix CI stability)
```bash
python -m experiments.reproduce_paper --tier robustness --workers 8 --slow
```
50 seeds. Demonstrates that headline confidence intervals are stable.

## Why no GPU?

GPU does NOT help this workload. The math per event is a 12x4 matmul --
the GPU host-device transfer overhead alone is 1000x the compute. The
bottleneck was Python/sklearn per-call overhead in
``SGDClassifier.partial_fit(single_sample)`` (~150 us per call across 24M
calls). The fast classifier (`experiments/fast_classifier.py`) replaces
that with a hand-rolled NumPy multinomial logistic regression that has no
per-call overhead, giving ~30x per-event speedup. Combined with joblib
parallelism, that takes headline from 7 hours to under 10 minutes.

## --fast vs --slow

- ``--fast`` -- pure NumPy backend. Default for ``medium`` only (the
  iteration tier). ~30x faster than sklearn per event. It is an online
  logistic regression but the optimisation trajectory differs from
  sklearn's SGD, and that difference is **not** cosmetic.

  > **Backend validation note.** The NumPy (`--fast`) and sklearn (`--slow`)
  > backends are different online optimisers, and the headline method
  > ranking (ITLM strongest on Admission-Efficiency, the RAIL policies the
  > next average-rank tier) should not be assumed identical across them.
  > `python -m experiments.diagnose_backend` replays the same seeds through
  > both backends and prints a side-by-side ranking so the headline result
  > can be audited for backend sensitivity. Run this, plus a full-scale
  > `--slow` headline pass, before camera-ready.

- ``--slow`` -- the sklearn SGDClassifier backend. Use this for the
  pre-submission confirmation run; pair it with the backend diagnosis above
  so the released artifacts and the reported ranking are auditable.

## Just one stage
```bash
# Theory grid is closed-form, takes ~1 second.
python -m experiments.reproduce_paper --stages theory

# Self-contained tables only.
python -m experiments.reproduce_paper --tier headline --workers 8 --stages self_contained

# Regime phase diagram only.
python -m experiments.reproduce_paper --tier medium --workers 8 --stages regime
```

## Real-operator pilot (~2 seconds)

Collect new participant sessions with the standalone labelling tool:

```powershell
python -m http.server 8000 --directory app
# Open http://127.0.0.1:8000/labeling.html
```

The page exports one `rail-human-telemetry-v1` JSON file and a CSV mirror per participant. Put the
JSON exports in a directory and pass that directory to the analysis command below.

```bash
python -m experiments.human_study "real tests" --output-dir publication_outputs/human_study
```
Audits the five `rail-human-telemetry-v1` session files in `real tests/`
(re-computes every V from raw telemetry through `experiments/rail_core.py` and
cross-checks the console's recorded scores and admission decisions), then
regenerates the pilot table (`table_human_pilot.tex`), the two-panel figure
(`fig_human_pilot.pdf`), and the summary JSON reported in the manuscript's
real-operator pilot section. Copy `fig_human_pilot.pdf` into `paper/figs/`
after regeneration.

## Custom knobs
- `--seeds N` -- cap to first N seeds of the tier (quick parameter sweep)
- `--max-cells N` -- stride-subsample the 24-cell regime grid
- `--n-replay-events N` -- per-cell replay length (1200 OK for winner detection)
- `--workers N` -- joblib loky workers (use os.cpu_count() - 1)

## Checkpointed / pre-emptible runs (`experiments/chunked_driver.py`)

For environments that can kill the process at any moment (CI, sandboxes,
laptops), the chunked driver runs the self-contained benchmark one
(dataset, seed, policy) replay at a time with on-disk checkpoints, so it can
be re-invoked until complete and is fully idempotent:

```
python -m experiments.chunked_driver --out publication_outputs/v4 --tier headline --budget 300
# repeat until it prints DONE; parallel shards: --shard 0:2 / --shard 1:2 (separate processes)
python -m experiments.chunked_driver --out publication_outputs/v4 --finalize
```

`self_contained_v4_hybrid/` was produced this way (fast backend). IMPORTANT:
never mix rows produced in different environments -- final Macro-F1 and
model-dependent admission counts drift across numpy/python versions (the
'always' reference counts are env-invariant and are cross-checked in the
manifest).

## Outputs
Each stage writes to `publication_outputs/<stage>/`:
- `self_contained_v3/`: summary_metrics.csv, run_metrics.csv, stats_report.md, cd_diagram_ae.pdf
- `regime/`: regime_long.csv, regime_winners.csv, phase_diagram_ae.pdf
- `theory/`: theory_grid.json
- `run_manifest.json`: tier, seeds, environment fingerprint, git SHA

## Why this is fast enough now

The original `--tier headline` was ~7 hours serial because of a real
bottleneck I should explain so you can give an honest accounting in the paper:

- 24 regime cells x 30 seeds x 14 methods x 2400 replay events = ~24M
  per-event evaluations
- Each event runs `SGDClassifier.partial_fit(single_sample)`, which has a
  large per-call Python overhead (the dominant cost, not the math)
- The integrated baselines (CoTeaching, JointAgreement) additionally
  sample two Dirichlet draws per event for the two-view loss

The parallelism uses joblib's loky backend across (cell, seed) pairs --
each pair is fully independent (no shared mutable state). The integrated
policies are reinstantiated per seed, which is a separate correctness fix
(they hold sliding-window quantile state that must not leak across seeds).

For the camera-ready, run `--tier headline` once and freeze the manifest.
