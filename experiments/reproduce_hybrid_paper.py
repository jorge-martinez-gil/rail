"""Reproduction driver for the RAIL-H paper (paper 2).

Stages
------
``temporal``
    Headline benchmark (4 self-contained workloads x 30 seeds x 16 policies)
    re-run with per-event trace capture, producing temporal admission-quality
    metrics (AUCC, NCG, time-to-first-contamination, contamination half-life,
    yield loss) and per-gate diagnostics for the RAIL-H policies (empirical
    alpha/rho of each gate and of the conjunction, plus gate-indicator phi
    correlations for the independence check). Protocol mirrors
    ``experiments.chunked_driver`` exactly: bundles are built once from the
    first manifest seed; per-seed variation enters through the classifier's
    ``random_state`` and per-seed policy state.

``adversarial``
    Operator-model robustness study: 3 conditions (standard / blind /
    inverted; see :mod:`experiments.adversarial_operator`) x 30 seeds x 16
    policies on the published synthetic geometry, with per-seed bundles
    (regime-sweep protocol) and full trace capture.

``imbalance``
    The 11-cell class-imbalance regime sweep (``REGIMES_IMBALANCE``) x 30
    seeds x 16 policies via :func:`experiments.regime_sweep.run_regime_sweep`.

All stages are checkpointed at (unit) granularity and idempotent; re-running
skips completed units. Seeds come from ``SEED_MANIFEST.json`` (headline_30).

Usage::

    python -m experiments.reproduce_hybrid_paper --stage temporal \
        --out publication_outputs/hybrid_paper [--fast]
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def _load_seeds(repo_root: Path, tier: str = "headline") -> list:
    with (repo_root / "SEED_MANIFEST.json").open() as fh:
        manifest = json.load(fh)
    key = {"smoke": "smoke", "headline": "headline_30", "robustness": "robustness_50"}[tier]
    return list(manifest["seeds"][key])


def _write_manifest(out: Path, stage: str, tier: str, use_fast: bool) -> None:
    import numpy as np
    import sklearn

    manifest = {
        "stage": stage,
        "tier": tier,
        "fast_backend": use_fast,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "note": (
            "All rows in this directory come from a single environment; "
            "do not mix with rows produced elsewhere."
        ),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def _run_traced_benchmark(
    out: Path,
    bundle_factory,
    unit_keys,
    seeds,
    use_fast: bool,
    budget_s: float,
    methods: list | None = None,
) -> bool:
    """Shared checkpointed loop for the temporal and adversarial stages.

    ``bundle_factory(unit_key, seed)`` must return a DatasetBundle;
    ``unit_keys`` iterates the outer axis (dataset name or condition).
    Checkpoint granularity: one (unit_key, seed, policy) replay.
    """
    from . import baselines_integrated, rail_paper
    from .hybrid_traces import (
        TRACE_FIELDS,
        replay_once_integrated_traced,
        replay_once_traced,
    )

    if use_fast:
        from .fast_classifier import install_fast_classifier_into_rail_paper

        install_fast_classifier_into_rail_paper()

    out.mkdir(parents=True, exist_ok=True)
    legacy_policies = rail_paper.make_default_policies()
    legacy_names = [p.name for p in legacy_policies]
    integrated_names = list(baselines_integrated.INTEGRATED_POLICY_NAMES)
    if methods is not None:
        integrated_names = [n for n in integrated_names if n in methods]
        # 'always' always runs: it defines the AE reference counts.
        legacy_names = [n for n in legacy_names if n in methods or n == "always"]

    done_path = out / "done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    refs_path = out / "always_refs.json"
    refs = json.loads(refs_path.read_text()) if refs_path.exists() else {}
    csv_path = out / "run_metrics_partial.csv"
    write_header = not csv_path.exists()

    start = time.time()
    n_this_call = 0
    all_units = [(u, s, n) for u in unit_keys for s in seeds for n in legacy_names + integrated_names]

    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRACE_FIELDS)
        if write_header:
            writer.writeheader()

        for unit in unit_keys:
            for seed in seeds:
                pair_key = f"{unit}:{seed}"
                todo = [n for n in legacy_names + integrated_names if f"{pair_key}:{n}" not in done]
                if not todo:
                    continue
                if time.time() - start > budget_s:
                    break

                bundle = bundle_factory(unit, seed)
                classes = sorted(set(bundle.y_warmup.tolist()))
                model = rail_paper.SklearnOnlineClassifier(classes=classes, random_state=seed)
                model.fit_initial(bundle.X_warmup, bundle.y_warmup)
                calibrated = rail_paper.calibrate_gated_baselines_to_rail_yield(
                    base_model=model,
                    validation_events=bundle.validation_events,
                    policies=legacy_policies,
                )
                by_name = {p.name: p for p in calibrated}

                # 'always' first: defines the AE reference counts.
                if pair_key not in refs or f"{pair_key}:always" not in done:
                    row = replay_once_traced(
                        bundle.name, model, bundle.replay_events,
                        bundle.X_test, bundle.y_test, by_name["always"], run_id=seed,
                    )
                    refs[pair_key] = [row.contaminated_admissions, row.admitted_feedback]
                    refs_path.write_text(json.dumps(refs))
                    if f"{pair_key}:always" not in done:
                        writer.writerow(asdict(row))
                        fh.flush()
                        done.add(f"{pair_key}:always")
                        done_path.write_text(json.dumps(sorted(done)))
                        n_this_call += 1
                ref = (int(refs[pair_key][0]), int(refs[pair_key][1]))

                integ_by_name = {
                    p.name: p
                    for p in baselines_integrated.make_integrated_policies(seed=seed)
                }
                for p in integ_by_name.values():
                    if hasattr(p, "calibrate_from_validation"):
                        p.calibrate_from_validation(bundle.validation_events)

                for name in todo:
                    if name == "always":
                        continue
                    if time.time() - start > budget_s:
                        break
                    if name in integ_by_name:
                        row = replay_once_integrated_traced(
                            bundle.name, model, bundle.replay_events,
                            bundle.X_test, bundle.y_test, integ_by_name[name],
                            run_id=seed, always_reference_counts=ref,
                        )
                    else:
                        row = replay_once_traced(
                            bundle.name, model, bundle.replay_events,
                            bundle.X_test, bundle.y_test, by_name[name],
                            run_id=seed, always_reference_counts=ref,
                        )
                    writer.writerow(asdict(row))
                    fh.flush()
                    done.add(f"{pair_key}:{name}")
                    done_path.write_text(json.dumps(sorted(done)))
                    n_this_call += 1

    remaining = sum(1 for u, s, n in all_units if f"{u}:{s}:{n}" not in done)
    print(f"[hybrid-paper] +{n_this_call} units this call; {remaining} remaining", flush=True)
    return remaining == 0


def stage_temporal(out: Path, seeds, use_fast: bool, budget_s: float, methods=None) -> bool:
    """Headline 4-dataset benchmark with trace capture (chunked protocol)."""
    from . import rail_paper

    cache = out / "datasets.pkl"
    out.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with cache.open("rb") as fh:
            bundles = pickle.load(fh)
    else:
        bundles = rail_paper.build_all_datasets(seed=seeds[0])
        with cache.open("wb") as fh:
            pickle.dump(bundles, fh)
    by_name = {b.name: b for b in bundles}
    return _run_traced_benchmark(
        out,
        bundle_factory=lambda unit, seed: by_name[unit],
        unit_keys=list(by_name),
        seeds=seeds,
        use_fast=use_fast,
        budget_s=budget_s,
        methods=methods,
    )


def stage_adversarial(out: Path, seeds, use_fast: bool, budget_s: float, methods=None) -> bool:
    """Operator-model robustness study (per-seed bundles)."""
    from .adversarial_operator import CONDITIONS, build_adversarial_bundle

    return _run_traced_benchmark(
        out,
        bundle_factory=lambda unit, seed: build_adversarial_bundle(unit, seed),
        unit_keys=list(CONDITIONS),
        seeds=seeds,
        use_fast=use_fast,
        budget_s=budget_s,
        methods=methods,
    )


def stage_imbalance(out: Path, seeds, use_fast: bool, budget_s: float = 1e9) -> bool:
    """11-cell class-imbalance sweep, checkpointed at (cell, seed) granularity.

    Uses :func:`experiments.regime_sweep._run_one_seed` (the exact per-seed
    protocol of the published regime sweep) but with chunk-friendly
    checkpointing so the sweep can complete across budgeted invocations.
    Call with ``--finalize`` afterwards to aggregate into
    ``regime_long.csv`` / ``regime_winners.csv``.
    """
    from .baselines_integrated import make_integrated_policies
    from .rail_paper import make_default_policies
    from .regime_sweep import REGIMES_IMBALANCE, _run_one_seed

    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    csv_path = out / "run_metrics_partial.csv"
    fields = [
        "dataset", "method", "run_id", "final_macro_f1", "contaminated_admissions",
        "admitted_feedback", "total_feedback", "admitted_yield", "ae",
    ]
    write_header = not csv_path.exists()
    policies = make_default_policies()
    start = time.time()
    n_this_call = 0

    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for cell in REGIMES_IMBALANCE:
            for seed in seeds:
                key = f"{cell.key()}:{seed}"
                if key in done:
                    continue
                if time.time() - start > budget_s:
                    break
                rows = _run_one_seed(
                    cell, int(seed), policies,
                    lambda s: make_integrated_policies(seed=int(s)),
                    n_replay_events=1200, use_fast=use_fast,
                )
                for row in rows:
                    writer.writerow(asdict(row))
                fh.flush()
                done.add(key)
                done_path.write_text(json.dumps(sorted(done)))
                n_this_call += 1

    total = len(REGIMES_IMBALANCE) * len(list(seeds))
    remaining = total - len(done)
    print(f"[hybrid-paper] +{n_this_call} (cell,seed) pairs this call; {remaining} remaining", flush=True)
    return remaining == 0


def finalize_imbalance(out: Path) -> None:
    """Aggregate the partial imbalance CSV into the standard regime artefacts."""
    from .rail_paper import RunMetrics
    from .regime_sweep import (
        REGIMES_IMBALANCE,
        CellResults,
        _write_long_csv,
        _write_winner_csv,
    )

    by_key = {c.key(): c for c in REGIMES_IMBALANCE}
    results = {k: CellResults(cell=c) for k, c in by_key.items()}
    with (out / "run_metrics_partial.csv").open() as fh:
        for rec in csv.DictReader(fh):
            row = RunMetrics(
                dataset=rec["dataset"],
                method=rec["method"],
                run_id=int(rec["run_id"]),
                final_macro_f1=float(rec["final_macro_f1"]),
                contaminated_admissions=int(rec["contaminated_admissions"]),
                admitted_feedback=int(rec["admitted_feedback"]),
                total_feedback=int(rec["total_feedback"]),
                admitted_yield=float(rec["admitted_yield"]),
                ae=float(rec["ae"]),
            )
            results[row.dataset].per_method_runs.setdefault(row.method, []).append(row)
    ordered = [results[c.key()] for c in REGIMES_IMBALANCE]
    _write_long_csv(ordered, out / "regime_long.csv", metric="ae")
    _write_winner_csv(ordered, out / "regime_winners.csv", metric="ae", higher_is_better=True)
    print(f"[hybrid-paper] finalized imbalance sweep -> {out}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["temporal", "adversarial", "imbalance"])
    parser.add_argument("--out", default="publication_outputs/hybrid_paper")
    parser.add_argument("--tier", default="headline", choices=["smoke", "headline", "robustness"])
    parser.add_argument("--budget", type=float, default=1e9)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--methods", default=None, help="comma list; restrict computed methods")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    seeds = _load_seeds(repo_root, args.tier)
    out = Path(args.out) / args.stage
    out.mkdir(parents=True, exist_ok=True)
    methods = args.methods.split(",") if args.methods else None

    if args.finalize:
        if args.stage == "imbalance":
            finalize_imbalance(out)
        return 0

    if args.stage == "temporal":
        finished = stage_temporal(out, seeds, args.fast, args.budget, methods=methods)
    elif args.stage == "adversarial":
        finished = stage_adversarial(out, seeds, args.fast, args.budget, methods=methods)
    else:
        finished = stage_imbalance(out, seeds, args.fast, budget_s=args.budget)

    _write_manifest(out, args.stage, args.tier, args.fast)
    print("DONE" if finished else "MORE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
