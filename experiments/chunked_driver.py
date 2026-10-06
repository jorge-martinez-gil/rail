"""Checkpointed chunk runner for the self-contained benchmark.

Runs (dataset, seed) pairs one at a time, appending rows to a partial CSV and
recording completed pairs, so the full headline benchmark can be produced in
short time-budgeted invocations (e.g. sandboxes or pre-emptible CI):

    $ python -m experiments.chunked_driver --budget 35 --out pubout/v4
    ... repeat until it prints DONE ...
    $ python -m experiments.chunked_driver --finalize --out pubout/v4

State lives entirely in ``<out>/run_metrics_partial.csv`` + ``<out>/done.json``;
re-running a completed pair is skipped, so the driver is idempotent.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

FIELDS = [
    "dataset",
    "method",
    "run_id",
    "final_macro_f1",
    "contaminated_admissions",
    "admitted_feedback",
    "total_feedback",
    "admitted_yield",
    "ae",
]


def _load_seeds(repo_root: Path, tier: str) -> list[int]:
    with (repo_root / "SEED_MANIFEST.json").open() as fh:
        manifest = json.load(fh)
    if tier == "smoke":
        return list(manifest["seeds"]["smoke"])
    if tier == "medium":
        return list(manifest["seeds"]["headline_30"][:10])
    if tier == "headline":
        return list(manifest["seeds"]["headline_30"])
    return list(manifest["seeds"]["robustness_50"])


def _datasets(cache: Path, seed: int):
    """Build (or load a pickle cache of) the four benchmark datasets."""
    from . import rail_paper

    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)
    bundles = rail_paper.build_all_datasets(seed=seed)
    with cache.open("wb") as fh:
        pickle.dump(bundles, fh)
    return bundles


def run_chunks(
    out: Path,
    tier: str,
    budget_s: float,
    use_fast: bool,
    shard: tuple[int, int] = (0, 1),
    methods: list[str] | None = None,
) -> bool:
    """Process (dataset, seed, policy) units until the budget is exhausted.

    Checkpoint granularity is a single policy replay so that even the largest
    stream (APS-like, ~8.5k events) fits comfortably inside one invocation.
    Returns True when every unit is complete.
    """
    from . import baselines_integrated, rail_paper, replay_integrated

    if use_fast:
        from .fast_classifier import install_fast_classifier_into_rail_paper

        install_fast_classifier_into_rail_paper()

    repo_root = Path(__file__).resolve().parents[1]
    out.mkdir(parents=True, exist_ok=True)
    seeds = _load_seeds(repo_root, tier)
    bundles = _datasets(out / "datasets.pkl", seeds[0])
    legacy_policies = rail_paper.make_default_policies()
    integrated_names = list(baselines_integrated.INTEGRATED_POLICY_NAMES)
    legacy_names = [p.name for p in legacy_policies]
    if methods is not None:
        integrated_names = [n for n in integrated_names if n in methods]
        legacy_names = [n for n in legacy_names if n in methods or n == "always"]

    done_path = out / "done.json"
    done: set[str] = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    refs_path = out / "always_refs.json"
    refs: dict[str, list[int]] = json.loads(refs_path.read_text()) if refs_path.exists() else {}
    csv_path = out / "run_metrics_partial.csv"
    write_header = not csv_path.exists()

    k, n = shard
    pairs = [(b, int(s)) for b in bundles for s in seeds]
    pairs = [p for i, p in enumerate(pairs) if i % n == k]
    all_units = [(b, s, name) for b, s in pairs for name in legacy_names + integrated_names]
    start = time.time()
    n_done_this_call = 0

    def out_of_budget() -> bool:
        return time.time() - start > budget_s

    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        def emit(row) -> None:
            writer.writerow(asdict(row))
            fh.flush()

        for bundle, seed in pairs:
            pair_key = f"{bundle.name}:{seed}"
            todo = [n for n in legacy_names + integrated_names if f"{pair_key}:{n}" not in done]
            if not todo:
                continue
            if out_of_budget():
                break

            # Deterministic per-(dataset, seed) model + calibration.
            classes = sorted(set(bundle.y_warmup.tolist()))
            model = rail_paper.SklearnOnlineClassifier(classes=classes, random_state=seed)
            model.fit_initial(bundle.X_warmup, bundle.y_warmup)
            calibrated = rail_paper.calibrate_gated_baselines_to_rail_yield(
                base_model=model,
                validation_events=bundle.validation_events,
                policies=legacy_policies,
            )
            by_name = {p.name: p for p in calibrated}

            # 'always' must run first: it defines the AE reference counts.
            if pair_key not in refs or f"{pair_key}:always" not in done:
                row = rail_paper.replay_once(
                    bundle.name,
                    model,
                    bundle.replay_events,
                    bundle.X_test,
                    bundle.y_test,
                    by_name["always"],
                    run_id=seed,
                )
                refs[pair_key] = [row.contaminated_admissions, row.admitted_feedback]
                refs_path.write_text(json.dumps(refs))
                if f"{pair_key}:always" not in done:
                    emit(row)
                    done.add(f"{pair_key}:always")
                    done_path.write_text(json.dumps(sorted(done)))
                    n_done_this_call += 1
            ref = (int(refs[pair_key][0]), int(refs[pair_key][1]))

            integ_by_name = {
                p.name: p for p in baselines_integrated.make_integrated_policies(seed=seed)
            }
            for p in integ_by_name.values():
                if hasattr(p, "calibrate_from_validation"):
                    p.calibrate_from_validation(bundle.validation_events)
            for name in todo:
                if name == "always":
                    continue
                if out_of_budget():
                    break
                if name in integ_by_name:
                    row = replay_integrated.replay_once_integrated(
                        bundle.name,
                        model,
                        bundle.replay_events,
                        bundle.X_test,
                        bundle.y_test,
                        integ_by_name[name],
                        run_id=seed,
                        always_reference_counts=ref,
                    )
                else:
                    row = rail_paper.replay_once(
                        bundle.name,
                        model,
                        bundle.replay_events,
                        bundle.X_test,
                        bundle.y_test,
                        by_name[name],
                        run_id=seed,
                        always_reference_counts=ref,
                    )
                emit(row)
                done.add(f"{pair_key}:{name}")
                done_path.write_text(json.dumps(sorted(done)))
                n_done_this_call += 1

    remaining = sum(1 for b, s, n in all_units if f"{b.name}:{s}:{n}" not in done)
    print(f"[chunk] +{n_done_this_call} units this call; {remaining} remaining")
    return remaining == 0


def finalize(out: Path) -> None:
    """Summarise the partial CSV into the standard stage outputs."""
    import contextlib

    from . import rail_paper, rail_stats_extra
    from .reproduce_paper import _write_multi_dataset_report, _write_summary_csv

    rows = []
    partials = sorted(out.rglob("run_metrics_partial.csv"))
    records = []
    for p in partials:
        with p.open() as fh:
            records.extend(list(csv.DictReader(fh)))
    if True:
        for rec in records:
            rows.append(
                rail_paper.RunMetrics(
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
            )
    summary = rail_paper.summarize_runs(rows)
    _write_summary_csv(summary, out / "summary_metrics.csv")

    per_metric: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        per_metric.setdefault(r.dataset, {}).setdefault(r.method, []).append(r.ae)
    report = rail_stats_extra.multi_dataset_report(
        per_dataset_scores=per_metric,
        baseline="rail_gated",
        higher_is_better=True,
    )
    _write_multi_dataset_report(report, out / "stats_report.md")
    with contextlib.suppress(ImportError):
        rail_stats_extra.critical_difference_diagram(
            report.nemenyi, output_path=str(out / "cd_diagram_ae.pdf")
        )
    print(f"[finalize] wrote summary for {len(rows)} rows -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--tier", default="headline")
    parser.add_argument("--budget", type=float, default=35.0)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--no-fast", action="store_true")
    parser.add_argument("--shard", default="0:1", help="k:n unit sharding for parallel calls")
    parser.add_argument("--methods", default=None, help="comma list; restrict computed methods")
    args = parser.parse_args()
    out = Path(args.out)
    if args.finalize:
        finalize(out)
        return 0
    k, n = (int(x) for x in args.shard.split(":"))
    shard_out = out if n == 1 else out / f"shard_{k}_of_{n}"
    methods = args.methods.split(",") if args.methods else None
    all_done = run_chunks(
        shard_out,
        args.tier,
        args.budget,
        use_fast=not args.no_fast,
        shard=(k, n),
        methods=methods,
    )
    print("DONE" if all_done else "MORE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
