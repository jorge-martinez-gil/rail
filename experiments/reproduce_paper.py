"""End-to-end reproduction driver for the RAIL Information Systems submission.

One stable entry point that reproduces every numeric result in the paper:

  $ python -m experiments.reproduce_paper --tier headline
  $ python -m experiments.reproduce_paper --tier smoke
  $ python -m experiments.reproduce_paper --tier robustness

Tiers
-----
* ``smoke``       : single seed, single dataset, fast sanity check (~1 min)
* ``headline``    : 30 seeds, all datasets, all policies. The paper numbers.
* ``robustness``  : 50 seeds for the *headline* table only, used in the
                    appendix to show CI stability.

Outputs land under ``publication_outputs/<dir>`` per ``SEED_MANIFEST.json``.
A ``run_manifest.json`` is written next to each artefact and contains the
git SHA (if available), Python version, NumPy version, and the actual seeds
used. Reviewers can verify reproducibility by diffing the published manifest
against their own re-run.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import (
        baselines_integrated,
        rail_paper,
        rail_stats_extra,
        regime_sweep,
        replay_integrated,
        theory_risk,
    )
except ImportError:  # pragma: no cover - script-style fallback
    import baselines_integrated  # type: ignore[no-redef]
    import rail_paper  # type: ignore[no-redef]
    import rail_stats_extra  # type: ignore[no-redef]
    import regime_sweep  # type: ignore[no-redef]
    import replay_integrated  # type: ignore[no-redef]
    import theory_risk  # type: ignore[no-redef]


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _REPO_ROOT / "SEED_MANIFEST.json"


def _load_manifest() -> dict:
    with _MANIFEST_PATH.open() as fh:
        return json.load(fh)


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except Exception:
        return None


def _env_fingerprint() -> dict:
    fp = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    for name in ("numpy", "pandas", "sklearn", "matplotlib"):
        try:
            mod = __import__(name)
            fp[name] = getattr(mod, "__version__", "?")
        except ImportError:
            fp[name] = "missing"
    return fp


# ---------------------------------------------------------------------------
# Stage runners.
# ---------------------------------------------------------------------------


def _self_contained_one(
    dataset_seed: int, bundle, legacy_policies, seed: int, use_fast: bool = False
):
    """Run a single (dataset, seed) combination -- top-level so it pickles."""
    if use_fast:
        from .fast_classifier import install_fast_classifier_into_rail_paper

        install_fast_classifier_into_rail_paper()
    # Re-resolve from rail_paper so workers see the monkey-patch.
    from . import rail_paper as _rp

    ClassifierCls = _rp.SklearnOnlineClassifier
    classes = sorted(set(bundle.y_warmup.tolist()))
    model = ClassifierCls(classes=classes, random_state=seed)
    model.fit_initial(bundle.X_warmup, bundle.y_warmup)
    calibrated = rail_paper.calibrate_gated_baselines_to_rail_yield(
        base_model=model,
        validation_events=bundle.validation_events,
        policies=legacy_policies,
    )
    legacy_rows = rail_paper.run_all_policies_once(
        dataset_name=bundle.name,
        base_model=model,
        events=bundle.replay_events,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        policies=calibrated,
        run_id=seed,
    )
    always = next(r for r in legacy_rows if r.method == "always")
    ref = (always.contaminated_admissions, always.admitted_feedback)
    # Reinstantiate integrated policies per seed -- they hold mutable state.
    integ_policies = baselines_integrated.make_integrated_policies(seed=seed)
    integ_rows = replay_integrated.run_all_integrated_policies(
        dataset_name=bundle.name,
        base_model=model,
        events=bundle.replay_events,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        policies=integ_policies,
        always_reference_counts=ref,
        run_id=seed,
    )
    return list(legacy_rows) + list(integ_rows)


def _stage_self_contained(
    seeds: Sequence[int],
    output_dir: Path,
    n_workers: int = 1,
    use_fast: bool = False,
) -> dict:
    """Run the self-contained benchmark with optional parallelism."""

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    datasets = rail_paper.build_all_datasets(seed=seeds[0])
    legacy_policies = rail_paper.make_default_policies()
    pairs = [(b, int(s)) for b in datasets for s in seeds]
    print(
        f"[self_contained] {len(datasets)} datasets x {len(list(seeds))} seeds "
        f"= {len(pairs)} runs, workers={n_workers}"
    )

    if n_workers > 1:
        try:
            from joblib import Parallel, delayed

            rows_by_pair = Parallel(n_jobs=n_workers, verbose=10, backend="loky")(
                delayed(_self_contained_one)(seeds[0], b, legacy_policies, s, use_fast)
                for b, s in pairs
            )
        except ImportError:
            print("[self_contained] joblib not available; running serial")
            rows_by_pair = [
                _self_contained_one(seeds[0], b, legacy_policies, s, use_fast) for b, s in pairs
            ]
    else:
        rows_by_pair = []
        for i, (b, s) in enumerate(pairs):
            if i % max(1, len(pairs) // 10) == 0:
                print(f"[self_contained] {i}/{len(pairs)} ...")
            rows_by_pair.append(_self_contained_one(seeds[0], b, legacy_policies, s, use_fast))

    all_rows: list[rail_paper.RunMetrics] = []
    per_metric_scores: dict[str, dict[str, list[float]]] = {}
    for (bundle, _s), rows in zip(pairs, rows_by_pair, strict=False):
        per_metric_scores.setdefault(bundle.name, {})
        for row in rows:
            all_rows.append(row)
            per_metric_scores[bundle.name].setdefault(row.method, []).append(row.ae)

    summary = rail_paper.summarize_runs(all_rows)
    _write_summary_csv(summary, output_dir / "summary_metrics.csv")
    _write_runs_csv(all_rows, output_dir / "run_metrics.csv")

    # Multi-dataset stats with rail_gated as baseline-of-interest.
    report = rail_stats_extra.multi_dataset_report(
        per_dataset_scores=per_metric_scores,
        baseline="rail_gated",
        higher_is_better=True,
    )
    _write_multi_dataset_report(report, output_dir / "stats_report.md")
    with contextlib.suppress(ImportError):  # matplotlib missing in some CI
        rail_stats_extra.critical_difference_diagram(
            report.nemenyi,
            output_path=str(output_dir / "cd_diagram_ae.pdf"),
        )

    return {
        "wall_clock_s": time.time() - start,
        "n_seeds": len(seeds),
        "n_datasets": len(datasets),
        "n_rows": len(all_rows),
    }


def _stage_regime(
    seeds: Sequence[int],
    output_dir: Path,
    n_workers: int = 1,
    max_cells: int | None = None,
    n_replay_events: int = 2400,
    use_fast: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    cells = list(regime_sweep.REGIMES_2D_DEFAULT)
    if max_cells is not None and max_cells > 0:
        # Stride-sample so we cover the corners of the (noise, load) grid.
        step = max(1, len(cells) // max_cells)
        cells = cells[::step][:max_cells]
    results, _ = regime_sweep.run_regime_sweep(
        regimes=cells,
        seeds=seeds,
        output_dir=output_dir,
        metric="ae",
        higher_is_better=True,
        n_workers=n_workers,
        n_replay_events=n_replay_events,
        use_fast=use_fast,
    )
    with contextlib.suppress(ImportError):  # pragma: no cover - matplotlib optional
        regime_sweep.phase_diagram(
            results,
            output_path=str(output_dir / "phase_diagram_ae.pdf"),
        )
    return {
        "wall_clock_s": time.time() - start,
        "n_cells": len(results),
        "n_seeds_per_cell": len(seeds),
    }


def _stage_theory(output_dir: Path) -> dict:
    """Compute the excess-risk bound across a representative grid and persist."""

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    grid = []
    for c_rail in (0.01, 0.03, 0.05):
        for c_comp in (0.10, 0.20, 0.30):
            for y_rail in (0.40, 0.55, 0.70):
                for y_comp in (0.85, 1.00):
                    if y_comp < y_rail:
                        continue
                    chk = theory_risk.regime_dominance_condition(
                        rail_contamination=c_rail,
                        competitor_contamination=c_comp,
                        rail_yield=y_rail,
                        competitor_yield=y_comp,
                        horizon_total=2000,
                        diameter=1.0,
                        gradient_bound=1.0,
                        loss_gap=1.0,
                    )
                    cross = theory_risk.crossover_loss_gap(
                        rail_contamination=c_rail,
                        competitor_contamination=c_comp,
                        rail_yield=y_rail,
                        competitor_yield=y_comp,
                        horizon_total=2000,
                    )
                    grid.append(
                        {
                            "c_rail": c_rail,
                            "c_comp": c_comp,
                            "y_rail": y_rail,
                            "y_comp": y_comp,
                            "rail_bound": chk.rail_bound,
                            "competitor_bound": chk.competitor_bound,
                            "rail_wins": chk.holds,
                            "crossover_loss_gap": cross,
                        }
                    )

    with (output_dir / "theory_grid.json").open("w") as fh:
        json.dump(grid, fh, indent=2)
    return {"wall_clock_s": time.time() - start, "n_grid_cells": len(grid)}


# ---------------------------------------------------------------------------
# Writers.
# ---------------------------------------------------------------------------


def _write_summary_csv(summary, path: Path) -> None:
    import csv

    rows = [_summary_row_dict(r) for r in summary]
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _summary_row_dict(r) -> dict:
    return {
        "dataset": r.dataset,
        "method": r.method,
        "final_macro_f1_mean": r.final_macro_f1_mean,
        "final_macro_f1_std": r.final_macro_f1_std,
        "contaminated_admissions_mean": r.contaminated_admissions_mean,
        "contaminated_admissions_std": r.contaminated_admissions_std,
        "admitted_yield_mean": r.admitted_yield_mean,
        "admitted_yield_std": r.admitted_yield_std,
        "ae_mean": r.ae_mean,
        "ae_std": r.ae_std,
    }


def _write_runs_csv(rows, path: Path) -> None:
    import csv

    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "method",
                "run_id",
                "final_macro_f1",
                "contaminated_admissions",
                "admitted_feedback",
                "total_feedback",
                "admitted_yield",
                "ae",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "dataset": r.dataset,
                    "method": r.method,
                    "run_id": r.run_id,
                    "final_macro_f1": r.final_macro_f1,
                    "contaminated_admissions": r.contaminated_admissions,
                    "admitted_feedback": r.admitted_feedback,
                    "total_feedback": r.total_feedback,
                    "admitted_yield": r.admitted_yield,
                    "ae": r.ae,
                }
            )


def _write_multi_dataset_report(report, path: Path) -> None:
    f = report.friedman
    n = report.nemenyi
    lines = [
        "# Multi-dataset statistical report",
        "",
        "## Friedman omnibus test (Iman-Davenport corrected)",
        "",
        f"- Statistic: {f.statistic:.4f}",
        f"- p-value:   {f.p_value:.4g}",
        f"- Datasets:  {f.n_datasets}",
        f"- Methods:   {f.n_methods}",
        f"- Reject H0 at alpha=0.05: {f.reject_null}",
        "",
        "## Average ranks (lower = better)",
        "",
        "| Method | Avg rank |",
        "|---|---|",
    ]
    for m, r in sorted(f.average_ranks.items(), key=lambda kv: kv[1]):
        lines.append(f"| {m} | {r:.3f} |")
    lines.extend(
        [
            "",
            "## Nemenyi post-hoc",
            "",
            f"- Critical difference: {n.critical_difference:.4f} (alpha={n.alpha})",
            "- Pairs significantly different: see significant matrix below.",
            "",
            "| Method A | Method B | Sig? |",
            "|---|---|---|",
        ]
    )
    methods = sorted(n.average_ranks.keys())
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            lines.append(f"| {a} | {b} | {'yes' if n.significant[a][b] else 'no'} |")
    lines.extend(
        [
            "",
            "## Holm-corrected paired Wilcoxon vs rail_gated",
            "",
            "| Method | Holm-corrected p-value |",
            "|---|---|",
        ]
    )
    baseline_key = next(iter(report.pairwise_wilcoxon_holm))
    for m, p in sorted(report.pairwise_wilcoxon_holm[baseline_key].items()):
        lines.append(f"| {m} | {p:.4g} |")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the RAIL paper.")
    parser.add_argument(
        "--tier",
        choices=("smoke", "medium", "headline", "robustness"),
        default="medium",
        help=(
            "Which seed tier from SEED_MANIFEST.json to use. "
            "smoke=1 seed, medium=10 seeds, headline=30, robustness=50."
        ),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("self_contained", "regime", "theory"),
        default=("self_contained", "regime", "theory"),
        help="Which stages to run.",
    )
    parser.add_argument(
        "--output-root",
        default=str(_REPO_ROOT / "publication_outputs"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (joblib loky backend). Use os.cpu_count()-1.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Cap the number of regime cells (stride-subsamples the default grid).",
    )
    parser.add_argument(
        "--n-replay-events",
        type=int,
        default=2400,
        help="Replay events per (cell, seed) in the regime sweep. "
        "1200 is fine for winner detection.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override seed count (cap to first N seeds of the tier).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the pure-NumPy online classifier (~30x faster per event "
        "than sklearn's SGDClassifier; on by default with --tier medium).",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Force the original sklearn SGDClassifier backend (for "
        "bit-exact reproduction of the v1 numbers).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Install the fast classifier BEFORE any module-level dataset build.
    #
    # IMPORTANT: ``headline`` no longer auto-selects the fast NumPy backend.
    # The fast classifier has NOT been validated as equivalent for the headline
    # conclusion: the most recent --fast headline run (self_contained_v3) and
    # the regime sweep show ITLM overtaking RAIL on Admission-Efficiency, which
    # contradicts the manuscript (built on the sklearn-era self_contained_v2).
    # The root cause is unresolved (backend vs. replay length / seeds / an ITLM
    # change), so canonical paper numbers must come from the sklearn backend and
    # be cross-checked with ``experiments/diagnose_backend.py``. ``--fast``
    # remains available for quick iteration; ``medium`` still defaults to fast.
    use_fast = (args.fast or args.tier == "medium") and not args.slow
    if use_fast:
        from .fast_classifier import install_fast_classifier_into_rail_paper

        install_fast_classifier_into_rail_paper()
        print("[backend] NumpyOnlineClassifier (fast)")
        if args.tier in ("headline", "robustness"):
            print(
                "[backend] WARNING: --fast on a paper tier is NON-CANONICAL; "
                "it can change the headline winner. Use --slow for paper numbers."
            )
    else:
        print("[backend] sklearn SGDClassifier (slow, canonical)")

    manifest = _load_manifest()
    if args.tier == "smoke":
        seeds = manifest["seeds"]["smoke"]
    elif args.tier == "medium":
        seeds = manifest["seeds"]["headline_30"][:10]
    elif args.tier == "headline":
        seeds = manifest["seeds"]["headline_30"]
    else:
        seeds = manifest["seeds"]["robustness_50"]
    if args.seeds is not None:
        seeds = list(seeds)[: max(1, args.seeds)]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _env_fingerprint()
    stage_info: dict[str, dict] = {}

    if "self_contained" in args.stages:
        out = output_root / "self_contained_v3"
        stage_info["self_contained"] = _stage_self_contained(
            seeds,
            out,
            n_workers=args.workers,
            use_fast=use_fast,
        )
    if "regime" in args.stages:
        out = output_root / "regime"
        regime_seeds = seeds if args.tier != "smoke" else seeds[:1]
        stage_info["regime"] = _stage_regime(
            regime_seeds,
            out,
            n_workers=args.workers,
            max_cells=args.max_cells,
            n_replay_events=args.n_replay_events,
            use_fast=use_fast,
        )
    if "theory" in args.stages:
        out = output_root / "theory"
        stage_info["theory"] = _stage_theory(out)

    manifest_out = {
        "tier": args.tier,
        "seeds_used": list(seeds),
        "stages": stage_info,
        "environment": fingerprint,
    }
    with (output_root / "run_manifest.json").open("w") as fh:
        json.dump(manifest_out, fh, indent=2)
    print(json.dumps(manifest_out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
