"""Controlled fast-vs-slow backend diagnosis for the RAIL headline result.

Motivation
----------
The headline Admission-Efficiency (AE) winner changed between the sklearn
SGDClassifier backend (paper / ``self_contained_v2``: a RAIL variant leads)
and the NumPy ``--fast`` backend (``self_contained_v3`` + regime sweep: ITLM
leads and wins all 24 regime cells). Because both runs differed in *seeds and
backend at once*, this script isolates the backend: it replays the **same
seeds** through the **same pipeline** under both backends and prints a
side-by-side AE comparison, then states whether the winner flips.

This is the experiment that settles which backend is canonical. Run it where
you have real cores (it reuses the full pipeline, including the integrated
baselines such as ITLM and Co-Teaching).

Usage
-----
    # Quick directional check (1 dataset, few seeds, short replay):
    python -m experiments.diagnose_backend --seeds 5 --datasets Synthetic --max-events 800

    # Fuller check across all real datasets:
    python -m experiments.diagnose_backend --seeds 10 --max-events 1500

Notes
-----
* sklearn (slow) is run first; the fast NumPy classifier is then monkey-patched
  in for the second pass, so both passes live in one process and share seeds.
* ``--max-events`` truncates each dataset's replay stream so sklearn stays
  tractable; the truncation is identical across backends, so the comparison
  remains fair. Drop it for the full stream.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

try:
    from . import baselines_integrated, rail_paper, replay_integrated
except ImportError:  # script-style fallback
    import baselines_integrated  # type: ignore[no-redef]
    import rail_paper  # type: ignore[no-redef]
    import replay_integrated  # type: ignore[no-redef]

KEY = ["rail_gated", "rail_weighted", "itlm", "dynamic_quantile", "self_paced", "static", "always"]


def _truncate(bundle, max_events: int | None):
    if not max_events:
        return bundle.replay_events
    ev = bundle.replay_events
    try:
        return ev[:max_events]
    except TypeError:
        return list(ev)[:max_events]


def _run_one(bundle, seed: int, max_events: int | None, legacy_policies):
    """One (dataset, seed): legacy + integrated rows. ClassifierCls resolved live
    so a prior monkey-patch (fast backend) is picked up."""
    ClassifierCls = rail_paper.SklearnOnlineClassifier
    classes = sorted(set(bundle.y_warmup.tolist()))
    model = ClassifierCls(classes=classes, random_state=seed)
    model.fit_initial(bundle.X_warmup, bundle.y_warmup)
    calibrated = rail_paper.calibrate_gated_baselines_to_rail_yield(
        base_model=model,
        validation_events=bundle.validation_events,
        policies=legacy_policies,
    )
    events = _truncate(bundle, max_events)
    legacy_rows = rail_paper.run_all_policies_once(
        dataset_name=bundle.name,
        base_model=model,
        events=events,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        policies=calibrated,
        run_id=seed,
    )
    always = next(r for r in legacy_rows if r.method == "always")
    ref = (always.contaminated_admissions, always.admitted_feedback)
    integ_policies = baselines_integrated.make_integrated_policies(seed=seed)
    integ_rows = replay_integrated.run_all_integrated_policies(
        dataset_name=bundle.name,
        base_model=model,
        events=events,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        policies=integ_policies,
        always_reference_counts=ref,
        run_id=seed,
    )
    return list(legacy_rows) + list(integ_rows)


def _backend_pass(seeds, dataset_names, max_events):
    """Return {dataset: {method: mean_ae}} for the current backend."""
    datasets = rail_paper.build_all_datasets(seed=seeds[0])
    if dataset_names:
        wanted = set(dataset_names)
        datasets = [b for b in datasets if b.name in wanted]
        if not datasets:
            avail = [b.name for b in rail_paper.build_all_datasets(seed=seeds[0])]
            raise SystemExit(f"No datasets matched {dataset_names}. Available: {avail}")
    legacy_policies = rail_paper.make_default_policies()
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for b in datasets:
        for s in seeds:
            for row in _run_one(b, int(s), max_events, legacy_policies):
                acc[b.name][row.method].append(float(row.ae))
    return {ds: {m: sum(v) / len(v) for m, v in meth.items()} for ds, meth in acc.items()}


def _avg_ranks(per_ds: dict[str, dict[str, float]]):
    methods = sorted({m for d in per_ds.values() for m in d})
    rsum = {m: 0.0 for m in methods}
    n = 0
    for _ds, scores in per_ds.items():
        present = [(m, scores[m]) for m in methods if m in scores]
        present.sort(key=lambda kv: -kv[1])  # higher AE = better = rank 1
        for m, v in present:
            better = sum(1 for _, v2 in present if v2 > v + 1e-12)
            ties = sum(1 for _, v2 in present if abs(v2 - v) < 1e-12)
            rsum[m] += 1 + better + (ties - 1) / 2
        n += 1
    return {m: rsum[m] / max(n, 1) for m in methods}


def _fmt_table(slow, fast, dataset_names):
    dss = dataset_names or sorted(set(slow) | set(fast))
    methods = sorted({m for d in (*slow.values(), *fast.values()) for m in d})
    methods = [m for m in KEY if m in methods] + [m for m in methods if m not in KEY]
    lines = []
    header = (
        f"{'method':<18}"
        + "".join(f"{d[:10]:>12}" for d in dss)
        + f"{'rank(slow)':>12}{'rank(fast)':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    rs, rf = _avg_ranks(slow), _avg_ranks(fast)
    for m in methods:
        cells = ""
        for d in dss:
            sv = slow.get(d, {}).get(m)
            fv = fast.get(d, {}).get(m)
            sv = f"{sv:.3f}" if sv is not None else "-"
            fv = f"{fv:.3f}" if fv is not None else "-"
            cells += f"{sv + '/' + fv:>12}"
        lines.append(
            f"{m:<18}{cells}{rs.get(m, float('nan')):>12.2f}{rf.get(m, float('nan')):>12.2f}"
        )
    return "\n".join(lines), rs, rf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seeds", type=int, default=5, help="Number of seeds (from SEED_MANIFEST order)."
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names to include (default: all built). e.g. Synthetic ATC-like",
    )
    ap.add_argument(
        "--max-events",
        type=int,
        default=1000,
        help="Truncate each replay stream to this many events (0 = full).",
    )
    args = ap.parse_args(argv)

    import json
    from pathlib import Path

    manifest = json.loads((Path(__file__).resolve().parents[1] / "SEED_MANIFEST.json").read_text())
    all_seeds = manifest.get("seeds") or manifest.get("headline") or list(range(args.seeds))
    if isinstance(all_seeds, dict):
        all_seeds = next(iter(all_seeds.values()))
    seeds = [int(s) for s in all_seeds[: args.seeds]] if all_seeds else list(range(args.seeds))
    max_events = args.max_events or None

    print(f"[diagnose] seeds={seeds}")
    print(f"[diagnose] datasets={args.datasets or 'ALL'}  max_events={max_events}")

    # Pass 1: sklearn (canonical). Resolve the original class first.
    print("\n[diagnose] === sklearn (slow) pass ===")
    slow = _backend_pass(seeds, args.datasets, max_events)

    # Pass 2: install the fast NumPy backend, then repeat.
    print("[diagnose] === NumPy (fast) pass ===")
    try:
        from .fast_classifier import install_fast_classifier_into_rail_paper
    except ImportError:
        from fast_classifier import install_fast_classifier_into_rail_paper  # type: ignore
    install_fast_classifier_into_rail_paper()
    fast = _backend_pass(seeds, args.datasets, max_events)

    table, rs, rf = _fmt_table(slow, fast, args.datasets)
    print("\n=== Admission-Efficiency per dataset (slow/fast) and average rank ===")
    print(table)

    slow_winner = min(rs, key=rs.get) if rs else None
    fast_winner = min(rf, key=rf.get) if rf else None
    print("\n=== VERDICT ===")
    print(
        f"  best AE rank under sklearn (slow): {slow_winner}  ({rs.get(slow_winner, float('nan')):.2f})"
    )
    print(
        f"  best AE rank under NumPy  (fast):  {fast_winner}  ({rf.get(fast_winner, float('nan')):.2f})"
    )
    if slow_winner != fast_winner:
        print("  >> WINNER FLIPS between backends. The headline result is backend-dependent;")
        print("     regenerate ALL paper artifacts from the sklearn (--slow) backend.")
    else:
        print("  >> Winner is stable across backends at this scope. Increase --seeds /")
        print("     --max-events to confirm before trusting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
