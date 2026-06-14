"""Regime characterization sweeps for the RAIL paper.

This module produces the *phase diagrams* that turn aggregate per-dataset
numbers into a precise dominance map: for each (noise rate, operator skill,
class prevalence) cell, run every method for K seeds and record the winner
plus its margin over the runner-up.

The output of :func:`run_regime_sweep` is three artefacts:

1. A long-format CSV (one row per cell x method) with mean / std of each
   metric, suitable for direct LaTeX rendering.
2. A wide "winner matrix" CSV (one row per cell, columns = best method by
   each metric, margin, statistical significance flag from paired Wilcoxon
   over the K seeds).
3. A matplotlib heatmap-style phase diagram tagging each cell with the
   winner and shading by margin.

The sweep is built on top of the existing :mod:`experiments.rail_paper`
event simulator so it inherits all the dataset realism (drift, overload,
focus, edits) for free. The only knobs the sweep varies are the operator
*correctness* parameters and the *class prevalence* of the underlying data
generator, which jointly span the regimes the paper claims to characterize.

Quick usage::

    from experiments.regime_sweep import run_regime_sweep, REGIMES_2D_DEFAULT
    df_long, df_winner, fig = run_regime_sweep(
        regimes=REGIMES_2D_DEFAULT,
        seeds=range(30),
        output_dir="publication_outputs/regime",
    )

The sweep deliberately uses a synthetic-but-realistic data factory rather
than the real datasets so the regime axes are controllable. The paper
treats the resulting phase diagrams as a *characterisation* result, with
the real-dataset numbers as point estimates within this regime space.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from .baselines_integrated import (
        StatefulPolicy,
        make_integrated_policies,
    )
    from .rail_paper import (
        DatasetBundle,
        PolicyConfig,
        RunMetrics,
        calibrate_gated_baselines_to_rail_yield,
        make_default_policies,
        make_multiclass_data,
        preprocess_splits,
        run_all_policies_once,
        simulate_events,
    )
    from .rail_stats import paired_wilcoxon
    from .replay_integrated import run_all_integrated_policies
except ImportError:  # pragma: no cover
    from baselines_integrated import (  # type: ignore[no-redef]
        StatefulPolicy,
        make_integrated_policies,
    )
    from rail_paper import (  # type: ignore[no-redef]
        DatasetBundle,
        PolicyConfig,
        RunMetrics,
        calibrate_gated_baselines_to_rail_yield,
        make_default_policies,
        make_multiclass_data,
        preprocess_splits,
        run_all_policies_once,
        simulate_events,
    )
    from rail_stats import paired_wilcoxon  # type: ignore[no-redef]
    from replay_integrated import (  # type: ignore[no-redef]
        run_all_integrated_policies,
    )


# ---------------------------------------------------------------------------
# Regime specification.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeCell:
    """One cell of the phase diagram.

    Attributes
    ----------
    correct_prob_normal
        P(operator gives correct label | normal workload). Lower = harder regime.
    correct_prob_overload
        P(operator gives correct label | overloaded). Lower still.
    overload_prob
        Fraction of events in the overloaded state. Higher = noisier overall.
    class_imbalance
        Bias on class 0 logits (positive = more class-0 prevalent). 0 = balanced.
    label
        Human-readable cell name, used in CSV / figure titles.
    """

    correct_prob_normal: float
    correct_prob_overload: float
    overload_prob: float
    class_imbalance: float
    label: str = ""

    def key(self) -> str:
        return self.label or (
            f"cpn{self.correct_prob_normal:.2f}_"
            f"cpo{self.correct_prob_overload:.2f}_"
            f"olp{self.overload_prob:.2f}_"
            f"imb{self.class_imbalance:+.2f}"
        )


REGIMES_2D_DEFAULT: list[RegimeCell] = [
    RegimeCell(
        correct_prob_normal=cpn,
        correct_prob_overload=max(0.30, cpn - 0.20),
        overload_prob=olp,
        class_imbalance=0.0,
        label=f"noise{1.0 - cpn:.2f}_load{olp:.2f}",
    )
    for cpn in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
    for olp in (0.10, 0.25, 0.40, 0.55)
]


REGIMES_IMBALANCE: list[RegimeCell] = [
    RegimeCell(
        correct_prob_normal=0.85,
        correct_prob_overload=0.65,
        overload_prob=0.35,
        class_imbalance=imb,
        label=f"imb{imb:+.2f}",
    )
    for imb in (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
]


# ---------------------------------------------------------------------------
# Per-cell dataset factory.
# ---------------------------------------------------------------------------


def _bundle_for_cell(
    cell: RegimeCell,
    seed: int,
    n_replay_events: int = 2400,
    n_warmup: int = 500,
    n_val: int = 250,
    n_test: int = 600,
) -> DatasetBundle:
    """Build a synthetic ``DatasetBundle`` parameterised by the regime cell.

    Sizes are configurable so the regime sweep can run on a smaller stream
    when its purpose is *winner detection* rather than absolute performance
    estimation.
    """

    rng = np.random.default_rng(seed)
    n_features = 12
    n_classes = 4
    bias = np.array([cell.class_imbalance, -0.1, 0.0, 0.0])

    X_warmup, y_warmup = make_multiclass_data(
        rng, n_warmup, n_features, n_classes, 0.0, bias, 0.9, 1.0
    )
    X_val, y_val = make_multiclass_data(rng, n_val, n_features, n_classes, 0.25, bias, 0.9, 1.0)
    X_rep, y_rep = make_multiclass_data(
        rng, n_replay_events, n_features, n_classes, 0.55, bias, 0.9, 1.0
    )
    X_test, y_test = make_multiclass_data(rng, n_test, n_features, n_classes, 0.55, bias, 0.9, 1.0)
    X_warmup, X_val, X_rep, X_test = preprocess_splits(X_warmup, X_val, X_rep, X_test)

    def _events(X, y):
        return simulate_events(
            X,
            y,
            n_classes,
            rng,
            overload_prob=cell.overload_prob,
            correct_prob_normal=cell.correct_prob_normal,
            correct_prob_overload=cell.correct_prob_overload,
            fast_band=(0.1, 0.8),
            good_band=(1.2, 5.5),
            slow_band=(7.0, 12.0),
            focus_normal=(1.5, 4.0),
            focus_overload=(0.1, 1.3),
            edits_normal=(1, 3),
            edits_overload=(0, 1),
        )

    return DatasetBundle(
        name=cell.key(),
        X_warmup=X_warmup,
        y_warmup=y_warmup,
        validation_events=_events(X_val, y_val),
        replay_events=_events(X_rep, y_rep),
        X_test=X_test,
        y_test=y_test,
    )


# ---------------------------------------------------------------------------
# Sweep runner.
# ---------------------------------------------------------------------------


@dataclass
class CellResults:
    cell: RegimeCell
    per_method_runs: dict[str, list[RunMetrics]] = field(default_factory=dict)

    def mean_metric(self, method: str, metric: str) -> float:
        rows = self.per_method_runs.get(method, [])
        return statistics.mean(getattr(r, metric) for r in rows) if rows else float("nan")

    def std_metric(self, method: str, metric: str) -> float:
        rows = self.per_method_runs.get(method, [])
        if len(rows) < 2:
            return 0.0
        return statistics.stdev(getattr(r, metric) for r in rows)


def _run_one_seed(
    cell: RegimeCell,
    seed: int,
    policies: Sequence[PolicyConfig],
    integrated_factory: Callable[[int], Sequence[StatefulPolicy]],
    n_replay_events: int = 2400,
    use_fast: bool = False,
) -> list[RunMetrics]:
    """One (cell, seed) replay -- returns all RunMetrics for both legacy and
    integrated policies. Independent of all other seeds, so safe to run in
    parallel.

    When ``use_fast`` is True we install the NumpyOnlineClassifier
    monkey-patch inside the worker process (the patch made in the parent
    does not survive ``loky``'s fresh imports).
    """

    if use_fast:
        try:
            from .fast_classifier import install_fast_classifier_into_rail_paper
        except ImportError:  # pragma: no cover
            from fast_classifier import (
                install_fast_classifier_into_rail_paper,  # type: ignore[no-redef]
            )
        install_fast_classifier_into_rail_paper()

    # Re-resolve SklearnOnlineClassifier from rail_paper at call time so the
    # monkey-patch above is honoured.
    try:
        from . import rail_paper as _rp
    except ImportError:  # pragma: no cover
        import rail_paper as _rp  # type: ignore[no-redef]
    ClassifierCls = _rp.SklearnOnlineClassifier

    bundle = _bundle_for_cell(cell, seed, n_replay_events=n_replay_events)
    classes = sorted(set(bundle.y_warmup.tolist()))
    model = ClassifierCls(classes=classes, random_state=seed)
    model.fit_initial(bundle.X_warmup, bundle.y_warmup)
    calibrated = calibrate_gated_baselines_to_rail_yield(
        base_model=model,
        validation_events=bundle.validation_events,
        policies=policies,
    )
    legacy_rows = run_all_policies_once(
        dataset_name=cell.key(),
        base_model=model,
        events=bundle.replay_events,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        policies=calibrated,
        run_id=seed,
    )
    always = next(r for r in legacy_rows if r.method == "always")
    ref = (always.contaminated_admissions, always.admitted_feedback)
    # Reinstantiate integrated policies per seed -- they hold mutable state
    # (sliding-window quantiles, age counters) that must NOT carry over.
    integ_policies = list(integrated_factory(seed))
    integ_rows = run_all_integrated_policies(
        dataset_name=cell.key(),
        base_model=model,
        events=bundle.replay_events,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        policies=integ_policies,
        always_reference_counts=ref,
        run_id=seed,
    )
    return list(legacy_rows) + list(integ_rows)


def _run_one_cell(
    cell: RegimeCell,
    seeds: Sequence[int],
    policies: Sequence[PolicyConfig],
    integrated_factory: Callable[[int], Sequence[StatefulPolicy]],
    n_replay_events: int = 2400,
) -> CellResults:
    results = CellResults(cell=cell)
    for seed in seeds:
        rows = _run_one_seed(
            cell,
            seed,
            policies,
            integrated_factory,
            n_replay_events,
        )
        for row in rows:
            results.per_method_runs.setdefault(row.method, []).append(row)
    return results


def run_regime_sweep(
    regimes: Sequence[RegimeCell] = REGIMES_2D_DEFAULT,
    seeds: Sequence[int] = range(30),
    output_dir: str | Path = "publication_outputs/regime",
    policies: Sequence[PolicyConfig] | None = None,
    integrated_factory: Callable[[int], Sequence[StatefulPolicy]] | None = None,
    metric: str = "ae",
    higher_is_better: bool = True,
    n_workers: int = 1,
    n_replay_events: int = 2400,
    verbose: bool = True,
    use_fast: bool = False,
) -> tuple[list[CellResults], Path]:
    """Run the full regime sweep, persist artefacts, return ``(results, dir)``.

    Parameters
    ----------
    n_workers
        If > 1, run (cell, seed) pairs in parallel via joblib. Each worker
        is a separate process so this scales linearly until I/O-bound.
    n_replay_events
        Number of replay events per (cell, seed). 2400 reproduces the
        headline numbers; 1200 is fine for the regime sweep where the
        question is *winner detection* not absolute performance.
    integrated_factory
        Callable ``seed -> [StatefulPolicy]``. Reinstantiated per seed so
        mutable per-event state does not leak across replays. Defaults to
        :func:`baselines_integrated.make_integrated_policies`.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if policies is None:
        policies = make_default_policies()
    if integrated_factory is None:

        def integrated_factory(s):
            return make_integrated_policies(seed=int(s))

    pairs: list[tuple[RegimeCell, int]] = [(c, int(s)) for c in regimes for s in seeds]
    if verbose:
        print(
            f"[regime] {len(regimes)} cells x {len(list(seeds))} seeds "
            f"= {len(pairs)} runs, workers={n_workers}, "
            f"n_replay_events={n_replay_events}"
        )

    if n_workers > 1:
        try:
            from joblib import Parallel, delayed
        except ImportError:  # pragma: no cover
            print("[regime] joblib not available; falling back to serial")
            n_workers = 1

    if n_workers > 1:
        rows_by_pair: list[list[RunMetrics]] = Parallel(
            n_jobs=n_workers,
            verbose=10 if verbose else 0,
            backend="loky",
        )(
            delayed(_run_one_seed)(
                cell,
                seed,
                list(policies),
                integrated_factory,
                n_replay_events,
                use_fast,
            )
            for cell, seed in pairs
        )
    else:
        # Install once in the parent for the serial path.
        if use_fast:
            try:
                from .fast_classifier import install_fast_classifier_into_rail_paper
            except ImportError:  # pragma: no cover
                from fast_classifier import (
                    install_fast_classifier_into_rail_paper,  # type: ignore[no-redef]
                )
            install_fast_classifier_into_rail_paper()
        rows_by_pair = []
        for i, (cell, seed) in enumerate(pairs):
            if verbose and i % max(1, len(pairs) // 20) == 0:
                print(f"[regime] {i}/{len(pairs)} ...")
            rows_by_pair.append(
                _run_one_seed(
                    cell,
                    seed,
                    list(policies),
                    integrated_factory,
                    n_replay_events,
                    use_fast,
                )
            )

    # Aggregate by cell.
    by_cell: dict[str, CellResults] = {c.key(): CellResults(cell=c) for c in regimes}
    for (cell, _seed), rows in zip(pairs, rows_by_pair, strict=False):
        cr = by_cell[cell.key()]
        for row in rows:
            cr.per_method_runs.setdefault(row.method, []).append(row)
    all_results: list[CellResults] = [by_cell[c.key()] for c in regimes]

    _write_long_csv(all_results, out_dir / "regime_long.csv", metric=metric)
    _write_winner_csv(
        all_results,
        out_dir / "regime_winners.csv",
        metric=metric,
        higher_is_better=higher_is_better,
    )
    return all_results, out_dir


# ---------------------------------------------------------------------------
# Artefact writers.
# ---------------------------------------------------------------------------


def _write_long_csv(results: Sequence[CellResults], path: Path, metric: str) -> None:
    fieldnames = [
        "cell",
        "correct_prob_normal",
        "correct_prob_overload",
        "overload_prob",
        "class_imbalance",
        "method",
        f"{metric}_mean",
        f"{metric}_std",
        "macro_f1_mean",
        "macro_f1_std",
        "contaminated_admissions_mean",
        "admitted_yield_mean",
        "n_seeds",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for cr in results:
            cell = cr.cell
            for method, runs in cr.per_method_runs.items():
                w.writerow(
                    {
                        "cell": cell.key(),
                        "correct_prob_normal": cell.correct_prob_normal,
                        "correct_prob_overload": cell.correct_prob_overload,
                        "overload_prob": cell.overload_prob,
                        "class_imbalance": cell.class_imbalance,
                        "method": method,
                        f"{metric}_mean": cr.mean_metric(method, metric),
                        f"{metric}_std": cr.std_metric(method, metric),
                        "macro_f1_mean": cr.mean_metric(method, "final_macro_f1"),
                        "macro_f1_std": cr.std_metric(method, "final_macro_f1"),
                        "contaminated_admissions_mean": cr.mean_metric(
                            method, "contaminated_admissions"
                        ),
                        "admitted_yield_mean": cr.mean_metric(method, "admitted_yield"),
                        "n_seeds": len(runs),
                    }
                )


def _write_winner_csv(
    results: Sequence[CellResults],
    path: Path,
    metric: str,
    higher_is_better: bool = True,
) -> None:
    fieldnames = [
        "cell",
        "correct_prob_normal",
        "overload_prob",
        "class_imbalance",
        "winner",
        "winner_mean",
        "runner_up",
        "runner_up_mean",
        "margin",
        "wilcoxon_p",
        "significant_at_0_05",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for cr in results:
            cell = cr.cell
            methods = sorted(cr.per_method_runs.keys())
            if not methods:
                continue
            means = {m: cr.mean_metric(m, metric) for m in methods}
            sign = 1.0 if higher_is_better else -1.0
            sorted_m = sorted(methods, key=lambda m: -sign * means[m])
            winner = sorted_m[0]
            runner = sorted_m[1] if len(sorted_m) > 1 else winner
            a_vals = [getattr(r, metric) for r in cr.per_method_runs[winner]]
            b_vals = [getattr(r, metric) for r in cr.per_method_runs[runner]]
            try:
                res = paired_wilcoxon(a_vals, b_vals)
                if isinstance(res, dict):
                    p_val = float(res.get("p_value", float("nan")))
                else:  # pragma: no cover - backwards-compat
                    p_val = float(res[1])
            except Exception:  # pragma: no cover
                p_val = float("nan")
            w.writerow(
                {
                    "cell": cell.key(),
                    "correct_prob_normal": cell.correct_prob_normal,
                    "overload_prob": cell.overload_prob,
                    "class_imbalance": cell.class_imbalance,
                    "winner": winner,
                    "winner_mean": means[winner],
                    "runner_up": runner,
                    "runner_up_mean": means[runner],
                    "margin": sign * (means[winner] - means[runner]),
                    "wilcoxon_p": p_val,
                    "significant_at_0_05": (
                        "1" if (not math.isnan(p_val) and p_val < 0.05) else "0"
                    ),
                }
            )


# ---------------------------------------------------------------------------
# Heatmap phase diagram.
# ---------------------------------------------------------------------------


def phase_diagram(
    results: Sequence[CellResults],
    x_attr: str = "overload_prob",
    y_attr: str = "correct_prob_normal",
    metric: str = "ae",
    higher_is_better: bool = True,
    output_path: str | None = None,
):  # pragma: no cover - matplotlib I/O
    """Render a 2-D phase diagram of cell winners."""

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    xs = sorted({getattr(cr.cell, x_attr) for cr in results})
    ys = sorted({getattr(cr.cell, y_attr) for cr in results})
    if not xs or not ys:
        raise ValueError("Empty sweep")

    method_to_color: dict[str, str] = {}
    palette = list(mcolors.TABLEAU_COLORS.values())
    cells: dict[tuple[float, float], CellResults] = {}
    for cr in results:
        cells[(getattr(cr.cell, x_attr), getattr(cr.cell, y_attr))] = cr

    fig, ax = plt.subplots(figsize=(1.6 * len(xs) + 2.0, 1.0 * len(ys) + 1.5))
    sign = 1.0 if higher_is_better else -1.0

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            cr = cells.get((x, y))
            if cr is None:
                continue
            methods = sorted(cr.per_method_runs.keys())
            means = {m: cr.mean_metric(m, metric) for m in methods}
            sorted_m = sorted(methods, key=lambda m: -sign * means[m])
            winner = sorted_m[0]
            margin = sign * (means[winner] - means[sorted_m[1]] if len(sorted_m) > 1 else 0.0)
            if winner not in method_to_color:
                method_to_color[winner] = palette[len(method_to_color) % len(palette)]
            color = method_to_color[winner]
            ax.add_patch(plt.Rectangle((i, j), 1, 1, facecolor=color, edgecolor="white"))
            ax.text(
                i + 0.5,
                j + 0.55,
                winner.replace("_", "\n"),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )
            ax.text(
                i + 0.5,
                j + 0.18,
                f"+{margin:.3f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )

    ax.set_xlim(0, len(xs))
    ax.set_ylim(0, len(ys))
    ax.set_xticks([i + 0.5 for i in range(len(xs))])
    ax.set_xticklabels([f"{v:.2f}" for v in xs])
    ax.set_yticks([j + 0.5 for j in range(len(ys))])
    ax.set_yticklabels([f"{v:.2f}" for v in ys])
    ax.set_xlabel(x_attr)
    ax.set_ylabel(y_attr)
    ax.set_title(f"Phase diagram of winners by {metric}")
    ax.set_aspect("auto")
    if output_path is not None:
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
    return fig


__all__ = [
    "REGIMES_2D_DEFAULT",
    "REGIMES_IMBALANCE",
    "CellResults",
    "RegimeCell",
    "phase_diagram",
    "run_regime_sweep",
]
