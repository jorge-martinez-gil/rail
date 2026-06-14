"""Restrained, journal-format empirical figures for the RAIL manuscript.

The design is intentionally conservative: sans-serif typography, one muted
accent colour, no in-figure headlines, and uncertainty shown from matched
random seeds wherever seed-level data are available.

Run::

    python -m experiments.make_journal_figures \
        --runs publication_outputs/self_contained_v3/run_metrics.csv \
        --regimes publication_outputs/regime/regime_long.csv \
        --out publication_outputs/journal_figures
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")


NAVY = "#264F73"
CHARCOAL = "#242424"
MID_GREY = "#777777"
LIGHT_GREY = "#B8B8B8"
PALE_GREY = "#E6E6E6"
WARM_GREY = "#8C6D62"

DATASETS = ["Synthetic", "SECOM-like", "APS-like", "ATC-like"]
COMPARATORS = [
    "confidence_gated",
    "loss_gated",
    "margin_gated",
    "dynamic_quantile",
]
COMPARATOR_LABEL = {
    "confidence_gated": "Confidence",
    "loss_gated": "Loss",
    "margin_gated": "Margin",
    "dynamic_quantile": "Dynamic-quantile",
}
METHOD_LABEL = {
    "always": "Unfiltered",
    "co_teaching": "Co-teaching",
    "confidence_gated": "Confidence",
    "dynamic_quantile": "Dynamic-quantile",
    "gce_weight": "GCE",
    "itlm": "ITLM",
    "joint_agreement": "Joint-agreement",
    "loss_gated": "Loss",
    "margin_gated": "Margin",
    "rail_gated": "RAIL",
    "rail_weighted": "RAIL-weighted",
    "sce_weight": "SCE",
    "self_paced": "Self-paced",
    "static": "Static",
}

RUN_COLUMNS = {
    "dataset",
    "method",
    "run_id",
    "final_macro_f1",
    "contaminated_admissions",
    "admitted_feedback",
    "admitted_yield",
    "ae",
}
REGIME_COLUMNS = {
    "cell",
    "correct_prob_normal",
    "overload_prob",
    "method",
    "ae_mean",
    "macro_f1_mean",
    "n_seeds",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.7,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260611,
) -> Interval:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    mean = float(array.mean())
    if array.size == 1 or np.allclose(array, array[0]):
        return Interval(mean, mean, mean, int(array.size))
    rng = np.random.default_rng(seed)
    sampled_means = rng.choice(array, size=(n_boot, array.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(sampled_means, [alpha, 1.0 - alpha])
    return Interval(mean, float(low), float(high), int(array.size))


def pareto_frontier(points: Iterable[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """High-yield, low-contamination non-dominated points."""

    rows = list(points)
    frontier = []
    for method, yield_value, contamination in rows:
        dominated = any(
            other_method != method
            and other_yield >= yield_value - 1e-12
            and other_contamination <= contamination + 1e-12
            and (other_yield > yield_value + 1e-12 or other_contamination < contamination - 1e-12)
            for other_method, other_yield, other_contamination in rows
        )
        if not dominated:
            frontier.append((method, yield_value, contamination))
    return sorted(frontier, key=lambda row: row[1])


def load_run_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = RUN_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Run metrics missing columns: {sorted(missing)}")
    if frame.duplicated(["dataset", "method", "run_id"]).any():
        raise ValueError("Run metrics must contain one row per dataset/method/run_id")
    frame = frame.copy()
    frame["contamination_rate"] = np.where(
        frame["admitted_feedback"] > 0,
        frame["contaminated_admissions"] / frame["admitted_feedback"],
        np.nan,
    )
    return frame


def load_regimes(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REGIME_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Regime data missing columns: {sorted(missing)}")
    return frame


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{extension}"), format=extension)
    plt.close(fig)


def _paired_values(
    frame: pd.DataFrame,
    dataset: str,
    method: str,
    comparator: str,
    metric: str,
) -> np.ndarray:
    subset = frame[(frame["dataset"] == dataset) & frame["method"].isin([method, comparator])]
    paired = subset.pivot(index="run_id", columns="method", values=metric).dropna()
    return (paired[method] - paired[comparator]).to_numpy(dtype=float)


def benchmark_contrast_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        if dataset not in set(frame["dataset"]):
            continue
        for comparator in COMPARATORS:
            for metric in ("ae", "final_macro_f1"):
                values = _paired_values(frame, dataset, "rail_gated", comparator, metric)
                estimate = bootstrap_mean_ci(
                    values,
                    seed=abs(hash((dataset, comparator, metric))) % (2**32),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "comparator": comparator,
                        "metric": metric,
                        "mean": estimate.mean,
                        "ci_low": estimate.low,
                        "ci_high": estimate.high,
                        "n": estimate.n,
                    }
                )
    return pd.DataFrame(rows)


def fig_benchmark_contrasts(
    frame: pd.DataFrame,
    stem: Path,
) -> list[dict[str, object]]:
    """Matched-seed RAIL contrasts against score-based ingress gates."""

    datasets = [dataset for dataset in DATASETS if dataset in set(frame["dataset"])]
    row_spec = [(dataset, comparator) for dataset in datasets for comparator in COMPARATORS]
    y_values = np.arange(len(row_spec))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 5.05), sharey=True)
    output_rows: list[dict[str, object]] = []
    metric_specs = [
        ("ae", "Admission Efficiency difference"),
        ("final_macro_f1", "End-of-stream Macro-F1 difference"),
    ]

    jitter_rng = np.random.default_rng(519)
    for panel_index, (ax, (metric, label)) in enumerate(zip(axes, metric_specs, strict=True)):
        for row_index, ((dataset, comparator), y_value) in enumerate(
            zip(row_spec, y_values, strict=True)
        ):
            differences = _paired_values(frame, dataset, "rail_gated", comparator, metric)
            estimate = bootstrap_mean_ci(
                differences,
                seed=10_000 + panel_index * 1_000 + row_index,
            )
            output_rows.append(
                {
                    "figure": "benchmark_contrasts",
                    "dataset": dataset,
                    "comparison": f"rail_gated-minus-{comparator}",
                    "metric": metric,
                    "mean": estimate.mean,
                    "ci_low": estimate.low,
                    "ci_high": estimate.high,
                    "n": estimate.n,
                }
            )

            jitter = jitter_rng.uniform(-0.095, 0.095, size=differences.size)
            ax.scatter(
                differences,
                y_value + jitter,
                s=5.5,
                color=LIGHT_GREY,
                edgecolors="none",
                alpha=0.55,
                rasterized=True,
                zorder=1,
            )
            ax.hlines(y_value, estimate.low, estimate.high, color=NAVY, linewidth=1.25, zorder=3)
            ax.scatter(
                estimate.mean,
                y_value,
                s=17,
                marker="s",
                color=NAVY,
                edgecolors="white",
                linewidths=0.45,
                zorder=4,
            )

        ax.axvline(0.0, color=CHARCOAL, linewidth=0.75, linestyle=(0, (3, 2)), zorder=0)
        ax.set_xlabel(label + "\npositive values favour RAIL")
        ax.grid(axis="x", color=PALE_GREY, linewidth=0.45)
        ax.set_axisbelow(True)
        ax.text(
            -0.08,
            1.015,
            f"({chr(97 + panel_index)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )

    axes[0].set_yticks(y_values)
    axes[0].set_yticklabels([COMPARATOR_LABEL[comparator] for _, comparator in row_spec])
    axes[1].tick_params(axis="y", labelleft=False)

    group_size = len(COMPARATORS)
    for dataset_index, dataset in enumerate(datasets):
        top_index = dataset_index * group_size
        y_centre = (y_values[top_index] + y_values[top_index + group_size - 1]) / 2.0
        axes[0].text(
            -0.39,
            y_centre,
            dataset,
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.0,
            fontweight="semibold",
            clip_on=False,
        )
        if dataset_index < len(datasets) - 1:
            separator = y_values[top_index + group_size - 1] - 0.5
            for ax in axes:
                ax.axhline(separator, color=PALE_GREY, linewidth=0.7, zorder=0)

    raw_handle = mlines.Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=3.2,
        markerfacecolor=LIGHT_GREY,
        markeredgecolor="none",
        label="Matched seeds",
    )
    mean_handle = mlines.Line2D(
        [],
        [],
        color=NAVY,
        linewidth=1.2,
        marker="s",
        markersize=4.0,
        markerfacecolor=NAVY,
        markeredgecolor="white",
        label="Mean and 95% CI",
    )
    axes[1].legend(
        handles=[raw_handle, mean_handle],
        loc="lower right",
        frameon=False,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.27, right=0.99, bottom=0.12, top=0.985, wspace=0.12)
    _save(fig, stem)
    return output_rows


def _mean_operating_points(frame: pd.DataFrame, dataset: str) -> list[tuple[str, float, float]]:
    subset = frame[(frame["dataset"] == dataset) & (frame["admitted_feedback"] > 0)]
    grouped = subset.groupby("method", sort=False).agg(
        admitted_yield=("admitted_yield", "mean"),
        contamination_rate=("contamination_rate", "mean"),
    )
    return [
        (method, float(row.admitted_yield), float(row.contamination_rate))
        for method, row in grouped.iterrows()
    ]


def fig_operating_frontier(frame: pd.DataFrame, stem: Path) -> list[dict[str, object]]:
    """Operating frontier with seed clouds for RAIL and ITLM."""

    datasets = [dataset for dataset in DATASETS if dataset in set(frame["dataset"])]
    fig, axes = plt.subplots(2, 2, figsize=(7.08, 4.35), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    output_rows: list[dict[str, object]] = []

    for panel_index, (ax, dataset) in enumerate(zip(axes, datasets, strict=False)):
        points = _mean_operating_points(frame, dataset)
        frontier = pareto_frontier(points)
        if len(frontier) > 1:
            ax.plot(
                [row[1] for row in frontier],
                [row[2] for row in frontier],
                color=MID_GREY,
                linewidth=0.75,
                zorder=1,
            )

        for method, yield_value, contamination in points:
            if method in {"rail_gated", "rail_weighted", "itlm"}:
                continue
            ax.scatter(
                yield_value,
                contamination,
                s=13,
                facecolors="white",
                edgecolors=LIGHT_GREY,
                linewidths=0.65,
                zorder=2,
            )

        for method, colour, marker in (
            ("rail_gated", NAVY, "o"),
            ("itlm", CHARCOAL, "^"),
        ):
            method_rows = frame[(frame["dataset"] == dataset) & (frame["method"] == method)].dropna(
                subset=["contamination_rate"]
            )
            ax.scatter(
                method_rows["admitted_yield"],
                method_rows["contamination_rate"],
                s=7,
                color=colour,
                edgecolors="none",
                alpha=0.16,
                rasterized=True,
                zorder=3,
            )
            mean_yield = float(method_rows["admitted_yield"].mean())
            mean_contamination = float(method_rows["contamination_rate"].mean())
            ax.scatter(
                mean_yield,
                mean_contamination,
                s=28,
                color=colour,
                marker=marker,
                edgecolors="white",
                linewidths=0.55,
                zorder=5,
            )
            output_rows.append(
                {
                    "figure": "operating_frontier",
                    "dataset": dataset,
                    "comparison": method,
                    "metric": "admitted_yield",
                    "mean": mean_yield,
                    "ci_low": "",
                    "ci_high": "",
                    "n": len(method_rows),
                }
            )
            output_rows.append(
                {
                    "figure": "operating_frontier",
                    "dataset": dataset,
                    "comparison": method,
                    "metric": "contamination_rate",
                    "mean": mean_contamination,
                    "ci_low": "",
                    "ci_high": "",
                    "n": len(method_rows),
                }
            )

        rail_point = next(row for row in points if row[0] == "rail_gated")
        itlm_point = next(row for row in points if row[0] == "itlm")
        ax.annotate(
            "RAIL",
            (rail_point[1], rail_point[2]),
            xytext=(4, 5),
            textcoords="offset points",
            color=NAVY,
            fontsize=6.4,
            fontweight="semibold",
        )
        ax.annotate(
            "ITLM",
            (itlm_point[1], itlm_point[2]),
            xytext=(4, -9),
            textcoords="offset points",
            color=CHARCOAL,
            fontsize=6.4,
        )
        ax.text(
            0.025,
            0.95,
            f"({chr(97 + panel_index)})  {dataset}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            fontweight="semibold",
        )
        ax.grid(color=PALE_GREY, linewidth=0.45)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.025, 1.025)
        ax.set_ylim(-0.003, 0.218)

    for ax in axes[2:]:
        ax.set_xlabel("Admitted yield")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Per-admission contamination")

    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.11, top=0.99, hspace=0.14, wspace=0.10)
    _save(fig, stem)
    return output_rows


def regime_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """RAIL minus the strongest score-based gate in each regime and metric."""

    rail = frame[frame["method"] == "rail_gated"][
        ["cell", "correct_prob_normal", "overload_prob", "ae_mean", "macro_f1_mean", "n_seeds"]
    ].rename(columns={"ae_mean": "rail_ae", "macro_f1_mean": "rail_f1"})
    gates = frame[frame["method"].isin(COMPARATORS)]
    best_ae = (
        gates.sort_values("ae_mean")
        .groupby("cell", as_index=False)
        .tail(1)[["cell", "method", "ae_mean"]]
        .rename(columns={"method": "ae_comparator", "ae_mean": "best_gate_ae"})
    )
    best_f1 = (
        gates.sort_values("macro_f1_mean")
        .groupby("cell", as_index=False)
        .tail(1)[["cell", "method", "macro_f1_mean"]]
        .rename(columns={"method": "f1_comparator", "macro_f1_mean": "best_gate_f1"})
    )
    merged = rail.merge(best_ae, on="cell").merge(best_f1, on="cell")
    merged["ae_difference"] = merged["rail_ae"] - merged["best_gate_ae"]
    merged["f1_difference"] = merged["rail_f1"] - merged["best_gate_f1"]
    return merged


def _grid_from_regimes(
    frame: pd.DataFrame, value_column: str
) -> tuple[np.ndarray, list[float], list[float]]:
    correctness = sorted(frame["correct_prob_normal"].unique())
    overload = sorted(frame["overload_prob"].unique())
    pivot = frame.pivot(
        index="correct_prob_normal", columns="overload_prob", values=value_column
    ).reindex(index=correctness, columns=overload)
    return pivot.to_numpy(dtype=float), correctness, overload


def fig_regime_map(frame: pd.DataFrame, stem: Path) -> list[dict[str, object]]:
    """Regime map of AE gain and Macro-F1 cost versus the best score gate."""

    contrasts = regime_contrasts(frame)
    ae_grid, correctness, overload = _grid_from_regimes(contrasts, "ae_difference")
    f1_grid, _, _ = _grid_from_regimes(contrasts, "f1_difference")

    ae_cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_to_navy", ["#F7F7F7", "#B8C8D6", NAVY]
    )
    f1_cmap = mcolors.LinearSegmentedColormap.from_list(
        "muted_diverging", [WARM_GREY, "#F7F7F7", NAVY]
    )
    f1_limit = max(abs(float(np.nanmin(f1_grid))), abs(float(np.nanmax(f1_grid))))

    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.85))
    images = [
        axes[0].imshow(
            ae_grid,
            origin="lower",
            aspect="auto",
            cmap=ae_cmap,
            vmin=0.0,
            vmax=0.10,
            interpolation="nearest",
        ),
        axes[1].imshow(
            f1_grid,
            origin="lower",
            aspect="auto",
            cmap=f1_cmap,
            vmin=-f1_limit,
            vmax=f1_limit,
            interpolation="nearest",
        ),
    ]
    labels = [
        "Admission Efficiency difference",
        "End-of-stream Macro-F1 difference",
    ]

    for panel_index, (ax, image, label) in enumerate(zip(axes, images, labels, strict=True)):
        ax.set_xticks(np.arange(len(overload)))
        ax.set_xticklabels([f"{value:.2f}" for value in overload])
        ax.set_yticks(np.arange(len(correctness)))
        ax.set_yticklabels([f"{value:.2f}" for value in correctness])
        ax.set_xlabel("Overload probability")
        if panel_index == 0:
            ax.set_ylabel("Operator correctness")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xticks(np.arange(-0.5, len(overload), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(correctness), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.75)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            -0.02,
            1.035,
            f"({chr(97 + panel_index)})",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
        colourbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.025)
        colourbar.set_label(label + "\nRAIL minus best score gate", fontsize=6.7)
        colourbar.ax.tick_params(labelsize=6.3, width=0.5, length=2)
        colourbar.outline.set_linewidth(0.5)

    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.18, top=0.94, wspace=0.28)
    _save(fig, stem)

    output_rows = []
    for row in contrasts.itertuples(index=False):
        output_rows.extend(
            [
                {
                    "figure": "regime_map",
                    "dataset": row.cell,
                    "comparison": f"rail_gated-minus-{row.ae_comparator}",
                    "metric": "ae_mean",
                    "mean": row.ae_difference,
                    "ci_low": "",
                    "ci_high": "",
                    "n": row.n_seeds,
                },
                {
                    "figure": "regime_map",
                    "dataset": row.cell,
                    "comparison": f"rail_gated-minus-{row.f1_comparator}",
                    "metric": "macro_f1_mean",
                    "mean": row.f1_difference,
                    "ci_low": "",
                    "ci_high": "",
                    "n": row.n_seeds,
                },
            ]
        )
    return output_rows


def write_statistics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "figure",
        "dataset",
        "comparison",
        "metric",
        "mean",
        "ci_low",
        "ci_high",
        "n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_captions(path: Path, run_path: Path, regime_path: Path) -> None:
    path.write_text(
        f"""# Suggested captions

Source data: `{run_path.as_posix()}` and `{regime_path.as_posix()}`.

## Benchmark contrasts

**Matched-seed contrasts against score-based ingress gates.** Gray points are
paired differences for each of 30 random seeds. Navy squares and horizontal
segments show the mean paired difference and percentile-bootstrap 95% confidence
interval. Each contrast is RAIL-Gated minus the named comparator; positive values
favour RAIL. RAIL improves Admission Efficiency in nearly every benchmark-gate
comparison, while end-of-stream Macro-F1 effects are smaller and mixed.

## Operating frontier

**Empirical yield-contamination frontier.** Hollow gray points are mean operating
points for the remaining baselines. Small points show individual seed outcomes
for RAIL and ITLM; the larger symbols show their means. The thin gray line joins
the non-dominated empirical operating points. RAIL-Weighted is omitted because it
shares the admission decisions, yield, and contamination rate of RAIL-Gated.
Static is omitted because per-admission contamination is undefined when no
feedback is admitted.

## Regime map

**RAIL relative to the strongest score-based ingress gate across 24 controlled
regimes.** Each cell is a 30-seed mean difference between RAIL-Gated and the
best-performing member of Confidence, Loss, Margin, and Dynamic-Quantile,
selected separately for each metric. Panel (a) shows Admission Efficiency and
panel (b) end-of-stream Macro-F1. Positive values favour RAIL. The map separates
the consistent efficiency gain from the smaller accuracy trade-off.
""",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate journal-format RAIL figures.")
    parser.add_argument(
        "--runs",
        default="publication_outputs/self_contained_v3/run_metrics.csv",
        help="Seed-level benchmark metrics.",
    )
    parser.add_argument(
        "--regimes",
        default="publication_outputs/regime/regime_long.csv",
        help="Long-format controlled regime sweep.",
    )
    parser.add_argument(
        "--out",
        default="publication_outputs/journal_figures",
        help="Output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_path = Path(args.runs)
    regime_path = Path(args.regimes)
    output_dir = Path(args.out)
    runs = load_run_metrics(run_path)
    regimes = load_regimes(regime_path)

    rows: list[dict[str, object]] = []
    rows.extend(fig_benchmark_contrasts(runs, output_dir / "fig_benchmark_contrasts"))
    rows.extend(fig_operating_frontier(runs, output_dir / "fig_operating_frontier"))
    rows.extend(fig_regime_map(regimes, output_dir / "fig_regime_map"))
    write_statistics(output_dir / "figure_statistics.csv", rows)
    write_captions(output_dir / "FIGURE_CAPTIONS.md", run_path, regime_path)
    print(f"Wrote journal figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
