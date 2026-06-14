"""Journal-grade main-text figures for the RAIL submission (Information Systems, Elsevier).

Produces the two figures referenced in the main body of ``paper/name.tex``:

* ``fig_pareto_yield_vs_contamination`` -- yield vs. per-admission contamination
  across all methods, with the non-dominated (lower-left) frontier highlighted.
* ``fig_contamination_prevention`` -- contamination prevented relative to the
  Unfiltered reference, per method per dataset, with paired 30-seed error bars.

Design follows conservative Elsevier conventions:
* No in-figure titles -- the LaTeX caption carries the description.
* Serif typography (Computer Modern / Times) matching elsarticle body text.
* Wong (Nature Methods 2011) colour-blind-safe palette, redundant marker shapes.
* Neutral, uniform-weight labels -- no coloured or bold method names.
* RAIL emphasis through marker size/edge only, never through colour of text.
* Vector PDF masters with embedded TrueType fonts (pdf.fonttype = 42).

Run::

    python -m experiments.make_main_figures --out publication_outputs/main_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style.
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "STIXGeneral",
            "Computer Modern Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "legend.frameon": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
    }
)

# Wong (2011) colour-blind-safe palette.
WONG = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermil": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "skyblue": "#56B4E9",
    "black": "#1A1A1A",
    "grey": "#7F7F7F",
    "lgrey": "#C9C9C9",
}
INK = "#222222"  # neutral text / axis ink
GRID = "#B5B5B5"  # subtle gridlines

# Family-grouped (colour, marker). Within a family, marker shape disambiguates.
METHOD_STYLE = {
    "static": (WONG["lgrey"], "X"),
    "always": (WONG["black"], "P"),
    "confidence_gated": (WONG["yellow"], "v"),
    "loss_gated": (WONG["orange"], "^"),
    "margin_gated": (WONG["purple"], "D"),
    "co_teaching": (WONG["skyblue"], "o"),
    "self_paced": (WONG["skyblue"], "s"),
    "joint_agreement": (WONG["skyblue"], "p"),
    "dynamic_quantile": (WONG["green"], "h"),
    "gce_weight": (WONG["green"], "<"),
    "sce_weight": (WONG["green"], ">"),
    "itlm": (WONG["green"], "*"),
    "rail_gated": (WONG["vermil"], "*"),
    "rail_weighted": (WONG["blue"], "*"),
}

METHOD_LABEL = {
    "static": "Static",
    "always": "Unfiltered",
    "confidence_gated": "Confidence",
    "loss_gated": "Loss",
    "margin_gated": "Margin",
    "co_teaching": "Co-Teaching",
    "self_paced": "Self-Paced",
    "joint_agreement": "Joint-Agree.",
    "dynamic_quantile": "Dyn-Quantile",
    "gce_weight": "GCE",
    "sce_weight": "SCE",
    "itlm": "ITLM",
    "rail_gated": "RAIL-Gated",
    "rail_weighted": "RAIL-Weighted",
}

METHOD_ORDER = [
    "static",
    "always",
    "confidence_gated",
    "loss_gated",
    "margin_gated",
    "co_teaching",
    "self_paced",
    "joint_agreement",
    "dynamic_quantile",
    "gce_weight",
    "sce_weight",
    "itlm",
    "rail_gated",
    "rail_weighted",
]

DATASET_PRETTY = {
    "Synthetic": "Synthetic",
    "SECOM-like": "SECOM",
    "APS-like": "APS Failure",
    "ATC-like": "ATC",
}
DATASET_ORDER = ["Synthetic", "SECOM-like", "APS-like", "ATC-like"]

RAIL = {"rail_gated", "rail_weighted"}


def _is_rail(m: str) -> bool:
    return m in RAIL


def _style_axis(ax) -> None:
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(colors=INK, labelcolor=INK)


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{fmt}"), format=fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: yield vs. contamination Pareto (2 x 2 panels, real headline data).
# ---------------------------------------------------------------------------


def _pareto_points(run: pd.DataFrame, ds: str):
    sub = run[run["dataset"] == ds]
    rows = []
    for m in METHOD_ORDER:
        mm = sub[sub["method"] == m]
        if mm.empty:
            continue
        yld = mm["admitted_yield"].mean()
        adm = mm["admitted_feedback"].sum()
        con = mm["contaminated_admissions"].sum()
        rate = con / max(adm, 1)
        rows.append((m, float(yld), float(rate)))
    return rows


def fig_pareto(stem: Path, run: pd.DataFrame) -> None:
    datasets = [d for d in DATASET_ORDER if d in run["dataset"].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.6), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, ds in zip(axes, datasets, strict=False):
        pts = _pareto_points(run, ds)
        ordered = [p for p in pts if not _is_rail(p[0])] + [p for p in pts if _is_rail(p[0])]
        for m, yld, rate in ordered:
            color, marker = METHOD_STYLE.get(m, (WONG["grey"], "o"))
            if _is_rail(m):
                ax.scatter(
                    yld,
                    rate,
                    s=135,
                    marker=marker,
                    facecolor=color,
                    edgecolor=INK,
                    linewidths=0.9,
                    zorder=6,
                )
            else:
                ax.scatter(
                    yld,
                    rate,
                    s=46,
                    marker=marker,
                    facecolor=color,
                    edgecolor="white",
                    linewidths=0.6,
                    zorder=3,
                )
        # Non-dominated frontier (lower-right preferred): high yield, low contamination.
        non_dom = []
        for m, yld, rate in pts:
            dominated = any(
                (m2 != m)
                and (y2 + 1e-9 >= yld)
                and (r2 <= rate + 1e-9)
                and ((y2 > yld + 1e-9) or (r2 < rate - 1e-9))
                for m2, y2, r2 in pts
            )
            if not dominated:
                non_dom.append((m, yld, rate))
        env = sorted(non_dom, key=lambda kv: kv[1])
        if len(env) >= 2:
            ax.plot(
                [p[1] for p in env],
                [p[2] for p in env],
                "-",
                color=INK,
                lw=1.0,
                alpha=0.55,
                zorder=2,
                solid_joinstyle="round",
                solid_capstyle="round",
            )

        ax.text(
            0.04,
            0.93,
            DATASET_PRETTY.get(ds, ds),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            color=INK,
        )
        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(-0.015, 0.42)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5)
        _style_axis(ax)

        # Directional cue: the desirable corner is high yield, low contamination.
        if ds == datasets[0]:
            ax.annotate(
                "better",
                xy=(1.02, 0.012),
                xytext=(0.66, 0.135),
                ha="center",
                va="center",
                fontsize=7.0,
                color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8, alpha=0.8),
            )

    for ax in axes[2:]:
        ax.set_xlabel("Admitted yield")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Per-admission\ncontamination rate")

    handles = []
    for m in METHOD_ORDER:
        color, marker = METHOD_STYLE.get(m, (WONG["grey"], "o"))
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="none",
                marker=marker,
                markerfacecolor=color,
                markeredgecolor=INK if _is_rail(m) else "white",
                markeredgewidth=0.7,
                markersize=9 if _is_rail(m) else 6.5,
                label=METHOD_LABEL.get(m, m),
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.11),
        frameon=False,
        fontsize=7.6,
        handletextpad=0.4,
        columnspacing=1.1,
        labelcolor=INK,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Figure: contamination prevented vs. Unfiltered (grouped bars, paired errors).
# ---------------------------------------------------------------------------

PREVENT_METHODS = ["confidence_gated", "loss_gated", "margin_gated", "rail_gated", "rail_weighted"]


def _prevention_stats(run: pd.DataFrame, ds: str):
    """Paired per-seed contamination prevented (%) relative to Unfiltered."""
    sub = run[run["dataset"] == ds]
    base = sub[sub["method"] == "always"].set_index("run_id")["contaminated_admissions"]
    out = {}
    for m in PREVENT_METHODS:
        mm = sub[sub["method"] == m].set_index("run_id")["contaminated_admissions"]
        common = base.index.intersection(mm.index)
        b = base.loc[common].to_numpy(dtype=float)
        v = mm.loc[common].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            prevented = np.where(b > 0, (b - v) / b * 100.0, np.nan)
        prevented = prevented[np.isfinite(prevented)]
        out[m] = (float(np.mean(prevented)), float(np.std(prevented, ddof=1)))
    return out


def fig_contamination_prevention(stem: Path, run: pd.DataFrame) -> None:
    datasets = [d for d in DATASET_ORDER if d in run["dataset"].unique()]
    stats = {ds: _prevention_stats(run, ds) for ds in datasets}

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    n_meth = len(PREVENT_METHODS)
    group_w = 0.82
    bar_w = group_w / n_meth
    x = np.arange(len(datasets))

    for j, m in enumerate(PREVENT_METHODS):
        color, _ = METHOD_STYLE.get(m, (WONG["grey"], "o"))
        means = [stats[ds][m][0] for ds in datasets]
        errs = [stats[ds][m][1] for ds in datasets]
        offset = (j - (n_meth - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            means,
            width=bar_w * 0.92,
            color=color,
            edgecolor=INK,
            linewidth=0.5,
            label=METHOD_LABEL[m],
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=errs,
            fmt="none",
            ecolor=INK,
            elinewidth=0.7,
            capsize=1.8,
            capthick=0.7,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_PRETTY.get(d, d) for d in datasets])
    ax.set_ylabel("Contamination prevented vs.\nUnfiltered admission (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5)
    _style_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=n_meth,
        frameon=False,
        fontsize=7.8,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.0,
        labelcolor=INK,
    )
    fig.tight_layout()
    _save(fig, stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="publication_outputs/self_contained_v3")
    ap.add_argument("--out", default="publication_outputs/main_figures")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    run = pd.read_csv(root / args.data / "run_metrics.csv")
    run = run[run["method"].isin(METHOD_ORDER)]
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    fig_pareto(out / "fig_pareto_yield_vs_contamination", run)
    print("  - fig_pareto_yield_vs_contamination.{pdf,png}")
    fig_contamination_prevention(out / "fig_contamination_prevention", run)
    print("  - fig_contamination_prevention.{pdf,png}")
    print(f"[main-figs] written to {out}")


if __name__ == "__main__":
    main()
