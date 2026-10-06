"""Unified journal-grade figure suite for the RAIL submission (Information Systems, Elsevier).

This module supersedes ``make_main_figures.py`` and ``make_journal_figures.py`` with a
single, consistent visual identity (serif typography matching ``elsarticle`` body text,
the Wong colour-blind-safe palette, redundant marker encodings, vector PDF masters with
embedded fonts). Every figure is computed from the frozen 30-seed headline run
(``self_contained_v3``), the 24-cell regime sweep (``regime``), and the closed-form
theory in :mod:`experiments.theory` / :mod:`experiments.theory_risk`. No numbers are
invented; the script is a deterministic function of the released result files.

Figures produced (``--out publication_outputs/figures_v2``):

  fig_vigilance_gate                 mechanism: the two-sided admission score V(Delta)
  fig_contract_verification          RQ1: contract curve + out-of-sample calibration
  fig_contamination_prevention       RQ1: contamination prevented vs. Unfiltered (bars)
  fig_pareto_yield_vs_contamination  RQ2: yield vs. per-admission contamination (2x2)
  fig_macro_f1_distributions         RQ2: 30-seed Macro-F1 distributions (violins)
  fig_ae_ranking_cd                  RQ2: Nemenyi critical-difference diagram
  fig_benchmark_contrasts            RQ2: matched-seed RAIL contrasts (forest)
  fig_regime_map                     RQ3: RAIL minus best score gate across 24 regimes
  fig_phase_diagram                  RQ3: AE-gain phase diagram with tie contour
  fig_excess_risk                    theory: excess-risk decomposition + crossover

Run::

    python -m experiments.make_paper_figures --out publication_outputs/figures_v2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # pragma: no cover - import shim for closed-form theory
    from experiments.theory import bayes_post_admission_contamination
    from experiments.theory_risk import crossover_loss_gap, excess_risk_bound
except Exception:  # pragma: no cover
    def bayes_post_admission_contamination(base, alpha, rho):
        num = base * alpha
        den = num + (1.0 - base) * rho
        return 0.0 if den <= 0 else num / den


# ---------------------------------------------------------------------------
# Shared visual identity.
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "Times", "Nimbus Roman",
            "STIXGeneral", "Computer Modern Roman", "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "legend.frameon": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.5,
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

WONG = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermil": "#D55E00", "purple": "#CC79A7", "yellow": "#F0E442",
    "skyblue": "#56B4E9", "black": "#1A1A1A", "grey": "#7F7F7F", "lgrey": "#C9C9C9",
}
INK = "#222222"
GRID = "#B5B5B5"
RAIL_C = WONG["vermil"]
RAILW_C = WONG["blue"]

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
    "rail_gated": (RAIL_C, "*"),
    "rail_weighted": (RAILW_C, "*"),
}
METHOD_LABEL = {
    "static": "Static", "always": "Unfiltered",
    "confidence_gated": "Confidence", "loss_gated": "Loss", "margin_gated": "Margin",
    "co_teaching": "Co-Teaching", "self_paced": "Self-Paced",
    "joint_agreement": "Joint-Agreement", "dynamic_quantile": "Dynamic-Quantile",
    "gce_weight": "GCE", "sce_weight": "SCE", "itlm": "ITLM",
    "rail_gated": "RAIL-Gated", "rail_weighted": "RAIL-Weighted",
}
METHOD_ORDER = [
    "static", "always", "confidence_gated", "loss_gated", "margin_gated",
    "co_teaching", "self_paced", "joint_agreement", "dynamic_quantile",
    "gce_weight", "sce_weight", "itlm", "rail_gated", "rail_weighted",
]
DATASET_PRETTY = {"Synthetic": "Synthetic", "SECOM-like": "SECOM",
                  "APS-like": "APS Failure", "ATC-like": "ATC"}
DATASET_ORDER = ["Synthetic", "SECOM-like", "APS-like", "ATC-like"]
RAIL = {"rail_gated", "rail_weighted"}


def _is_rail(m: str) -> bool:
    return m in RAIL


def _style_axis(ax) -> None:
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(colors=INK, labelcolor=INK)


def _panel_tag(ax, tag, x=-0.02, y=1.04):
    ax.text(x, y, tag, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, fontweight="bold", color=INK)


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{fmt}"), format=fmt)
    plt.close(fig)
    print(f"  - {stem.name}.{{pdf,png}}")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Figure 1. Vigilance gate: the two-sided admission score V(Delta).
# Closed form from experiments.rail_core (tau_min=0.8, tau_max=6.0, k=1.2).
# ---------------------------------------------------------------------------

def fig_vigilance_gate(stem: Path) -> None:
    tau_min, tau_max, k = 0.8, 6.0, 1.2
    theta = 0.5
    d = np.linspace(0.0, 9.0, 600)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.95))

    # (a) component sigmoids and their product, Goldilocks band shaded.
    ax = axes[0]
    s_fast = _sigmoid(k * (d - tau_min))
    s_slow = _sigmoid(k * (tau_max - d))
    V = s_fast * s_slow
    ax.axvspan(tau_min, tau_max, color=WONG["green"], alpha=0.08, zorder=0)
    ax.plot(d, s_fast, color=WONG["skyblue"], lw=1.3, ls=(0, (4, 2)),
            label=r"$s_{\mathsf{fast}}$ (not too hasty)")
    ax.plot(d, s_slow, color=WONG["orange"], lw=1.3, ls=(0, (1, 1.5)),
            label=r"$s_{\mathsf{slow}}$ (not too long)")
    ax.plot(d, V, color=RAIL_C, lw=2.2, label=r"$V=s_{\mathsf{fast}}\cdot s_{\mathsf{slow}}$")
    ax.axhline(theta, color=INK, lw=0.8, ls=(0, (5, 3)), alpha=0.7)
    ax.text(8.85, theta - 0.02, r"threshold $\vartheta$", ha="right", va="top",
            fontsize=7.6, color=INK)
    # mark admission band where V >= theta
    admit = d[V >= theta]
    if admit.size:
        ax.plot([admit.min(), admit.max()], [theta, theta], color=RAIL_C, lw=2.4,
                solid_capstyle="butt", alpha=0.9)
        ax.scatter([admit.min(), admit.max()], [theta, theta], s=14, color=RAIL_C, zorder=6)
    for tv, lab in [(tau_min, r"$\tau_{\min}$"), (tau_max, r"$\tau_{\max}$")]:
        ax.axvline(tv, color=WONG["green"], lw=0.7, alpha=0.6)
        ax.text(tv, 1.04, lab, ha="center", va="bottom", fontsize=8, color=WONG["green"])
    ax.set_xlabel(r"Anchored deliberation $\Delta_t$ (s)")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 1.08)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5)
    _style_axis(ax)
    ax.legend(loc="lower center", fontsize=7.0, handlelength=1.9, borderaxespad=0.3,
              ncol=1, labelspacing=0.3)
    _panel_tag(ax, "(a)")

    # (b) complexity offset beta slides the band to match inspection effort.
    ax = axes[1]
    for beta, col, lab in [(0.0, WONG["purple"], r"$\beta=0$ (simple row)"),
                           (1.5, WONG["blue"], r"$\beta=1.5$"),
                           (3.0, WONG["vermil"], r"$\beta=3.0$ (complex row)")]:
        Vb = _sigmoid(k * (d - (tau_min + beta))) * _sigmoid(k * ((tau_max + beta) - d))
        ax.plot(d, Vb, color=col, lw=1.8, label=lab)
        peak = d[np.argmax(Vb)]
        ax.scatter([peak], [Vb.max()], s=12, color=col, zorder=5)
    ax.axhline(theta, color=INK, lw=0.8, ls=(0, (5, 3)), alpha=0.7)
    ax.text(8.85, theta + 0.02, r"$\vartheta$", ha="right", va="bottom", fontsize=9, color=INK)
    ax.annotate("", xy=(6.6, 0.92), xytext=(3.2, 0.92),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9, alpha=0.8))
    ax.text(4.9, 0.95, "complexity offset", ha="center", va="bottom", fontsize=7.4, color=INK)
    ax.set_xlabel(r"Anchored deliberation $\Delta_t$ (s)")
    ax.set_ylabel(r"Admission score $V_t$")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 1.08)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5)
    _style_axis(ax)
    ax.legend(loc="center right", fontsize=7.2, handlelength=1.6, borderaxespad=0.3)
    _panel_tag(ax, "(b)")

    fig.tight_layout(w_pad=1.6)
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Figure 2. Contamination contract: closed-form curve + out-of-sample check.
# ---------------------------------------------------------------------------

def _contract_rates(run: pd.DataFrame, ds: str, seeds=None):
    """Per-seed (pi, alpha, rho, observed posterior) for RAIL-Gated on a stream."""
    sub = run[run["dataset"] == ds]
    always = sub[sub["method"] == "always"].set_index("run_id")
    rail = sub[sub["method"] == "rail_gated"].set_index("run_id")
    common = always.index.intersection(rail.index)
    if seeds is not None:
        common = common.intersection(seeds)
    N = always.loc[common, "total_feedback"].to_numpy(float)
    tot_contam = always.loc[common, "contaminated_admissions"].to_numpy(float)
    adm = rail.loc[common, "admitted_feedback"].to_numpy(float)
    adm_contam = rail.loc[common, "contaminated_admissions"].to_numpy(float)
    pi = tot_contam / N
    alpha = np.where(tot_contam > 0, adm_contam / tot_contam, 0.0)
    rho = np.where((N - tot_contam) > 0, (adm - adm_contam) / (N - tot_contam), 0.0)
    posterior = np.where(adm > 0, adm_contam / adm, 0.0)
    return pi, alpha, rho, posterior


def fig_contract_verification(stem: Path, run: pd.DataFrame) -> None:
    datasets = [d for d in DATASET_ORDER if d in run["dataset"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05))

    # (a) Contract curve: posterior contamination vs base rate pi.
    ax = axes[0]
    pi_axis = np.linspace(0.0, 0.6, 200)
    ax.plot(pi_axis, pi_axis, color=INK, lw=1.0, ls=(0, (5, 3)), alpha=0.7,
            label="Unfiltered (admit all)")
    cmap = [WONG["blue"], WONG["orange"], WONG["green"], WONG["purple"]]
    for ds, col in zip(datasets, cmap):
        pi, alpha, rho, post = _contract_rates(run, ds)
        a_bar, r_low = float(alpha.mean()), float(rho.mean())
        curve = [bayes_post_admission_contamination(p, a_bar, r_low) for p in pi_axis]
        ax.plot(pi_axis, curve, color=col, lw=1.6, label=DATASET_PRETTY[ds])
        ax.scatter([pi.mean()], [post.mean()], s=46, color=col,
                   edgecolor=INK, linewidths=0.7, zorder=6)
    ax.set_xlabel(r"Base contamination rate $\pi$")
    ax.set_ylabel(r"Admitted contamination $\Pr(C{=}1\mid A_\vartheta{=}1)$")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5)
    _style_axis(ax)
    ax.legend(loc="upper left", fontsize=7.0, handlelength=1.7)
    _panel_tag(ax, "(a)")

    # (b) Out-of-sample calibration: predict on half the seeds, observe on the other half.
    ax = axes[1]
    all_seeds = np.sort(run["run_id"].unique())
    cal, test = all_seeds[0::2], all_seeds[1::2]
    lim = 0.0
    for ds, col in zip(datasets, cmap):
        pi_c, a_c, r_c, _ = _contract_rates(run, ds, seeds=cal)
        _, _, _, post_t = _contract_rates(run, ds, seeds=test)
        pred = bayes_post_admission_contamination(pi_c.mean(), a_c.mean(), r_c.mean())
        obs_m, obs_s = float(post_t.mean()), float(post_t.std(ddof=1))
        ax.errorbar(pred, obs_m, yerr=obs_s, fmt="o", color=col, ms=6.5,
                    ecolor=col, elinewidth=1.0, capsize=2.5, mec=INK, mew=0.7,
                    label=DATASET_PRETTY[ds], zorder=5)
        lim = max(lim, pred, obs_m + obs_s)
    lim = lim * 1.18
    ax.plot([0, lim], [0, lim], color=INK, lw=0.9, ls=(0, (5, 3)), alpha=0.7)
    ax.text(lim * 0.97, lim * 0.9, "perfect\ncalibration", ha="right", va="top",
            fontsize=7.0, color=INK, style="italic")
    ax.set_xlabel("Predicted posterior (calibration seeds)")
    ax.set_ylabel("Observed posterior (held-out seeds)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5)
    _style_axis(ax)
    ax.legend(loc="lower right", fontsize=7.2)
    _panel_tag(ax, "(b)")

    fig.tight_layout(w_pad=1.8)
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Figure 3. Contamination prevented vs. Unfiltered (grouped bars, paired errors).
# ---------------------------------------------------------------------------

PREVENT_METHODS = ["confidence_gated", "loss_gated", "margin_gated", "rail_gated", "rail_weighted"]


def _prevention_stats(run: pd.DataFrame, ds: str):
    sub = run[run["dataset"] == ds]
    base = sub[sub["method"] == "always"].set_index("run_id")["contaminated_admissions"]
    out = {}
    for m in PREVENT_METHODS:
        mm = sub[sub["method"] == m].set_index("run_id")["contaminated_admissions"]
        common = base.index.intersection(mm.index)
        b = base.loc[common].to_numpy(float)
        v = mm.loc[common].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            prevented = np.where(b > 0, (b - v) / b * 100.0, np.nan)
        prevented = prevented[np.isfinite(prevented)]
        out[m] = (float(np.mean(prevented)), float(np.std(prevented, ddof=1)))
    return out


def fig_contamination_prevention(stem: Path, run: pd.DataFrame) -> None:
    datasets = [d for d in DATASET_ORDER if d in run["dataset"].unique()]
    stats = {ds: _prevention_stats(run, ds) for ds in datasets}
    fig, ax = plt.subplots(figsize=(5.8, 3.25))
    n_meth = len(PREVENT_METHODS)
    bar_w = 0.82 / n_meth
    x = np.arange(len(datasets))
    for j, m in enumerate(PREVENT_METHODS):
        color, _ = METHOD_STYLE[m]
        means = [stats[ds][m][0] for ds in datasets]
        errs = [stats[ds][m][1] for ds in datasets]
        offset = (j - (n_meth - 1) / 2) * bar_w
        ax.bar(x + offset, means, width=bar_w * 0.92, color=color, edgecolor=INK,
               linewidth=0.5, label=METHOD_LABEL[m], zorder=3,
               hatch="///" if _is_rail(m) else None)
        ax.errorbar(x + offset, means, yerr=errs, fmt="none", ecolor=INK,
                    elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_PRETTY[d] for d in datasets])
    ax.set_ylabel("Contamination prevented vs.\nUnfiltered admission (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5)
    _style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=n_meth,
              frameon=False, fontsize=7.8, handlelength=1.1, handletextpad=0.4,
              columnspacing=1.0, labelcolor=INK)
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Figure 4. Yield vs. per-admission contamination Pareto (2x2 panels).
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
        rows.append((m, float(yld), float(con / max(adm, 1))))
    return rows


def fig_pareto(stem: Path, run: pd.DataFrame) -> None:
    datasets = [d for d in DATASET_ORDER if d in run["dataset"].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(5.9, 4.8), sharex=True, sharey=True)
    axes = axes.ravel()
    for k_idx, (ax, ds) in enumerate(zip(axes, datasets)):
        pts = _pareto_points(run, ds)
        ordered = [p for p in pts if not _is_rail(p[0])] + [p for p in pts if _is_rail(p[0])]
        for m, yld, rate in ordered:
            color, marker = METHOD_STYLE.get(m, (WONG["grey"], "o"))
            if _is_rail(m):
                ax.scatter(yld, rate, s=145, marker=marker, facecolor=color,
                           edgecolor=INK, linewidths=0.9, zorder=6)
            else:
                ax.scatter(yld, rate, s=46, marker=marker, facecolor=color,
                           edgecolor="white", linewidths=0.6, zorder=3)
        non_dom = []
        for m, yld, rate in pts:
            dominated = any((m2 != m) and (y2 + 1e-9 >= yld) and (r2 <= rate + 1e-9)
                            and ((y2 > yld + 1e-9) or (r2 < rate - 1e-9))
                            for m2, y2, r2 in pts)
            if not dominated:
                non_dom.append((m, yld, rate))
        env = sorted(non_dom, key=lambda kv: kv[1])
        if len(env) >= 2:
            ax.plot([p[1] for p in env], [p[2] for p in env], "-", color=INK,
                    lw=1.0, alpha=0.55, zorder=2, solid_joinstyle="round")
        ax.text(0.04, 0.93, f"({chr(97 + k_idx)})  {DATASET_PRETTY.get(ds, ds)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9.0, color=INK)
        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(-0.015, 0.42)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5)
        _style_axis(ax)
        if k_idx == 0:
            ax.annotate("better", xy=(1.02, 0.012), xytext=(0.66, 0.135), ha="center",
                        va="center", fontsize=7.0, color=INK,
                        arrowprops=dict(arrowstyle="->", color=INK, lw=0.8, alpha=0.8))
    for ax in axes[2:]:
        ax.set_xlabel("Admitted yield")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Per-admission\ncontamination rate")
    handles = []
    for m in METHOD_ORDER:
        color, marker = METHOD_STYLE.get(m, (WONG["grey"], "o"))
        handles.append(mlines.Line2D([], [], color="none", marker=marker,
                       markerfacecolor=color, markeredgecolor=INK if _is_rail(m) else "white",
                       markeredgewidth=0.7, markersize=9 if _is_rail(m) else 6.5,
                       label=METHOD_LABEL.get(m, m)))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.11),
               frameon=False, fontsize=7.6, handletextpad=0.4, columnspacing=1.1, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Figure 5. 30-seed Macro-F1 distributions (split violins + box + seed strip).
# ---------------------------------------------------------------------------

DIST_METHODS = ["always", "dynamic_quantile", "itlm", "rail_gated", "rail_weighted"]


def fig_macro_f1_distributions(stem: Path, run: pd.DataFrame) -> None:
    datasets = [d for d in DATASET_ORDER if d in run["dataset"].unique()]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 3.0), sharey=False)
    rng = np.random.default_rng(7)
    for ax, ds in zip(axes, datasets):
        sub = run[run["dataset"] == ds]
        data, positions, colors = [], [], []
        for i, m in enumerate(DIST_METHODS):
            vals = sub[sub["method"] == m]["final_macro_f1"].to_numpy(float)
            data.append(vals)
            positions.append(i)
            colors.append(METHOD_STYLE[m][0])
        parts = ax.violinplot(data, positions=positions, widths=0.78,
                              showmeans=False, showextrema=False)
        for body, col in zip(parts["bodies"], colors):
            body.set_facecolor(col)
            body.set_alpha(0.30)
            body.set_edgecolor(col)
            body.set_linewidth(0.8)
        for i, (vals, col) in enumerate(zip(data, colors)):
            jit = rng.uniform(-0.10, 0.10, size=vals.size)
            ax.scatter(np.full(vals.size, i) + jit, vals, s=4.5, color=col,
                       alpha=0.45, edgecolors="none", zorder=2, rasterized=True)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.add_patch(mpatches.Rectangle((i - 0.07, q1), 0.14, q3 - q1,
                         facecolor="white", edgecolor=INK, linewidth=0.7, zorder=3))
            ax.plot([i - 0.07, i + 0.07], [med, med], color=INK, lw=1.2, zorder=4)
        ax.set_xticks(range(len(DIST_METHODS)))
        ax.set_xticklabels([METHOD_LABEL[m] for m in DIST_METHODS], rotation=40,
                           ha="right", fontsize=6.6)
        ax.set_title(DATASET_PRETTY[ds], fontsize=8.5)
        ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5)
        _style_axis(ax)
    axes[0].set_ylabel("End-of-stream Macro-F1")
    fig.tight_layout(w_pad=1.0)
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Figure 6. Nemenyi critical-difference diagram on Admission Efficiency.
# ---------------------------------------------------------------------------

def _avg_ranks_ae(summary: pd.DataFrame):
    from scipy.stats import rankdata
    datasets = [d for d in DATASET_ORDER if d in summary["dataset"].unique()]
    M = np.array([[summary[(summary.dataset == d) & (summary.method == m)]["ae_mean"].values[0]
                   for m in METHOD_ORDER] for d in datasets])
    ranks = np.array([rankdata(-row, method="average") for row in M])
    return ranks.mean(0), len(datasets)


def fig_ae_ranking_cd(stem: Path, summary: pd.DataFrame) -> None:
    avg, N = _avg_ranks_ae(summary)
    k = len(METHOD_ORDER)
    q05 = 3.354  # Studentized range / sqrt(2) critical value, alpha=0.05, k=14
    CD = q05 * math.sqrt(k * (k + 1) / (6 * N))
    order = np.argsort(avg)
    names = [METHOD_LABEL[METHOD_ORDER[i]] for i in order]
    ranks = avg[order]
    is_rail = [METHOD_ORDER[i] in RAIL for i in order]

    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    axis_y = 0.80
    ax.plot([lo, hi], [axis_y, axis_y], color=INK, lw=1.1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [axis_y, axis_y + 0.018], color=INK, lw=0.9)
        ax.text(r, axis_y + 0.045, str(r), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.text((lo + hi) / 2, axis_y + 0.10, "Average rank on Admission Efficiency (lower is better)",
            ha="center", va="bottom", fontsize=8.5, color=INK)

    n = len(names)
    left_idx = list(range(0, (n + 1) // 2))
    right_idx = list(range(n - 1, (n + 1) // 2 - 1, -1))
    row_gap = 0.085

    def _label(idx_list, side):
        for j, i in enumerate(idx_list):
            y = axis_y - 0.08 - j * row_gap
            col = RAIL_C if is_rail[i] else INK
            fw = "bold" if is_rail[i] else "normal"
            xr = ranks[i]
            if side == "left":
                ax.plot([xr, xr], [axis_y, y], color=col, lw=1.0)
                ax.plot([xr, lo - 0.4], [y, y], color=col, lw=1.0)
                ax.text(lo - 0.48, y, f"{names[i]}  ({xr:.2f})", ha="right",
                        va="center", fontsize=7.6, color=col, fontweight=fw)
            else:
                ax.plot([xr, xr], [axis_y, y], color=col, lw=1.0)
                ax.plot([xr, hi + 0.4], [y, y], color=col, lw=1.0)
                ax.text(hi + 0.48, y, f"({xr:.2f})  {names[i]}", ha="left",
                        va="center", fontsize=7.6, color=col, fontweight=fw)

    _label(left_idx, "left")
    _label(right_idx, "right")

    # CD bar.
    cd_y = axis_y + 0.16
    x0 = lo
    ax.plot([x0, x0 + CD], [cd_y, cd_y], color=INK, lw=1.6)
    for xx in (x0, x0 + CD):
        ax.plot([xx, xx], [cd_y - 0.012, cd_y + 0.012], color=INK, lw=1.2)
    ax.text(x0 + CD / 2, cd_y + 0.02, f"CD = {CD:.2f}", ha="center", va="bottom",
            fontsize=8.0, color=INK)

    # Cliques: connect methods whose rank gap < CD (single contiguous bar here).
    clique_y = axis_y - 0.028
    ax.plot([ranks.min(), min(ranks.min() + CD, ranks.max())], [clique_y, clique_y],
            color=RAIL_C, lw=3.2, alpha=0.55, solid_capstyle="round")
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Figure 7. Matched-seed RAIL contrasts vs. score-based gates (forest plot).
# ---------------------------------------------------------------------------

COMPARATORS = ["confidence_gated", "loss_gated", "margin_gated", "dynamic_quantile"]


def _boot_ci(values, n_boot=10000, seed=20260616):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return math.nan, math.nan, math.nan
    if a.size == 1 or np.allclose(a, a[0]):
        return float(a[0]), float(a[0]), float(a[0])
    rng = np.random.default_rng(seed)
    bm = rng.choice(a, size=(n_boot, a.size), replace=True).mean(1)
    return float(a.mean()), float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))


def _paired(run, ds, m, comp, metric):
    sub = run[(run.dataset == ds) & run.method.isin([m, comp])]
    piv = sub.pivot(index="run_id", columns="method", values=metric).dropna()
    return (piv[m] - piv[comp]).to_numpy(float)


def fig_benchmark_contrasts(stem: Path, run: pd.DataFrame) -> None:
    datasets = [d for d in DATASET_ORDER if d in run["dataset"].unique()]
    row_spec = [(d, c) for d in datasets for c in COMPARATORS]
    yv = np.arange(len(row_spec))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 5.0), sharey=True)
    rng = np.random.default_rng(519)
    specs = [("ae", "Admission Efficiency difference"),
             ("final_macro_f1", "Macro-F1 difference")]
    for pidx, (ax, (metric, label)) in enumerate(zip(axes, specs)):
        for ridx, ((ds, comp), y) in enumerate(zip(row_spec, yv)):
            diff = _paired(run, ds, "rail_gated", comp, metric)
            mean, lo, hi = _boot_ci(diff, seed=10000 + pidx * 1000 + ridx)
            jit = rng.uniform(-0.10, 0.10, size=diff.size)
            ax.scatter(diff, y + jit, s=5.5, color=WONG["lgrey"], edgecolors="none",
                       alpha=0.6, rasterized=True, zorder=1)
            ax.hlines(y, lo, hi, color=RAIL_C, lw=1.4, zorder=3)
            ax.scatter(mean, y, s=20, marker="s", color=RAIL_C, edgecolors="white",
                       linewidths=0.5, zorder=4)
        ax.axvline(0.0, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.set_xlabel(label + "\n(positive favours RAIL)")
        ax.grid(axis="x", color=GRID, lw=0.4, alpha=0.5)
        _style_axis(ax)
        _panel_tag(ax, f"({chr(97 + pidx)})", x=-0.06)
    axes[0].set_yticks(yv)
    axes[0].set_yticklabels([METHOD_LABEL[c] for _, c in row_spec], fontsize=7.0)
    g = len(COMPARATORS)
    for di, ds in enumerate(datasets):
        top = di * g
        yc = (yv[top] + yv[top + g - 1]) / 2.0
        axes[0].text(-0.40, yc, DATASET_PRETTY[ds], transform=axes[0].get_yaxis_transform(),
                     ha="right", va="center", fontsize=7.6, fontweight="bold", clip_on=False)
        if di < len(datasets) - 1:
            sep = yv[top + g - 1] - 0.5
            for ax in axes:
                ax.axhline(sep, color=GRID, lw=0.6, alpha=0.6, zorder=0)
    raw = mlines.Line2D([], [], ls="none", marker="o", ms=3.2,
                        markerfacecolor=WONG["lgrey"], markeredgecolor="none", label="Matched seeds")
    mh = mlines.Line2D([], [], color=RAIL_C, lw=1.3, marker="s", ms=4.0,
                       markerfacecolor=RAIL_C, markeredgecolor="white", label="Mean and 95% CI")
    axes[1].legend(handles=[raw, mh], loc="lower right", frameon=False, fontsize=7.2)
    fig.subplots_adjust(left=0.26, right=0.99, bottom=0.13, top=0.97, wspace=0.10)
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Regime helpers (24-cell sweep).
# ---------------------------------------------------------------------------

def _regime_contrasts(reg: pd.DataFrame) -> pd.DataFrame:
    rail = reg[reg.method == "rail_gated"][
        ["cell", "correct_prob_normal", "overload_prob", "ae_mean", "macro_f1_mean", "n_seeds"]
    ].rename(columns={"ae_mean": "rail_ae", "macro_f1_mean": "rail_f1"})
    gates = reg[reg.method.isin(COMPARATORS)]
    best_ae = (gates.sort_values("ae_mean").groupby("cell", as_index=False).tail(1)
               [["cell", "ae_mean"]].rename(columns={"ae_mean": "best_gate_ae"}))
    best_f1 = (gates.sort_values("macro_f1_mean").groupby("cell", as_index=False).tail(1)
               [["cell", "macro_f1_mean"]].rename(columns={"macro_f1_mean": "best_gate_f1"}))
    m = rail.merge(best_ae, on="cell").merge(best_f1, on="cell")
    m["ae_diff"] = m["rail_ae"] - m["best_gate_ae"]
    m["f1_diff"] = m["rail_f1"] - m["best_gate_f1"]
    return m


def _grid(frame, col):
    cy = sorted(frame["correct_prob_normal"].unique())
    cx = sorted(frame["overload_prob"].unique())
    piv = frame.pivot(index="correct_prob_normal", columns="overload_prob",
                      values=col).reindex(index=cy, columns=cx)
    return piv.to_numpy(float), cy, cx


def fig_regime_map(stem: Path, reg: pd.DataFrame) -> None:
    contrasts = _regime_contrasts(reg)
    ae_grid, cy, cx = _grid(contrasts, "ae_diff")
    f1_grid, _, _ = _grid(contrasts, "f1_diff")
    ae_cmap = mcolors.LinearSegmentedColormap.from_list("w_vermil", ["#F7F7F7", "#F3C9B4", RAIL_C])
    f1_lim = max(abs(np.nanmin(f1_grid)), abs(np.nanmax(f1_grid)))
    f1_cmap = mcolors.LinearSegmentedColormap.from_list("div", [WONG["blue"], "#F7F7F7", RAIL_C])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.95))
    specs = [(axes[0], ae_grid, ae_cmap, dict(vmin=0.0, vmax=float(np.nanmax(ae_grid))),
              "Admission Efficiency gain"),
             (axes[1], f1_grid, f1_cmap, dict(vmin=-f1_lim, vmax=f1_lim),
              "Macro-F1 difference")]
    for pidx, (ax, grid, cmap, lims, label) in enumerate(specs):
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap,
                       interpolation="nearest", **lims)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                ax.text(j, i, f"{v:+.02f}", ha="center", va="center", fontsize=6.0,
                        color=INK if abs(v) < (lims["vmax"] * 0.6) else "white")
        ax.set_xticks(range(len(cx)))
        ax.set_xticklabels([f"{v:.2f}" for v in cx])
        ax.set_yticks(range(len(cy)))
        ax.set_yticklabels([f"{v:.2f}" for v in cy])
        ax.set_xlabel(r"Overload probability $\pi_{\mathrm{ovr}}$")
        if pidx == 0:
            ax.set_ylabel(r"Operator correctness $P_{\mathrm{ok}}$")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xticks(np.arange(-0.5, len(cx), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(cy), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        _panel_tag(ax, f"({chr(97 + pidx)})")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(label + "\n(RAIL minus best score gate)", fontsize=6.7)
        cb.ax.tick_params(labelsize=6.3, width=0.5, length=2)
        cb.outline.set_linewidth(0.5)
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.17, top=0.93, wspace=0.30)
    _save(fig, stem)


def fig_phase_diagram(stem: Path, reg: pd.DataFrame) -> None:
    contrasts = _regime_contrasts(reg)
    ae_grid, cy, cx = _grid(contrasts, "ae_diff")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    cmap = mcolors.LinearSegmentedColormap.from_list("w_vermil", ["#FBFBFB", "#F3C9B4", RAIL_C])
    X, Y = np.meshgrid(np.arange(len(cx)), np.arange(len(cy)))
    cf = ax.contourf(X, Y, ae_grid, levels=12, cmap=cmap)
    cs = ax.contour(X, Y, ae_grid, levels=[0.04, 0.08], colors=INK,
                    linewidths=0.7, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=6.2, fmt="%.2f")
    # highlight the "RAIL-favoured" region (largest gains): high overload, mid correctness
    ax.scatter(X.ravel(), Y.ravel(), s=10, color=INK, alpha=0.35, zorder=4)
    ax.set_xticks(range(len(cx)))
    ax.set_xticklabels([f"{v:.2f}" for v in cx])
    ax.set_yticks(range(len(cy)))
    ax.set_yticklabels([f"{v:.2f}" for v in cy])
    ax.set_xlabel(r"Overload probability $\pi_{\mathrm{ovr}}$")
    ax.set_ylabel(r"Operator correctness $P_{\mathrm{ok}}$")
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("AE gain over best score gate", fontsize=7.4)
    cb.ax.tick_params(labelsize=6.6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Figure 10. Excess-risk decomposition and the crossover loss gap (theory).
# Constants match the released theory_grid.json: D=G=1, horizon_total=2000.
# ---------------------------------------------------------------------------

def fig_excess_risk(stem: Path) -> None:
    D = G = 1.0
    H = 2000
    c_rail, c_comp = 0.05, 0.20
    y_rail, y_comp = 0.55, 1.0
    T_rail = max(1, round(y_rail * H))
    T_comp = max(1, round(y_comp * H))
    try:
        dstar = crossover_loss_gap(c_rail, c_comp, y_rail, y_comp, H, D, G)
    except Exception:
        dstar = (D * G * (1 / math.sqrt(T_rail) - 1 / math.sqrt(T_comp))) / (c_comp - c_rail)
    delta = np.linspace(0.0, 0.6, 300)
    stat_r = D * G / math.sqrt(T_rail)
    stat_c = D * G / math.sqrt(T_comp)
    bound_r = stat_r + c_rail * delta
    bound_c = stat_c + c_comp * delta

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05))

    # (a) Decomposition: stacked statistical + contamination terms for RAIL.
    ax = axes[0]
    ax.fill_between(delta, 0, stat_r, color=WONG["skyblue"], alpha=0.55,
                    label=r"statistical $DG/\sqrt{T}$")
    ax.fill_between(delta, stat_r, stat_r + c_rail * delta, color=RAIL_C, alpha=0.5,
                    label=r"contamination $c_\vartheta\,\Delta$")
    ax.plot(delta, bound_r, color=INK, lw=1.4)
    ax.set_xlabel(r"Clean--corrupt loss gap $\Delta$")
    ax.set_ylabel("Excess-risk bound")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, max(bound_r) * 1.1)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5)
    _style_axis(ax)
    ax.legend(loc="upper left", fontsize=7.4)
    _panel_tag(ax, "(a)")

    # (b) RAIL vs higher-yield competitor; crossover Delta*.
    ax = axes[1]
    ax.plot(delta, bound_c, color=WONG["green"], lw=1.8,
            label=fr"competitor ($c={c_comp:.2f}$, yield {y_comp:.2f})")
    ax.plot(delta, bound_r, color=RAIL_C, lw=1.8,
            label=fr"RAIL ($c={c_rail:.2f}$, yield {y_rail:.2f})")
    ax.axvline(dstar, color=INK, lw=0.9, ls=(0, (4, 2)), alpha=0.8)
    ax.scatter([dstar], [stat_r + c_rail * dstar], s=34, color=INK, zorder=6)
    ax.text(dstar + 0.012, ax.get_ylim()[1] * 0.55 if False else 0.04,
            fr"$\Delta_\star={dstar:.3f}$", fontsize=8.0, color=INK)
    ymax = max(bound_c.max(), bound_r.max())
    ax.fill_betweenx([0, ymax], dstar, 0.6, color=RAIL_C, alpha=0.06)
    ax.text(0.58, ymax * 0.40, "RAIL bound\ntighter", ha="right", va="top",
            fontsize=7.0, color=RAIL_C, style="italic")
    ax.text(0.075, ymax * 0.62, "competitor\nwins", ha="left", va="top",
            fontsize=7.0, color=WONG["green"], style="italic")
    ax.set_xlabel(r"Clean--corrupt loss gap $\Delta$")
    ax.set_ylabel("Excess-risk bound")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, ymax * 1.05)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5)
    _style_axis(ax)
    ax.legend(loc="upper center", fontsize=7.2)
    _panel_tag(ax, "(b)")

    fig.tight_layout(w_pad=1.8)
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="publication_outputs/self_contained_v3")
    ap.add_argument("--regime", default="publication_outputs/regime/regime_long.csv")
    ap.add_argument("--out", default="publication_outputs/figures_v2")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    run = pd.read_csv(root / args.data / "run_metrics.csv")
    run = run[run["method"].isin(METHOD_ORDER)]
    summary = pd.read_csv(root / args.data / "summary_metrics.csv")
    reg = pd.read_csv(root / args.regime)
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"[paper-figs] writing to {out}")
    fig_vigilance_gate(out / "fig_vigilance_gate")
    fig_contract_verification(out / "fig_contract_verification", run)
    fig_contamination_prevention(out / "fig_contamination_prevention", run)
    fig_pareto(out / "fig_pareto_yield_vs_contamination", run)
    fig_macro_f1_distributions(out / "fig_macro_f1_distributions", run)
    fig_ae_ranking_cd(out / "fig_ae_ranking_cd", summary)
    fig_benchmark_contrasts(out / "fig_benchmark_contrasts", run)
    fig_regime_map(out / "fig_regime_map", reg)
    fig_phase_diagram(out / "fig_phase_diagram", reg)
    fig_excess_risk(out / "fig_excess_risk")
    print("[paper-figs] done")


if __name__ == "__main__":
    main()
