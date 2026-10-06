"""Regenerate all tables, statistics, and figures for the RAIL-H paper.

Inputs: publication_outputs/hybrid_paper/{temporal,adversarial,imbalance} run CSVs
plus temporal/ccc_curves.json. Outputs: publication_outputs/hybrid_paper/artifacts/.
Run from the repository root:  python -m experiments.make_hybrid_paper_artifacts
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps


REPO = Path(".")
HP = REPO / "publication_outputs/hybrid_paper"
ART = HP / "artifacts"
ART.mkdir(exist_ok=True)

from experiments.rail_stats_extra import multi_dataset_report, critical_difference_diagram
from experiments.reproduce_paper import _write_multi_dataset_report
from experiments.theory import bayes_post_admission_contamination, required_horizon_for_budget, variance_aware_horizon

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.bbox": "tight"})
COLORS = {"always": "#888888", "rail_gated": "#1f77b4", "rail_weighted": "#17becf",
          "itlm": "#d62728", "rail_h": "#2ca02c", "rail_h_cal": "#9467bd",
          "loss_gated": "#ff7f0e", "confidence_gated": "#bcbd22", "dynamic_quantile": "#e377c2"}
LABELS = {"always": "Always", "rail_gated": "RAIL-Gated", "rail_weighted": "RAIL-Weighted",
          "itlm": "ITLM", "rail_h": "RAIL-H", "rail_h_cal": "RAIL-H (cal.)",
          "loss_gated": "Loss-Gated", "confidence_gated": "Confidence-Gated",
          "dynamic_quantile": "Dyn.\\ Quantile", "static": "Static", "co_teaching": "Co-Teaching",
          "self_paced": "Self-Paced", "joint_agreement": "Joint Agreement",
          "gce_weight": "GCE", "sce_weight": "SCE", "margin_gated": "Margin-Gated"}
DS_ORDER = ["APS-like", "ATC-like", "SECOM-like", "Synthetic"]

df = pd.read_csv(HP / "temporal/run_metrics_merged.csv")
df["purity"] = df.contaminated_admissions / df.admitted_feedback.clip(lower=1)

# ---------------- Table 1: purity ranks (all 16 methods) --------------------
def fmt(m, s, pct=True, prec=1):
    if pct: return f"{100*m:.{prec}f} $\\pm$ {100*s:.{prec}f}"
    return f"{m:.3f} $\\pm$ {s:.3f}"

g = df.groupby(["dataset", "method"])
purity = g["purity"].agg(["mean", "std"]).reset_index()
lines = []
methods_sorted = purity[purity.dataset == "APS-like"].sort_values("mean").method.tolist()
for m in methods_sorted:
    cells = []
    for d in DS_ORDER:
        r = purity[(purity.dataset == d) & (purity.method == m)].iloc[0]
        rank = (purity[purity.dataset == d].sort_values("mean").reset_index().method == m).idxmax() + 1
        bold = rank <= 2
        cell = fmt(r["mean"], r["std"])
        cells.append(("\\textbf{%s}" % cell) if bold else cell)
    lines.append(f"{LABELS.get(m, m)} & " + " & ".join(cells) + r" \\")
(ART / "table_purity_full.tex").write_text("\n".join(lines) + "\n")

# ---------------- Table 2: temporal metrics (key methods) -------------------
KEY = ["always", "rail_gated", "itlm", "rail_h", "rail_h_cal"]
rows = []
for d in DS_ORDER:
    for m in KEY:
        sub = df[(df.dataset == d) & (df.method == m)]
        rows.append({
            "dataset": d, "method": LABELS[m],
            "ncg": fmt(sub.ncg.mean(), sub.ncg.std(), pct=False),
            "aucc": fmt(sub.aucc.mean(), sub.aucc.std(), pct=False),
            "half_life": f"{sub.ttfc.median():.0f}",
            "yield_loss": fmt(sub.yield_loss.mean(), sub.yield_loss.std(), pct=False),
            "yield": fmt(sub.admitted_yield.mean(), sub.admitted_yield.std(), pct=False),
            "f1": fmt(sub.final_macro_f1.mean(), sub.final_macro_f1.std(), pct=False),
        })
tex = []
for d in DS_ORDER:
    tex.append(r"\midrule")
    tex.append(r"\multicolumn{7}{l}{\emph{%s}} \\" % d)
    for r in [x for x in rows if x["dataset"] == d]:
        tex.append(f'{r["method"]} & {r["ncg"]} & {r["aucc"]} & {r["half_life"]} & {r["yield_loss"]} & {r["yield"]} & {r["f1"]} \\\\')
(ART / "table_temporal.tex").write_text("\n".join(tex) + "\n")

# ---------------- Table 3: gate diagnostics ---------------------------------
hx = df[df.method == "rail_h"].copy()
hx["alpha_prod"] = hx.alpha_v * hx.alpha_l
hx["rho_prod"] = hx.rho_v * hx.rho_l
tex = []
for d in DS_ORDER:
    s = hx[hx.dataset == d]
    tex.append(
        f"{d} & {s.alpha_v.mean():.3f} & {s.alpha_l.mean():.3f} & {s.alpha_prod.mean():.3f} & "
        f"{s.alpha_conj.mean():.3f} & {s.rho_v.mean():.3f} & {s.rho_l.mean():.3f} & "
        f"{s.rho_prod.mean():.3f} & {s.rho_conj.mean():.3f} & {s.phi_contaminated.mean():.3f} \\\\")
(ART / "table_gates.tex").write_text("\n".join(tex) + "\n")

# ---------------- Table 4: adversarial --------------------------------------
adv = pd.read_csv(HP / "adversarial/run_metrics_partial.csv")
adv["purity"] = adv.contaminated_admissions / adv.admitted_feedback.clip(lower=1)
tex = []
CONDS = [("adv_standard", "Standard"), ("adv_blind", "Telemetry-blind"), ("adv_inverted", "Inverted")]
for c, cl in CONDS:
    tex.append(r"\midrule")
    tex.append(r"\multicolumn{5}{l}{\emph{%s}} \\" % cl)
    for m in KEY:
        s = adv[(adv.dataset == c) & (adv.method == m)]
        tex.append(f"{LABELS[m]} & {fmt(s.purity.mean(), s.purity.std())} & "
                   f"{fmt(s.ncg.mean(), s.ncg.std(), pct=False)} & "
                   f"{fmt(s.admitted_yield.mean(), s.admitted_yield.std(), pct=False)} & "
                   f"{fmt(s.final_macro_f1.mean(), s.final_macro_f1.std(), pct=False)} \\\\")
(ART / "table_adversarial.tex").write_text("\n".join(tex) + "\n")

# ---------------- Stats: Friedman/Nemenyi/Holm on purity --------------------
per = {}
for d in DS_ORDER:
    per[d] = {m: df[(df.dataset == d) & (df.method == m)].sort_values("run_id").purity.tolist()
              for m in df.method.unique() if m != "static"}
report = multi_dataset_report(per_dataset_scores=per, baseline="rail_h", higher_is_better=False)
_write_multi_dataset_report(report, ART / "stats_purity_vs_rail_h.md")
try:
    critical_difference_diagram(report.nemenyi, output_path=str(ART / "cd_diagram_purity.pdf"))
except Exception as e:
    print("CD diagram failed:", e)

# Paired Wilcoxon: rail_h vs {rail_gated, itlm}, rail_h_cal vs rail_gated, per dataset + metric
wtests = []
for d in DS_ORDER:
    sub = df[df.dataset == d]
    piv = {m: sub[sub.method == m].sort_values("run_id") for m in ["rail_h", "rail_h_cal", "rail_gated", "itlm"]}
    for a, b in [("rail_h", "rail_gated"), ("rail_h", "itlm"), ("rail_h_cal", "rail_gated")]:
        for metric in ["purity", "ncg", "final_macro_f1", "admitted_yield"]:
            x = piv[a][metric].values; y = piv[b][metric].values
            if np.allclose(x, y): w, p = np.nan, 1.0
            else: w, p = sps.wilcoxon(x, y)
            wtests.append({"dataset": d, "a": a, "b": b, "metric": metric,
                           "mean_a": float(np.mean(x)), "mean_b": float(np.mean(y)),
                           "delta": float(np.mean(x) - np.mean(y)), "p_raw": float(p)})
wt = pd.DataFrame(wtests)
# Holm within each (a,b,metric) family across datasets
out_rows = []
for (a, b, metric), fam in wt.groupby(["a", "b", "metric"]):
    fam = fam.sort_values("p_raw").reset_index(drop=True)
    k = len(fam)
    holm = [min(1.0, fam.p_raw[i] * (k - i)) for i in range(k)]
    holm = np.maximum.accumulate(holm)
    fam["p_holm"] = holm
    out_rows.append(fam)
wt = pd.concat(out_rows)
wt.to_csv(ART / "wilcoxon_paired.csv", index=False)

# ---------------- Fig: CCC curves -------------------------------------------
curves = json.loads((HP / "temporal/ccc_curves.json").read_text())
fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True)
GRID = np.linspace(0.02, 1.0, 80)
for ax, d in zip(axes, DS_ORDER):
    for m in ["always", "rail_gated", "itlm", "rail_h"]:
        ys = []
        for key, entry in curves.items():
            if not key.startswith(d + ":"): continue
            flags = np.array(entry[m], dtype=float)
            if len(flags) == 0: continue
            frac = np.cumsum(flags) / np.arange(1, len(flags) + 1)
            x = np.arange(1, len(flags) + 1) / len(flags)
            ys.append(np.interp(GRID, x, frac))
        ys = np.array(ys)
        ax.plot(GRID, ys.mean(0), color=COLORS[m], label=LABELS[m].replace("\\", ""), lw=1.6)
        ax.fill_between(GRID, ys.mean(0) - ys.std(0), ys.mean(0) + ys.std(0), color=COLORS[m], alpha=0.15)
    ax.set_title(d, fontsize=9)
    ax.set_xlabel("Fraction of admissions")
axes[0].set_ylabel("Running contamination\nfraction")
axes[0].legend(frameon=False, fontsize=7.5)
fig.savefig(ART / "fig_ccc_curves.pdf"); fig.savefig(ART / "fig_ccc_curves.png", dpi=600)
plt.close(fig)

# ---------------- Fig: independence check -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
mk = dict(zip(DS_ORDER, ["o", "s", "^", "D"]))
for d in DS_ORDER:
    s = hx[hx.dataset == d]
    axes[0].scatter(s.alpha_v * s.alpha_l, s.alpha_conj, s=14, alpha=0.7, marker=mk[d], label=d)
    axes[1].scatter(s.rho_v * s.rho_l, s.rho_conj, s=14, alpha=0.7, marker=mk[d], label=d)
for ax, t in zip(axes, [r"Contaminated events: $\alpha_H$ vs $\alpha_V\alpha_L$",
                        r"Clean events: $\rho_H$ vs $\rho_V\rho_L$"]):
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", lw=0.8)
    ax.set_title(t, fontsize=9)
    ax.set_xlabel("Independence prediction (product)")
axes[0].set_ylabel("Measured conjunction rate")
axes[0].legend(frameon=False, fontsize=7.5)
fig.savefig(ART / "fig_independence.pdf"); fig.savefig(ART / "fig_independence.png", dpi=600)
plt.close(fig)

# ---------------- Fig: adversarial ------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.0))
width = 0.15
xs = np.arange(3)
for i, m in enumerate(KEY):
    means = [adv[(adv.dataset == c) & (adv.method == m)].purity.mean() for c, _ in CONDS]
    stds = [adv[(adv.dataset == c) & (adv.method == m)].purity.std() for c, _ in CONDS]
    ax.bar(xs + (i - 2) * width, means, width, yerr=stds, capsize=2,
           color=COLORS[m], label=LABELS[m].replace("\\", ""))
base = [adv[(adv.dataset == c) & (adv.method == "always")].purity.mean() for c, _ in CONDS]
for x, b in zip(xs, base):
    ax.hlines(b, x - 2.5 * width, x + 2.5 * width, color="k", ls=":", lw=1)
ax.set_xticks(xs); ax.set_xticklabels([cl for _, cl in CONDS])
ax.set_ylabel("Admitted-stream\ncontamination rate")
ax.legend(frameon=False, fontsize=7.5, ncol=3)
fig.savefig(ART / "fig_adversarial.pdf"); fig.savefig(ART / "fig_adversarial.png", dpi=600)
plt.close(fig)

# ---------------- Fig: imbalance sweep --------------------------------------
imb = pd.read_csv(HP / "imbalance/run_metrics_partial.csv")
imb["purity"] = imb.contaminated_admissions / imb.admitted_feedback.clip(lower=1)
imb["b"] = imb.dataset.str.replace("imb", "").astype(float)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
for m in KEY:
    s = imb[imb.method == m].groupby("b")
    for ax, col in zip(axes, ["purity", "ae"]):
        mu = s[col].mean(); sd = s[col].std() / np.sqrt(30)
        ax.plot(mu.index, mu.values, color=COLORS[m], marker="o", ms=3, lw=1.3,
                label=LABELS[m].replace("\\", ""))
        ax.fill_between(mu.index, mu - 1.96 * sd, mu + 1.96 * sd, color=COLORS[m], alpha=0.15)
axes[0].set_ylabel("Admitted-stream contamination")
axes[1].set_ylabel("Admission Efficiency")

for ax in axes: ax.set_xlabel("Class-imbalance bias $b$")
axes[0].legend(frameon=False, fontsize=7)
fig.savefig(ART / "fig_imbalance.pdf"); fig.savefig(ART / "fig_imbalance.png", dpi=600)
plt.close(fig)

# ---------------- Fig: contract tightening ----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
pis = np.linspace(0.005, 0.5, 200)
sel = {"rail_gated": ("alpha_v", "rho_v"), "itlm": ("alpha_l", "rho_l")}
s = hx[hx.dataset == "Synthetic"]
meas = {"rail_gated": (s.alpha_v.mean(), s.rho_v.mean()),
        "itlm": (s.alpha_l.mean(), s.rho_l.mean()),
        "rail_h": (s.alpha_conj.mean(), s.rho_conj.mean())}
for m, (a, r) in meas.items():
    ys = [bayes_post_admission_contamination(pi, a, r) for pi in pis]
    axes[0].plot(pis, ys, color=COLORS[m], label=LABELS[m].replace("\\", ""), lw=1.6)
axes[0].plot(pis, pis, "k:", lw=0.8, label="No gate")
axes[0].axvline(s.base_contamination.mean(), color="gray", lw=0.6, ls="--")
axes[0].set_xlabel(r"Base contamination rate $\pi$")
axes[0].set_ylabel(r"Certified $\mathbb{P}(C{=}1 \mid A{=}1)$")
axes[0].legend(frameon=False, fontsize=7.5)
# certified horizon: max N with P(K_N >= B) <= 0.05 (Bennett vs Hoeffding)
from experiments.theory import bennett_contamination_bound
import math
budgets = np.arange(5, 81, 3)
pi0 = float(s.base_contamination.mean())
def max_horizon(B, alpha, bound_fn):
    lo, hi = 1, 2_000_000
    def ok(n):
        eps = B / n - pi0 * alpha
        if eps <= 0: return False
        return bound_fn(n, pi0, alpha, eps) <= 0.05
    if not ok(1): return 0
    while ok(hi): hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid): lo = mid
        else: hi = mid - 1
    return lo
def hoeffding(n, pi, a, eps): return math.exp(-2 * n * eps * eps)
for m in ["rail_gated", "rail_h"]:
    a = meas[m][0]
    benn = [max_horizon(B, a, bennett_contamination_bound) for B in budgets]
    hoef = [max_horizon(B, a, hoeffding) for B in budgets]
    axes[1].plot(budgets, hoef, color=COLORS[m], ls="--", lw=1.2)
    axes[1].plot(budgets, benn, color=COLORS[m], ls="-", lw=1.6,
                 label=LABELS[m].replace("\\", ""))
axes[1].set_xlabel("Contamination budget $B$ (events)")
axes[1].set_ylabel("Max certified horizon $N$")
axes[1].legend(frameon=False, fontsize=7.5)
fig.savefig(ART / "fig_contract.pdf"); fig.savefig(ART / "fig_contract.png", dpi=600)
plt.close(fig)

print("artifacts written to", ART)
