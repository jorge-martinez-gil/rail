"""Publication-grade statistics for the RAIL experiments.

The paper reports mean +/- standard deviation over 30 independent seeds and
applies a paired Wilcoxon signed-rank test on key comparisons. For the
Information Systems submission we strengthen this with:

* :func:`bootstrap_ci` -- percentile and BCa confidence intervals.
* :func:`paired_difference_ci` -- paired bootstrap of the mean difference.
* :func:`cliffs_delta` -- non-parametric effect size for paired runs.
* :func:`rank_biserial` -- effect size companion to Wilcoxon signed-rank.
* :func:`paired_wilcoxon` -- exact / large-sample paired Wilcoxon p-value
  (two-sided), with continuity correction; we keep our own implementation
  so the module has zero hard dependencies on SciPy.
* :func:`holm_bonferroni` -- step-down family-wise error control.
* :func:`compute_statistical_report` -- one-call helper that turns a dict of
  ``policy -> list_of_metric_values`` into a markdown-ready report with CIs,
  effect sizes, and corrected p-values for every policy vs. ``baseline``.

The module is pure standard library so it can be invoked from CI without
adding dependencies. NumPy is used opportunistically when available for
speed but never required.

Note
----
This module is named ``rail_stats`` rather than ``statistics`` to avoid
shadowing the Python standard library ``statistics`` module when scripts
in ``experiments/`` are run directly (which puts the experiments directory
on ``sys.path``). Importing ``experiments.rail_stats`` always works; the
short name preserves the stdlib ``import statistics`` behaviour used by
``river`` and other dependencies.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean, median, stdev

try:  # NumPy is optional.
    import numpy as _np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Bootstrap intervals.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    lower: float
    upper: float
    level: float
    method: str

    def as_tuple(self) -> tuple[float, float, float]:
        return self.point, self.lower, self.upper


def _resample(values: Sequence[float], rng: random.Random) -> list[float]:
    n = len(values)
    return [values[rng.randrange(n)] for _ in range(n)]


def bootstrap_ci(
    values: Sequence[float],
    statistic: str = "mean",
    iterations: int = 5_000,
    level: float = 0.95,
    method: str = "percentile",
    seed: int = 0,
) -> ConfidenceInterval:
    """Bootstrap confidence interval for the chosen statistic.

    Parameters
    ----------
    values : sequence of float
        Observed values (one per run).
    statistic : {"mean", "median"}
    iterations : int
        Number of bootstrap resamples.
    level : float
        Two-sided confidence level, e.g. ``0.95``.
    method : {"percentile", "bca"}
        ``"percentile"`` is the classical Efron interval.
        ``"bca"`` is bias-corrected and accelerated (Efron, 1987).
    seed : int
        RNG seed for reproducibility.
    """

    if not values:
        raise ValueError("values must be non-empty")
    if statistic not in ("mean", "median"):
        raise ValueError("statistic must be 'mean' or 'median'")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    if iterations < 100:
        raise ValueError("iterations must be >= 100")
    if method not in ("percentile", "bca"):
        raise ValueError("method must be 'percentile' or 'bca'")

    rng = random.Random(seed)
    stat_fn = mean if statistic == "mean" else median
    obs = stat_fn(values)

    boot = [stat_fn(_resample(values, rng)) for _ in range(iterations)]
    boot.sort()
    alpha = (1.0 - level) / 2.0

    if method == "percentile":
        lo = _interp_quantile(boot, alpha)
        hi = _interp_quantile(boot, 1.0 - alpha)
        return ConfidenceInterval(obs, lo, hi, level, "percentile")

    # BCa
    z0 = _phi_inv(sum(1 for b in boot if b < obs) / iterations)
    # Acceleration via jackknife.
    n = len(values)
    jack = []
    for i in range(n):
        sample = list(values[:i]) + list(values[i + 1 :])
        jack.append(stat_fn(sample))
    jack_mean = mean(jack)
    num = sum((jack_mean - j) ** 3 for j in jack)
    den = 6.0 * (sum((jack_mean - j) ** 2 for j in jack) ** 1.5)
    a_hat = num / den if den > 0 else 0.0

    def _adj(p: float) -> float:
        z = z0 + (_phi_inv(p))
        denom = 1.0 - a_hat * z
        if denom == 0.0:
            denom = 1e-12
        return _phi(z0 + z / denom)

    lo = _interp_quantile(boot, max(0.0, min(1.0, _adj(alpha))))
    hi = _interp_quantile(boot, max(0.0, min(1.0, _adj(1.0 - alpha))))
    return ConfidenceInterval(obs, lo, hi, level, "bca")


def paired_difference_ci(
    a: Sequence[float],
    b: Sequence[float],
    iterations: int = 5_000,
    level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Paired-bootstrap interval for ``mean(a) - mean(b)``."""

    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    diffs = [ai - bi for ai, bi in zip(a, b, strict=False)]
    return bootstrap_ci(diffs, statistic="mean", iterations=iterations, level=level, seed=seed)


# ---------------------------------------------------------------------------
# Effect sizes.
# ---------------------------------------------------------------------------


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta non-parametric effect size in ``[-1, 1]``.

    ``delta = (#(a > b) - #(a < b)) / (n_a * n_b)``. Suitable companion to
    Wilcoxon. Magnitude bands (Romano et al., 2006): |d| < 0.147 negligible,
    < 0.33 small, < 0.474 medium, otherwise large.
    """

    if not a or not b:
        raise ValueError("inputs must be non-empty")
    greater = 0
    less = 0
    for ai in a:
        for bi in b:
            if ai > bi:
                greater += 1
            elif ai < bi:
                less += 1
    return (greater - less) / (len(a) * len(b))


def cliffs_delta_magnitude(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def rank_biserial(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank-biserial correlation for paired data.

    ``r = (f_pos - f_neg) / n`` over signed-rank pairs, ignoring zero
    differences (Kerby's simple formula). The result lies in ``[-1, 1]``.
    """

    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    diffs = [ai - bi for ai, bi in zip(a, b, strict=False)]
    nz = [d for d in diffs if d != 0]
    if not nz:
        return 0.0
    ranks = _rankdata([abs(d) for d in nz])
    total = sum(ranks)
    pos = sum(r for r, d in zip(ranks, nz, strict=False) if d > 0)
    neg = sum(r for r, d in zip(ranks, nz, strict=False) if d < 0)
    if total == 0:
        return 0.0
    return (pos - neg) / total


# ---------------------------------------------------------------------------
# Paired Wilcoxon signed-rank test.
# ---------------------------------------------------------------------------


def paired_wilcoxon(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test (large-sample normal approximation).

    Returns the test statistic ``W``, normal-approximation ``z``, two-sided
    ``p`` value, sample size of non-zero differences, and the rank-biserial
    effect size. Uses average ranks for ties and continuity correction.

    For small samples (``n < 10``) the normal approximation is conservative
    but documented; we report ``n_nonzero`` so reviewers can verify.
    """

    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be one of two-sided, greater, less")

    diffs = [ai - bi for ai, bi in zip(a, b, strict=False)]
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return {"W": 0.0, "z": 0.0, "p_value": 1.0, "n_nonzero": 0, "r_rb": 0.0}

    abs_nz = [abs(d) for d in nz]
    ranks = _rankdata(abs_nz)
    w_pos = sum(r for r, d in zip(ranks, nz, strict=False) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, nz, strict=False) if d < 0)
    W = float(min(w_pos, w_neg))

    mean_w = n * (n + 1) / 4.0
    # Variance with tie correction.
    tie_correction = _tie_correction(abs_nz)
    var_w = (n * (n + 1) * (2 * n + 1) - tie_correction) / 24.0
    if var_w <= 0.0:
        return {"W": W, "z": 0.0, "p_value": 1.0, "n_nonzero": n, "r_rb": 0.0}

    # Continuity correction: shift by 0.5 toward the mean.
    diff = w_pos - mean_w
    if diff > 0:
        z = (diff - 0.5) / math.sqrt(var_w)
    elif diff < 0:
        z = (diff + 0.5) / math.sqrt(var_w)
    else:
        z = 0.0

    if alternative == "two-sided":
        p = 2.0 * (1.0 - _phi(abs(z)))
    elif alternative == "greater":
        p = 1.0 - _phi(z)
    else:
        p = _phi(z)

    return {
        "W": W,
        "z": float(z),
        "p_value": float(min(1.0, max(0.0, p))),
        "n_nonzero": int(n),
        "r_rb": float(rank_biserial(a, b)),
    }


# ---------------------------------------------------------------------------
# Multiple-comparison correction.
# ---------------------------------------------------------------------------


def holm_bonferroni(p_values: Iterable[float]) -> list[float]:
    """Holm step-down adjusted p-values (Holm, 1979).

    Conservative family-wise error control at the same alpha as Bonferroni
    but uniformly more powerful.
    """

    ps = list(p_values)
    m = len(ps)
    if m == 0:
        return []
    indexed = sorted(range(m), key=lambda i: ps[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(indexed):
        scaled = ps[i] * (m - rank)
        running_max = max(running_max, scaled)
        adjusted[i] = min(1.0, running_max)
    return adjusted


# ---------------------------------------------------------------------------
# Convenience: per-comparison report ready for the paper.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonRow:
    policy: str
    n: int
    mean: float
    sd: float
    ci_lower: float
    ci_upper: float
    diff_mean: float
    diff_ci_lower: float
    diff_ci_upper: float
    cliffs_delta: float
    cliffs_magnitude: str
    rank_biserial: float
    wilcoxon_p: float
    wilcoxon_p_holm: float


def compute_statistical_report(
    results: dict[str, Sequence[float]],
    baseline: str,
    iterations: int = 5_000,
    level: float = 0.95,
    seed: int = 0,
) -> list[ComparisonRow]:
    """One-call helper that produces a comparison row for every non-baseline policy."""

    if baseline not in results:
        raise KeyError(f"baseline {baseline!r} not in results")
    base_vals = list(results[baseline])
    if not base_vals:
        raise ValueError("baseline must have at least one observation")

    rows: list[ComparisonRow] = []
    raw_ps: list[float] = []
    intermediate: list[tuple[str, ConfidenceInterval, ConfidenceInterval, float, float, float]] = []

    for policy, vals in results.items():
        if policy == baseline:
            continue
        vals = list(vals)
        if len(vals) != len(base_vals):
            raise ValueError(
                f"policy {policy!r} has {len(vals)} runs but baseline has {len(base_vals)}"
            )
        own_ci = bootstrap_ci(vals, "mean", iterations=iterations, level=level, seed=seed)
        diff_ci = paired_difference_ci(
            vals, base_vals, iterations=iterations, level=level, seed=seed + 1
        )
        wx = paired_wilcoxon(vals, base_vals)
        cd = cliffs_delta(vals, base_vals)
        raw_ps.append(wx["p_value"])
        intermediate.append((policy, own_ci, diff_ci, cd, wx["r_rb"], wx["p_value"]))

    adjusted = holm_bonferroni(raw_ps)
    for (policy, own_ci, diff_ci, cd, rb, p_raw), p_adj in zip(
        intermediate, adjusted, strict=False
    ):
        vals = list(results[policy])
        rows.append(
            ComparisonRow(
                policy=policy,
                n=len(vals),
                mean=float(mean(vals)),
                sd=float(stdev(vals)) if len(vals) > 1 else 0.0,
                ci_lower=own_ci.lower,
                ci_upper=own_ci.upper,
                diff_mean=diff_ci.point,
                diff_ci_lower=diff_ci.lower,
                diff_ci_upper=diff_ci.upper,
                cliffs_delta=cd,
                cliffs_magnitude=cliffs_delta_magnitude(cd),
                rank_biserial=rb,
                wilcoxon_p=p_raw,
                wilcoxon_p_holm=p_adj,
            )
        )
    return rows


def report_to_markdown(rows: Sequence[ComparisonRow], baseline: str) -> str:
    """Render a comparison report as a markdown table (paper-ready)."""

    header = (
        f"### Statistical comparison vs `{baseline}`\n\n"
        "| Policy | n | Mean | 95% CI | Δ vs baseline | Δ 95% CI | Cliff's δ (|δ|) | p (raw) | p (Holm) |\n"
        "|---|---:|---:|---|---:|---|---|---:|---:|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| `{r.policy}` | {r.n} | {r.mean:.4f} ± {r.sd:.4f} | [{r.ci_lower:.4f}, {r.ci_upper:.4f}] | {r.diff_mean:+.4f} | [{r.diff_ci_lower:+.4f}, {r.diff_ci_upper:+.4f}] | {r.cliffs_delta:+.3f} ({r.cliffs_magnitude}) | {r.wilcoxon_p:.4f} | {r.wilcoxon_p_holm:.4f} |"
        )
    return header + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Small numeric helpers (kept private to avoid SciPy dependence).
# ---------------------------------------------------------------------------


def report_to_latex(
    rows: "Sequence[ComparisonRow]",
    baseline: str,
    caption: str = "",
    label: str = "tab:comparison",
    alpha: float = 0.05,
) -> str:
    """Render a comparison report as a publication-ready LaTeX booktabs table.

    The table uses ``booktabs`` column rules and marks statistically significant
    results (Holm-corrected p < ``alpha``) with a dagger (†).  Bold the row
    with the largest mean improvement.

    Parameters
    ----------
    rows : sequence of ComparisonRow
        Output of :func:`compute_statistical_report`.
    baseline : str
        Name of the baseline policy (for the caption).
    caption : str
        Optional table caption.  A sensible default is provided if empty.
    label : str
        LaTeX ``\\label`` key.
    alpha : float
        Significance threshold for the dagger marker.

    Returns
    -------
    str
        Complete LaTeX ``table`` environment, ready to paste into a ``.tex`` file.
    """

    if not caption:
        caption = (
            f"Statistical comparison of all policies vs.~\\texttt{{{baseline}}}. "
            "95\\% bootstrap CIs (percentile). "
            "Effect sizes: Cliff's~$\\delta$ with magnitude and rank-biserial~$r$. "
            "$p$ values Holm-corrected; \\dag\\ $p_{\\text{Holm}} < "
            + f"{alpha}"
            + "$."
        )

    def _fmt_p(p: float) -> str:
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    def _sig(p_holm: float) -> str:
        return r"\dag" if p_holm < alpha else ""

    # Find row with largest mean improvement for bolding.
    best_idx = max(range(len(rows)), key=lambda i: rows[i].diff_mean) if rows else -1

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{tabular}{lrccrccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Policy & $n$ & Mean $\pm$ SD & 95\% CI & "
        r"$\Delta$ vs baseline & $\Delta$ 95\% CI & "
        r"Cliff's $\delta$ & $p_{\text{Holm}}$ \\"
    )
    lines.append(r"\midrule")
    for idx, r in enumerate(rows):
        bold_open = r"\textbf{" if idx == best_idx else ""
        bold_close = "}" if idx == best_idx else ""
        sig = _sig(r.wilcoxon_p_holm)
        sig_str = f"$^{{{sig}}}$" if sig else ""
        row_str = (
            f"{bold_open}\\texttt{{{r.policy}}}{bold_close} & "
            f"{r.n} & "
            f"{bold_open}{r.mean:.4f} $\\pm$ {r.sd:.4f}{bold_close} & "
            f"[{r.ci_lower:.4f}, {r.ci_upper:.4f}] & "
            f"{r.diff_mean:+.4f} & "
            f"[{r.diff_ci_lower:+.4f}, {r.diff_ci_upper:+.4f}] & "
            f"{r.cliffs_delta:+.3f} ({r.cliffs_magnitude[0].upper()}) & "
            f"{_fmt_p(r.wilcoxon_p_holm)}{sig_str} \\\\"
        )
        lines.append(row_str)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Small numeric helpers (kept private to avoid SciPy dependence).
# ---------------------------------------------------------------------------


def _interp_quantile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("empty sequence")
    p = max(0.0, min(1.0, p))
    idx = p * (len(sorted_values) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _phi(z: float) -> float:
    """Standard-normal CDF via the error function."""

    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Beasley-Springer-Moro inverse CDF, accurate to ~1e-7."""

    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    # Coefficients from Beasley & Springer (1977) / Moro (1995).
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def _rankdata(values: Sequence[float]) -> list[float]:
    """Return average ranks for a sequence (handles ties)."""

    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _tie_correction(values: Sequence[float]) -> float:
    """Tie correction term for the Wilcoxon variance (sum of t^3 - t)."""

    sorted_v = sorted(values)
    total = 0.0
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        t = j - i + 1
        if t > 1:
            total += t * t * t - t
        i = j + 1
    return total


__all__ = [
    "ComparisonRow",
    "ConfidenceInterval",
    "bootstrap_ci",
    "cliffs_delta",
    "cliffs_delta_magnitude",
    "compute_statistical_report",
    "holm_bonferroni",
    "paired_difference_ci",
    "paired_wilcoxon",
    "rank_biserial",
    "report_to_latex",
    "report_to_markdown",
]
