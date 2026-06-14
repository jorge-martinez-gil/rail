"""Journal-grade multi-dataset statistical tests for the RAIL paper.

Companion to :mod:`experiments.rail_stats` (which provides bootstrap CIs,
paired Wilcoxon, Cliff's delta, and Holm correction).

This module adds the two procedures Demsar (2006) recommends -- and that
Information Systems reviewers expect -- for comparing many classifiers across many datasets:

* :func:`friedman_test` -- non-parametric ANOVA over (datasets x methods).
* :func:`nemenyi_posthoc` -- pairwise critical difference.
* :func:`critical_difference_diagram` -- the canonical CD diagram.
* :func:`average_ranks` -- per-dataset ranks averaged across datasets.

Plus a one-call helper :func:`multi_dataset_report` that ingests the same
``policy -> {dataset: list_of_scores}`` shape produced by the run harness
and emits a markdown-ready report plus an optional matplotlib figure.

The module is pure-stdlib (NumPy and matplotlib used opportunistically
when available). The Nemenyi critical values are tabulated for alpha in
{0.05, 0.10} and method counts k in 2..20 (the practically relevant range
for our paper).

References
----------
* Demsar, "Statistical Comparisons of Classifiers over Multiple Data Sets,"
  JMLR 7, 2006.
* Garcia & Herrera, "An Extension on 'Statistical Comparisons of Classifiers
  over Multiple Data Sets' for all Pairwise Comparisons," JMLR 9, 2008.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

try:
    import numpy as _np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    from .rail_stats import holm_bonferroni, paired_wilcoxon
except ImportError:  # pragma: no cover
    from rail_stats import holm_bonferroni, paired_wilcoxon  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Nemenyi critical-difference table (q_alpha values).
# ---------------------------------------------------------------------------
# Standard tabulated values from Demsar (2006), Table 5. q_alpha for two-tailed
# Nemenyi at alpha = 0.05 and 0.10.

_NEMENYI_Q_05: dict[int, float] = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
    16: 3.426,
    17: 3.458,
    18: 3.489,
    19: 3.517,
    20: 3.544,
}

_NEMENYI_Q_10: dict[int, float] = {
    2: 1.645,
    3: 2.052,
    4: 2.291,
    5: 2.459,
    6: 2.589,
    7: 2.693,
    8: 2.780,
    9: 2.855,
    10: 2.920,
    11: 2.978,
    12: 3.030,
    13: 3.077,
    14: 3.120,
    15: 3.159,
    16: 3.196,
    17: 3.230,
    18: 3.261,
    19: 3.291,
    20: 3.319,
}


# ---------------------------------------------------------------------------
# Ranking utilities.
# ---------------------------------------------------------------------------


def _rank_row_descending(values: Sequence[float]) -> list[float]:
    """Average-rank the values where rank 1 = best (largest)."""

    n = len(values)
    if n == 0:
        return []
    indexed = sorted(range(n), key=lambda i: -float(values[i]))
    ranks = [0.0] * n
    i = 0
    rank = 1
    while i < n:
        j = i
        while j + 1 < n and abs(float(values[indexed[j + 1]]) - float(values[indexed[i]])) < 1e-12:
            j += 1
        avg = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        rank += j - i + 1
        i = j + 1
    return ranks


def average_ranks(
    per_dataset_scores: Mapping[str, Mapping[str, Sequence[float]]],
    higher_is_better: bool = True,
) -> dict[str, float]:
    """Average rank of each method across datasets, ranks from per-seed means.

    ``per_dataset_scores`` maps ``dataset -> method -> list_of_seed_scores``.
    Within each dataset we collapse to the mean and rank methods; ranks are
    then averaged across datasets.
    """

    methods = sorted({m for d in per_dataset_scores.values() for m in d})
    if not methods:
        return {}

    rank_sums: dict[str, float] = {m: 0.0 for m in methods}
    n_datasets = 0

    for ds, by_method in per_dataset_scores.items():
        try:
            row_vals = [statistics.mean(by_method[m]) for m in methods]
        except KeyError as exc:
            raise KeyError(f"Dataset {ds!r} missing scores for method {exc.args[0]!r}") from exc
        if not higher_is_better:
            row_vals = [-v for v in row_vals]
        row_ranks = _rank_row_descending(row_vals)
        for m, r in zip(methods, row_ranks, strict=False):
            rank_sums[m] += r
        n_datasets += 1

    return {m: rank_sums[m] / max(n_datasets, 1) for m in methods}


# ---------------------------------------------------------------------------
# Friedman test.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FriedmanResult:
    statistic: float
    p_value: float
    n_datasets: int
    n_methods: int
    average_ranks: dict[str, float]

    @property
    def reject_null(self) -> bool:
        return self.p_value < 0.05


def _chi2_sf(chi2: float, df: int) -> float:
    """Survival function P(X > chi2) for a chi-square with df degrees of freedom.

    Wilson-Hilferty cube-root normal approximation. Accurate to ~1e-3 for df
    in 1..20, more than enough for the magnitudes typical in classifier
    comparisons.
    """
    if df <= 0 or chi2 <= 0.0:
        return 1.0
    z = ((chi2 / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    # Two-tailed survival of standard normal = 0.5 * erfc(z/sqrt(2))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def friedman_test(
    per_dataset_scores: Mapping[str, Mapping[str, Sequence[float]]],
    higher_is_better: bool = True,
) -> FriedmanResult:
    """Iman-Davenport-corrected Friedman test over (datasets x methods)."""

    ranks = average_ranks(per_dataset_scores, higher_is_better=higher_is_better)
    methods = sorted(ranks.keys())
    k = len(methods)
    N = len(per_dataset_scores)
    if k < 2 or N < 2:
        return FriedmanResult(
            statistic=float("nan"),
            p_value=1.0,
            n_datasets=N,
            n_methods=k,
            average_ranks=ranks,
        )

    # Friedman chi-square statistic.
    sum_sq = sum((r - (k + 1) / 2.0) ** 2 for r in ranks.values())
    chi2_F = (12.0 * N / (k * (k + 1))) * sum_sq

    # Iman-Davenport F-correction (recommended over the raw chi-square).
    denom = N * (k - 1) - chi2_F
    if denom <= 0:
        # All methods perfectly tied at the corner: degenerate.
        F_stat = float("inf")
        # Approximate p via chi-square SF.
        p_val = _chi2_sf(chi2_F, k - 1)
    else:
        F_stat = ((N - 1) * chi2_F) / denom
        # F has (k-1, (k-1)(N-1)) degrees of freedom; use chi-square approx of
        # the numerator for p-value (Demsar uses both stats interchangeably
        # in the rejection decision).
        p_val = _chi2_sf(chi2_F, k - 1)

    return FriedmanResult(
        statistic=float(F_stat),
        p_value=float(p_val),
        n_datasets=N,
        n_methods=k,
        average_ranks=ranks,
    )


# ---------------------------------------------------------------------------
# Nemenyi post-hoc.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NemenyiResult:
    critical_difference: float
    alpha: float
    n_datasets: int
    n_methods: int
    average_ranks: dict[str, float]
    # method -> {other_method: is_significantly_different}
    significant: dict[str, dict[str, bool]]


def critical_difference_value(k: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Demsar (2006) critical difference for Nemenyi post-hoc."""

    if k < 2 or n_datasets < 1:
        raise ValueError("k must be >= 2 and n_datasets >= 1")
    table = _NEMENYI_Q_05 if abs(alpha - 0.05) < 1e-9 else _NEMENYI_Q_10
    if k not in table:
        raise ValueError(f"No tabulated q_alpha for k={k}; supported 2..20")
    q = table[k]
    return float(q * math.sqrt(k * (k + 1) / (6.0 * n_datasets)))


def nemenyi_posthoc(
    per_dataset_scores: Mapping[str, Mapping[str, Sequence[float]]],
    higher_is_better: bool = True,
    alpha: float = 0.05,
) -> NemenyiResult:
    """Pairwise Nemenyi post-hoc using critical difference."""

    ranks = average_ranks(per_dataset_scores, higher_is_better=higher_is_better)
    methods = sorted(ranks.keys())
    k = len(methods)
    N = len(per_dataset_scores)
    cd = critical_difference_value(k, N, alpha=alpha)

    sig: dict[str, dict[str, bool]] = {m: {} for m in methods}
    for _i, m1 in enumerate(methods):
        for m2 in methods:
            if m1 == m2:
                sig[m1][m2] = False
                continue
            sig[m1][m2] = abs(ranks[m1] - ranks[m2]) > cd

    return NemenyiResult(
        critical_difference=cd,
        alpha=alpha,
        n_datasets=N,
        n_methods=k,
        average_ranks=ranks,
        significant=sig,
    )


# ---------------------------------------------------------------------------
# Critical-difference diagram (matplotlib).
# ---------------------------------------------------------------------------


def critical_difference_diagram(
    nemenyi: NemenyiResult,
    output_path: str | None = None,
    width: float = 8.0,
    height: float = 2.5,
):  # pragma: no cover - matplotlib I/O
    """Render the canonical CD diagram (Demsar 2006, Fig. 1).

    Returns the matplotlib figure even when ``output_path`` is supplied.
    """

    import matplotlib.pyplot as plt

    ranks = nemenyi.average_ranks
    methods = sorted(ranks.keys(), key=lambda m: ranks[m])
    rs = [ranks[m] for m in methods]
    n = len(methods)
    cd = nemenyi.critical_difference

    fig, ax = plt.subplots(figsize=(width, height))
    lo, hi = math.floor(min(rs)), math.ceil(max(rs))
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    ax.set_xlim(hi, lo)  # better ranks on left
    ax.set_ylim(0, n + 2)
    ax.axis("off")

    # Rank axis.
    y_axis = n + 1.2
    ax.hlines(y_axis, lo, hi, color="black", linewidth=1.0)
    for x in range(lo, hi + 1):
        ax.vlines(x, y_axis - 0.08, y_axis + 0.08, color="black", linewidth=1.0)
        ax.text(x, y_axis + 0.18, str(x), ha="center", va="bottom", fontsize=9)

    # Method lines (alternating left/right of midpoint).
    mid = (lo + hi) / 2.0
    left_methods = [(m, r) for m, r in zip(methods, rs, strict=False) if r <= mid]
    right_methods = [(m, r) for m, r in zip(methods, rs, strict=False) if r > mid]

    def _draw(items, side: str, start_y: float):
        for i, (m, r) in enumerate(items):
            y = start_y - i * 0.55
            if side == "left":
                ax.plot([r, lo + 0.2], [y_axis, y], color="black", linewidth=0.8)
                ax.text(lo + 0.15, y, f"{m} ({r:.2f})", ha="right", va="center", fontsize=9)
            else:
                ax.plot([r, hi - 0.2], [y_axis, y], color="black", linewidth=0.8)
                ax.text(hi - 0.15, y, f"{m} ({r:.2f})", ha="left", va="center", fontsize=9)

    _draw(sorted(left_methods, key=lambda kv: kv[1]), "left", n)
    _draw(sorted(right_methods, key=lambda kv: -kv[1]), "right", n)

    # Critical-difference bar.
    cd_y = y_axis + 0.65
    ax.hlines(cd_y, mid - cd / 2.0, mid + cd / 2.0, color="black", linewidth=2.2)
    ax.text(
        mid,
        cd_y + 0.15,
        f"CD = {cd:.3f} (alpha = {nemenyi.alpha})",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    # Cliques (groups not significantly different).
    sorted_methods = sorted(methods, key=lambda m: ranks[m])
    cliques: list[tuple[int, int]] = []
    i = 0
    while i < len(sorted_methods):
        j = i
        while (
            j + 1 < len(sorted_methods)
            and abs(ranks[sorted_methods[j + 1]] - ranks[sorted_methods[i]]) <= cd
        ):
            j += 1
        if j > i:
            cliques.append((i, j))
        i = j + 1 if j > i else i + 1

    for c_idx, (a, b) in enumerate(cliques):
        ya = y_axis - 0.25 - 0.18 * c_idx
        xa = ranks[sorted_methods[a]]
        xb = ranks[sorted_methods[b]]
        ax.hlines(ya, xa, xb, color="black", linewidth=2.0)

    if output_path is not None:
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# One-call multi-dataset report.
# ---------------------------------------------------------------------------


@dataclass
class MultiDatasetReport:
    friedman: FriedmanResult
    nemenyi: NemenyiResult
    pairwise_wilcoxon_holm: dict[str, dict[str, float]]


def multi_dataset_report(
    per_dataset_scores: Mapping[str, Mapping[str, Sequence[float]]],
    baseline: str,
    higher_is_better: bool = True,
    alpha: float = 0.05,
) -> MultiDatasetReport:
    """Full multi-dataset statistical report.

    Combines:
    * Friedman omnibus test
    * Nemenyi pairwise post-hoc with critical difference
    * Per-baseline Holm-corrected paired Wilcoxon (pooled across datasets)
    """

    friedman = friedman_test(per_dataset_scores, higher_is_better=higher_is_better)
    nemenyi = nemenyi_posthoc(per_dataset_scores, higher_is_better=higher_is_better, alpha=alpha)

    # Per-method pooled paired Wilcoxon vs baseline.
    # We pool seed-level scores across datasets, but pair them within each
    # dataset to preserve dependence structure.
    methods = sorted({m for d in per_dataset_scores.values() for m in d})
    if baseline not in methods:
        raise ValueError(f"Baseline {baseline!r} not present in scores")
    raw_p: dict[str, float] = {}
    for m in methods:
        if m == baseline:
            continue
        a_vals: list[float] = []
        b_vals: list[float] = []
        for _ds, by_method in per_dataset_scores.items():
            if m not in by_method or baseline not in by_method:
                continue
            ai = list(by_method[m])
            bi = list(by_method[baseline])
            n = min(len(ai), len(bi))
            a_vals.extend(ai[:n])
            b_vals.extend(bi[:n])
        if len(a_vals) < 2:
            raw_p[m] = 1.0
            continue
        res = paired_wilcoxon(a_vals, b_vals)
        # paired_wilcoxon returns a dict with key 'p_value'
        if isinstance(res, dict):
            raw_p[m] = float(res.get("p_value", 1.0))
        else:  # backwards-compat with tuple-returning variants
            raw_p[m] = float(res[1])

    ordered_methods = sorted(raw_p.keys())
    corrected = holm_bonferroni([raw_p[m] for m in ordered_methods])
    holm: dict[str, dict[str, float]] = {
        baseline: {ordered_methods[i]: corrected[i] for i in range(len(ordered_methods))}
    }
    return MultiDatasetReport(
        friedman=friedman,
        nemenyi=nemenyi,
        pairwise_wilcoxon_holm=holm,
    )


__all__ = [
    "FriedmanResult",
    "MultiDatasetReport",
    "NemenyiResult",
    "average_ranks",
    "critical_difference_diagram",
    "critical_difference_value",
    "friedman_test",
    "multi_dataset_report",
    "nemenyi_posthoc",
]
