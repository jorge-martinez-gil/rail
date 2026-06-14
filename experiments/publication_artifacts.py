"""Helpers for journal-ready RAIL publication artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

PUBLICATION_DPI = 600
DEFAULT_FIGURE_FORMATS = ("png", "pdf", "svg")

METHOD_LABELS = {
    "static": "Static",
    "always": "Unfiltered",
    "confidence_gated": "Confidence-gated",
    "loss_gated": "Loss-gated",
    "margin_gated": "Margin-gated",
    "rail_gated": "RAIL-gated",
    "rail_weighted": "RAIL-weighted",
    "STATIC": "Static",
    "ALWAYS": "Unfiltered",
    "GATED": "RAIL-gated",
    "WEIGHTED": "RAIL-weighted",
}

METHOD_COLORS = {
    "static": "#4D4D4D",
    "always": "#D55E00",
    "confidence_gated": "#0072B2",
    "loss_gated": "#009E73",
    "margin_gated": "#CC79A7",
    "rail_gated": "#E69F00",
    "rail_weighted": "#56B4E9",
    "STATIC": "#4D4D4D",
    "ALWAYS": "#D55E00",
    "GATED": "#E69F00",
    "WEIGHTED": "#56B4E9",
}

METHOD_MARKERS = {
    "static": "o",
    "always": "s",
    "confidence_gated": "^",
    "loss_gated": "D",
    "margin_gated": "P",
    "rail_gated": "X",
    "rail_weighted": "*",
    "STATIC": "o",
    "ALWAYS": "s",
    "GATED": "X",
    "WEIGHTED": "*",
}

RAIL_METHODS = frozenset({"rail_gated", "rail_weighted", "GATED", "WEIGHTED"})


def configure_matplotlib() -> None:
    """Apply a conservative, print-friendly Matplotlib style."""

    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": PUBLICATION_DPI,
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.03,
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 9.6,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "legend.fontsize": 8.4,
            "legend.frameon": False,
            "legend.handlelength": 1.35,
            "legend.handletextpad": 0.35,
            "legend.columnspacing": 0.9,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "grid.color": "#9A9A9A",
            "grid.linewidth": 0.42,
            "grid.alpha": 0.28,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def method_label(method: str) -> str:
    """Return a compact manuscript label for a policy or method id."""

    return METHOD_LABELS.get(method, method.replace("_", "-"))


def method_color(method: str) -> str:
    """Return a color-blind-aware color for a policy or method id."""

    return METHOD_COLORS.get(method, "#333333")


def method_is_rail(method: str) -> bool:
    """Return whether a method id belongs to the RAIL method family."""

    return method in RAIL_METHODS or method.lower() in RAIL_METHODS


def style_panel_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    """Apply a consistent journal panel style to an axes object."""

    ax.grid(True, axis=grid_axis)
    ax.tick_params(direction="out")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#222222")
        ax.spines[spine].set_linewidth(0.7)


def save_publication_figure(
    fig: plt.Figure,
    stem: str | Path,
    formats: Sequence[str] = DEFAULT_FIGURE_FORMATS,
) -> list[str]:
    """Save a figure as high-DPI raster and vector masters.

    The provided path is treated as a stem. If it includes a suffix, that suffix
    is replaced by each requested output format.
    """

    path = Path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        out = path.with_suffix(f".{fmt}")
        kwargs: dict[str, Any] = {
            "format": fmt,
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
            kwargs["dpi"] = PUBLICATION_DPI
        fig.savefig(out, **kwargs)
        created.append(str(out))
    plt.close(fig)
    return created


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table."""

    display = df.copy()
    display = display.astype(object).where(pd.notna(display), "")
    headers = [str(col) for col in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in display.iterrows():
        cells = [str(row[col]) for col in display.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_table(df: pd.DataFrame, path: str | Path) -> str:
    """Write a Markdown table without adding a tabulate dependency."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dataframe_to_markdown(df), encoding="utf-8")
    return str(out)


def build_winner_audit(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str = "dataset",
    method_col: str = "method",
    greater_is_better: bool = True,
    tie_tol: float = 1e-12,
) -> pd.DataFrame:
    """Build an honest best-method audit, including the best RAIL-family method.

    The audit does not change any result. It simply records whether at least one
    RAIL method is tied for the best value within each group.
    """

    required = {group_col, method_col, metric_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"winner audit missing column(s): {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for group, subset in df.groupby(group_col, sort=False):
        work = subset[[method_col, metric_col]].copy()
        work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
        work = work.dropna(subset=[metric_col])
        if work.empty:
            continue

        idx = work[metric_col].idxmax() if greater_is_better else work[metric_col].idxmin()
        best_value = float(work.loc[idx, metric_col])
        if greater_is_better:
            best_mask = work[metric_col] >= best_value - tie_tol
        else:
            best_mask = work[metric_col] <= best_value + tie_tol
        best_methods = [str(v) for v in work.loc[best_mask, method_col]]

        rail_work = work[work[method_col].map(lambda method: method_is_rail(str(method)))]
        if rail_work.empty:
            best_rail_method = ""
            best_rail_value = float("nan")
            rail_family_wins = False
            margin = float("nan")
        else:
            rail_idx = (
                rail_work[metric_col].idxmax()
                if greater_is_better
                else rail_work[metric_col].idxmin()
            )
            best_rail_method = str(rail_work.loc[rail_idx, method_col])
            best_rail_value = float(rail_work.loc[rail_idx, metric_col])
            rail_family_wins = method_is_rail(best_rail_method) and (
                best_rail_method in best_methods
            )
            margin = (
                best_rail_value - best_value if greater_is_better else best_value - best_rail_value
            )

        rows.append(
            {
                group_col: group,
                "metric": metric_col,
                "best_method": ", ".join(best_methods),
                "best_value": best_value,
                "best_rail_method": best_rail_method,
                "best_rail_value": best_rail_value,
                "rail_family_wins": rail_family_wins,
                "rail_margin_to_best": margin,
            }
        )
    return pd.DataFrame(rows)


def write_winner_audit(
    df: pd.DataFrame,
    stem: str | Path,
    metric_col: str,
    group_col: str = "dataset",
    method_col: str = "method",
    greater_is_better: bool = True,
) -> pd.DataFrame:
    """Write CSV and Markdown views of :func:`build_winner_audit`."""

    audit = build_winner_audit(
        df,
        metric_col=metric_col,
        group_col=group_col,
        method_col=method_col,
        greater_is_better=greater_is_better,
    )
    path = Path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(path.with_suffix(".csv"), index=False)

    display = audit.copy()
    for col in ("best_value", "best_rail_value", "rail_margin_to_best"):
        display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    display["rail_family_wins"] = display["rail_family_wins"].map(
        lambda v: "yes" if bool(v) else "no"
    )
    write_markdown_table(display, path.with_suffix(".md"))
    return audit


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".pdf", ".svg", ".tif", ".tiff"}:
        return "figure"
    if suffix in {".tex", ".csv", ".md"} and path.name != "JOURNAL_ARTIFACTS.md":
        return "table"
    if suffix in {".json"}:
        return "metadata"
    return "other"


def collect_artifacts(root: str | Path) -> list[dict[str, Any]]:
    """Collect generated artifact metadata for reviewer-facing indexes."""

    base = Path(root)
    exts = {".png", ".pdf", ".svg", ".tex", ".csv", ".json", ".md"}
    rows: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        if path.name == "JOURNAL_ARTIFACTS.md":
            continue
        rows.append(
            {
                "kind": _artifact_kind(path),
                "format": path.suffix.lower().lstrip("."),
                "path": path.relative_to(base).as_posix(),
                "size_kb": round(path.stat().st_size / 1024.0, 1),
            }
        )
    return rows


def write_artifact_index(
    root: str | Path,
    metadata: dict[str, Any] | None = None,
    filename: str = "JOURNAL_ARTIFACTS.md",
) -> str:
    """Write a compact index of generated figures, tables, and metadata."""

    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    rows = collect_artifacts(base)
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row)

    metadata = metadata or {}
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# RAIL Journal Artifact Index",
        "",
        f"Generated at: `{now}`",
        f"Run status: `{metadata.get('status', 'unknown')}`",
        f"Mode: `{metadata.get('mode', 'unknown')}`",
        f"Runs: `{metadata.get('runs', 'unknown')}`",
        f"Seed: `{metadata.get('seed', 'unknown')}`",
        "",
        "Use `.pdf` or `.svg` files as vector figure masters for manuscript submission, and `.png` files as 600 DPI raster fallbacks.",
        "",
    ]

    for kind in ("figure", "table", "metadata", "other"):
        kind_rows = by_kind.get(kind, [])
        if not kind_rows:
            continue
        title = "Metadata" if kind == "metadata" else kind.capitalize() + "s"
        lines.extend([f"## {title}", "", "| Format | Size KB | Path |", "| --- | ---: | --- |"])
        for row in kind_rows:
            lines.append(f"| {row['format']} | {row['size_kb']} | `{row['path']}` |")
        lines.append("")

    manifest_path = base / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            lines.extend(
                [
                    "## Reproducibility Snapshot",
                    "",
                    "```json",
                    json.dumps(
                        {
                            "schema": manifest.get("schema"),
                            "mode": manifest.get("mode"),
                            "runs": manifest.get("runs"),
                            "seed": manifest.get("seed"),
                            "started_at": manifest.get("started_at"),
                            "completed_at": manifest.get("completed_at"),
                            "status": manifest.get("status"),
                            "components": manifest.get("components"),
                        },
                        indent=2,
                    ),
                    "```",
                    "",
                ]
            )
        except json.JSONDecodeError:
            pass

    out = base / filename
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def wide_metric_table(
    df: pd.DataFrame,
    value_col: str,
    std_col: str,
    datasets: Iterable[str],
    methods: Iterable[str],
    decimals: int = 3,
) -> pd.DataFrame:
    """Create a manuscript-friendly wide mean +/- std table."""

    rows_by_key = {(r.dataset, r.method): r for r in df.itertuples(index=False)}
    table_rows: list[dict[str, str]] = []
    for method in methods:
        row: dict[str, str] = {"Method": method_label(method)}
        for dataset in datasets:
            item = rows_by_key[(dataset, method)]
            mean = getattr(item, value_col)
            std = getattr(item, std_col)
            row[dataset] = f"{mean:.{decimals}f} +/- {std:.{decimals}f}"
        table_rows.append(row)
    return pd.DataFrame(table_rows)
