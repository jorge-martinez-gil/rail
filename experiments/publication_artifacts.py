"""Helpers for journal-ready RAIL publication artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

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
}


def configure_matplotlib() -> None:
    """Apply a conservative, print-friendly Matplotlib style."""

    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": PUBLICATION_DPI,
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.linewidth": 0.45,
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
