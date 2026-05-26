"""Aggregate RAIL activity reports exported from the browser console."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _num_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    nums = sorted(v for v in (_safe_float(v) for v in values) if v is not None)
    if not nums:
        return {"n": 0, "mean": None, "sd": None, "min": None, "median": None, "max": None}
    return {
        "n": len(nums),
        "mean": mean(nums),
        "sd": stdev(nums) if len(nums) > 1 else 0.0,
        "min": nums[0],
        "median": median(nums),
        "max": nums[-1],
    }


@dataclass
class SessionSummary:
    source_file: str
    session_id: str | None
    participant_id: str | None
    condition: str | None
    task_batch: str | None
    exported_at: str | None
    row_values_included: bool
    rows_total: int
    rows_decided: int
    rows_ok: int
    rows_non_ok: int
    rows_undecided: int
    eligible_non_ok: int
    decision_rate: float | None
    eligible_non_ok_rate: float | None
    mean_vigilance: float | None
    median_vigilance: float | None
    mean_delta_s: float | None
    median_delta_s: float | None
    mean_focus_ms: float | None
    mean_edit_count: float | None
    mean_abs_error: float | None
    events_total: int
    dropped_events: int


def load_activity_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "RAIL.activity_report.v1":
        raise ValueError(f"{path} is not a RAIL.activity_report.v1 file")
    return report


def summarize_report(path: Path, report: dict[str, Any]) -> SessionSummary:
    session = report.get("session", {})
    study = session.get("study", {}) or {}
    aggregate = report.get("aggregate", {})
    rows_aggregate = aggregate.get("rows", {})
    events_aggregate = aggregate.get("events", {})
    rows = report.get("rows", [])

    vigilance = _num_summary(row.get("vigilance", {}).get("score") for row in rows)
    delta = _num_summary(row.get("timing", {}).get("delta_s") for row in rows)
    focus = _num_summary(row.get("timing", {}).get("focus_ms") for row in rows)
    edits = _num_summary(row.get("timing", {}).get("edit_count") for row in rows)
    abs_error = _num_summary(row.get("content_summary", {}).get("abs_error") for row in rows)

    return SessionSummary(
        source_file=str(path),
        session_id=session.get("session_id"),
        participant_id=study.get("participant_id"),
        condition=study.get("condition"),
        task_batch=study.get("task_batch"),
        exported_at=report.get("exportedAt"),
        row_values_included=bool(session.get("row_values_included", False)),
        rows_total=_safe_int(rows_aggregate.get("total", len(rows))),
        rows_decided=_safe_int(rows_aggregate.get("decided")),
        rows_ok=_safe_int(rows_aggregate.get("ok")),
        rows_non_ok=_safe_int(rows_aggregate.get("non_ok")),
        rows_undecided=_safe_int(rows_aggregate.get("undecided")),
        eligible_non_ok=_safe_int(rows_aggregate.get("eligible_non_ok")),
        decision_rate=_safe_float(rows_aggregate.get("decision_rate")),
        eligible_non_ok_rate=_safe_float(rows_aggregate.get("eligible_non_ok_rate")),
        mean_vigilance=_safe_float(vigilance["mean"]),
        median_vigilance=_safe_float(vigilance["median"]),
        mean_delta_s=_safe_float(delta["mean"]),
        median_delta_s=_safe_float(delta["median"]),
        mean_focus_ms=_safe_float(focus["mean"]),
        mean_edit_count=_safe_float(edits["mean"]),
        mean_abs_error=_safe_float(abs_error["mean"]),
        events_total=_safe_int(events_aggregate.get("total", len(report.get("events", [])))),
        dropped_events=_safe_int(events_aggregate.get("dropped")),
    )


def iter_report_paths(input_paths: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in input_paths:
        if input_path.is_dir():
            paths.extend(sorted(input_path.glob("**/rail-activity-report_*.json")))
            paths.extend(sorted(input_path.glob("**/*activity_report*.json")))
        else:
            paths.append(input_path)
    return sorted(dict.fromkeys(paths))


def write_session_csv(summaries: list[SessionSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(summaries[0]).keys()) if summaries else list(SessionSummary.__annotations__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def group_by_condition(summaries: list[SessionSummary]) -> list[dict[str, Any]]:
    grouped: dict[str, list[SessionSummary]] = {}
    for summary in summaries:
        grouped.setdefault(summary.condition or "unspecified", []).append(summary)

    rows: list[dict[str, Any]] = []
    for condition, items in sorted(grouped.items()):
        rows.append(
            {
                "condition": condition,
                "sessions": len(items),
                "participants": len({i.participant_id for i in items if i.participant_id}),
                "rows_total": sum(i.rows_total for i in items),
                "rows_decided": sum(i.rows_decided for i in items),
                "eligible_non_ok": sum(i.eligible_non_ok for i in items),
                "mean_decision_rate": mean([i.decision_rate for i in items if i.decision_rate is not None])
                if any(i.decision_rate is not None for i in items)
                else None,
                "mean_vigilance": mean([i.mean_vigilance for i in items if i.mean_vigilance is not None])
                if any(i.mean_vigilance is not None for i in items)
                else None,
                "mean_delta_s": mean([i.mean_delta_s for i in items if i.mean_delta_s is not None])
                if any(i.mean_delta_s is not None for i in items)
                else None,
                "mean_focus_ms": mean([i.mean_focus_ms for i in items if i.mean_focus_ms is not None])
                if any(i.mean_focus_ms is not None for i in items)
                else None,
                "mean_edit_count": mean([i.mean_edit_count for i in items if i.mean_edit_count is not None])
                if any(i.mean_edit_count is not None for i in items)
                else None,
            }
        )
    return rows


def write_condition_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "sessions",
        "participants",
        "rows_total",
        "rows_decided",
        "eligible_non_ok",
        "mean_decision_rate",
        "mean_vigilance",
        "mean_delta_s",
        "mean_focus_ms",
        "mean_edit_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_reports(input_paths: Iterable[Path], output_dir: Path) -> dict[str, Any]:
    report_paths = iter_report_paths(input_paths)
    summaries = [summarize_report(path, load_activity_report(path)) for path in report_paths]
    condition_rows = group_by_condition(summaries)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_session_csv(summaries, output_dir / "activity_session_summary.csv")
    write_condition_csv(condition_rows, output_dir / "activity_condition_summary.csv")
    (output_dir / "activity_summary.json").write_text(
        json.dumps(
            {
                "schema": "RAIL.activity_summary.v1",
                "reports": len(summaries),
                "session_summary": [asdict(s) for s in summaries],
                "condition_summary": condition_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"reports": len(summaries), "output_dir": str(output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate RAIL activity report JSON exports.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Activity report JSON files or directories.")
    parser.add_argument("--output-dir", type=Path, default=Path("publication_outputs/activity"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = aggregate_reports(args.inputs, args.output_dir)
    print(f"Aggregated {result['reports']} activity report(s) into {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
