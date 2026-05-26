import csv
import json

from experiments.activity_report import aggregate_reports, load_activity_report, summarize_report


def _sample_report(participant_id="P001", condition="rail"):
    return {
        "schema": "RAIL.activity_report.v1",
        "exportedAt": "2026-05-23T10:00:00.000Z",
        "session": {
            "session_id": "session-1",
            "study": {
                "participant_id": participant_id,
                "condition": condition,
                "task_batch": "pilot-1",
            },
            "row_values_included": False,
            "recording_enabled_at_export": True,
            "local_only": True,
        },
        "params": {"theta": 0.6, "tauMin": 0.8, "tauMax": 6.0, "k": 1.2},
        "aggregate": {
            "rows": {
                "total": 2,
                "decided": 2,
                "ok": 1,
                "non_ok": 1,
                "undecided": 0,
                "eligible_non_ok": 1,
                "decision_rate": 1.0,
                "eligible_non_ok_rate": 1.0,
            },
            "events": {"total": 4, "dropped": 0},
        },
        "rows": [
            {
                "row_id": "row-1",
                "source": "csv",
                "status": "ok",
                "timing": {"delta_s": 2.0, "focus_ms": 900, "edit_count": 0},
                "vigilance": {"score": 0.75},
                "content_summary": {"abs_error": 0.1},
            },
            {
                "row_id": "row-2",
                "source": "csv",
                "status": "non-ok",
                "timing": {"delta_s": 3.0, "focus_ms": 1200, "edit_count": 1},
                "vigilance": {"score": 0.85},
                "content_summary": {"abs_error": 0.3},
            },
        ],
        "events": [{"id": 1, "at": "2026-05-23T10:00:01.000Z", "type": "recording_started", "details": {}}],
    }


def test_summarize_report_extracts_study_metadata(tmp_path):
    path = tmp_path / "rail-activity-report_1.json"
    path.write_text(json.dumps(_sample_report()), encoding="utf-8")

    summary = summarize_report(path, load_activity_report(path))

    assert summary.participant_id == "P001"
    assert summary.condition == "rail"
    assert summary.rows_total == 2
    assert summary.eligible_non_ok == 1
    assert summary.mean_vigilance == 0.8
    assert summary.median_delta_s == 2.5


def test_aggregate_reports_writes_session_and_condition_outputs(tmp_path):
    input_dir = tmp_path / "reports"
    input_dir.mkdir()
    (input_dir / "rail-activity-report_1.json").write_text(json.dumps(_sample_report("P001", "rail")), encoding="utf-8")
    (input_dir / "rail-activity-report_2.json").write_text(json.dumps(_sample_report("P002", "baseline")), encoding="utf-8")

    output_dir = tmp_path / "out"
    result = aggregate_reports([input_dir], output_dir)

    assert result["reports"] == 2
    session_rows = list(csv.DictReader((output_dir / "activity_session_summary.csv").open(encoding="utf-8")))
    condition_rows = list(csv.DictReader((output_dir / "activity_condition_summary.csv").open(encoding="utf-8")))

    assert len(session_rows) == 2
    assert {row["condition"] for row in condition_rows} == {"baseline", "rail"}
    assert (output_dir / "activity_summary.json").exists()
