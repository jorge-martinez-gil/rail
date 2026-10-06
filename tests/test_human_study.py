import json

from experiments.human_study import analyse, params_from_session
from experiments.rail_core import admission_diagnostics

RAIL_DEFAULTS = {
    "tauMin": 0.8,
    "tauMax": 6,
    "k": 1.2,
    "wDelta": 1,
    "theta": 0.5,
    "wf": 0.05,
    "we": 0.15,
    "ws": 0.02,
}


def _record(trial, delta, admitted, operator_label, truth, contaminated):
    diag = admission_diagnostics(
        delta_sec=delta,
        num_features=4,
        edit_count=0,
        focus_seconds=delta,
        params=params_from_session(RAIL_DEFAULTS),
    )
    return {
        "trial": trial,
        "alertId": f"ALT-{1000 + trial}",
        "condition": "overload",
        "t_render_ms": 1000.0 * trial,
        "t_anchor_ms": 1000.0 * trial + 100.0,
        "t_decision_ms": 1000.0 * trial + 100.0 + delta * 1000.0,
        "delta_t_s": delta,
        "focus_s": delta,
        "edits": 0,
        "n_features": 4,
        "interruptions": 0,
        "queue_depth": trial,
        "beta": round(diag["beta"], 3),
        "V": round(diag["score"], 4),
        "admitted": admitted,
        "operator_label": operator_label,
        "model_flag": "NORMAL",
        "truth": truth,
        "contaminated": contaminated,
        "note_len": 0,
    }


def _session(participant="p1", started="2026-07-01T10:00:00.000Z"):
    # deltas chosen so trials 1-3 fall inside the window (admitted) and
    # trial 4 far above it (withheld); trial 4 is the contaminated one.
    return {
        "schema": "rail-human-telemetry-v1",
        "participant": participant,
        "condition": "overload",
        "started": started,
        "rail_defaults": dict(RAIL_DEFAULTS),
        "records": [
            _record(1, 3.0, 1, "NORMAL", "NORMAL", 0),
            _record(2, 3.5, 1, "FAULT", "FAULT", 0),
            _record(3, 4.0, 1, "NORMAL", "NORMAL", 0),
            _record(4, 10.0, 0, "FAULT", "NORMAL", 1),
        ],
    }


def test_params_from_session_maps_console_keys():
    params = params_from_session(RAIL_DEFAULTS)
    assert params.tau_min == 0.8
    assert params.tau_max == 6.0
    assert params.theta == 0.5
    assert params.w_features == 0.05
    assert params.w_edits == 0.15
    assert params.w_focus == 0.02


def test_analyse_audits_and_summarises(tmp_path):
    input_dir = tmp_path / "sessions"
    input_dir.mkdir()
    (input_dir / "rail_p1_overload.json").write_text(
        json.dumps(_session("p1", "2026-07-01T10:00:00.000Z")), encoding="utf-8"
    )
    (input_dir / "rail_p2_overload.json").write_text(
        json.dumps(_session("p2", "2026-07-02T10:00:00.000Z")), encoding="utf-8"
    )

    output_dir = tmp_path / "out"
    summary = analyse([input_dir], output_dir, tolerance=0.005)

    pooled = summary["pooled"]
    assert pooled["n_participants"] == 2
    assert pooled["n_trials"] == 8
    assert pooled["v_mismatches"] == 0
    assert pooled["admit_mismatches"] == 0
    assert pooled["n_contaminated"] == 2
    assert pooled["n_contaminated_admitted"] == 0

    contract = summary["contamination_contract"]
    assert contract["base_contamination_rate"] == 0.25
    assert contract["false_admission_rate"] == 0.0
    assert contract["admitted_contamination_rate"] == 0.0
    assert contract["clean_admission_rate"] == 1.0

    # aliases are assigned by session start time
    aliases = {p["participant"]: p["alias"] for p in summary["participants"]}
    assert aliases == {"p1": "P1", "p2": "P2"}

    assert (output_dir / "human_pilot_trials.csv").exists()
    assert (output_dir / "human_pilot_participants.csv").exists()
    assert (output_dir / "human_pilot_summary.json").exists()
    table = (output_dir / "table_human_pilot.tex").read_text(encoding="utf-8")
    assert "Pooled & 8 & 6 & 2 & 0" in table
