import re
from pathlib import Path


PAGE = Path(__file__).parents[1] / "app" / "labeling.html"


def test_labeling_page_is_standalone_and_has_twenty_trials():
    html = PAGE.read_text(encoding="utf-8")
    assert "<script src=" not in html
    specs = html.split("const ALERT_SPECS", 1)[1].split("const SENSOR_DEFS", 1)[0]
    assert len(re.findall(r'^\s+\["', specs, flags=re.MULTILINE)) == 20


def test_labeling_export_matches_human_study_contract():
    html = PAGE.read_text(encoding="utf-8")
    assert 'schema: "rail-human-telemetry-v1"' in html
    assert 'theta: 0.5' in html
    for field in (
        "delta_t_s",
        "focus_s",
        "edits",
        "n_features",
        "interruptions",
        "queue_depth",
        "beta",
        "V",
        "admitted",
        "operator_label",
        "model_flag",
        "truth",
        "contaminated",
    ):
        assert field in html
