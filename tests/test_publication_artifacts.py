import pandas as pd

from experiments.publication_artifacts import build_winner_audit, method_is_rail


def test_method_is_rail_handles_canonical_ids():
    assert method_is_rail("rail_gated")
    assert method_is_rail("rail_weighted")
    assert method_is_rail("GATED")
    assert method_is_rail("WEIGHTED")
    assert not method_is_rail("confidence_gated")


def test_winner_audit_reports_rail_family_win_without_changing_values():
    df = pd.DataFrame(
        [
            {"dataset": "A", "method": "always", "final_macro_f1_mean": 0.40},
            {"dataset": "A", "method": "rail_gated", "final_macro_f1_mean": 0.45},
            {"dataset": "A", "method": "rail_weighted", "final_macro_f1_mean": 0.44},
            {"dataset": "B", "method": "always", "final_macro_f1_mean": 0.50},
            {"dataset": "B", "method": "rail_gated", "final_macro_f1_mean": 0.48},
        ]
    )

    audit = build_winner_audit(df, metric_col="final_macro_f1_mean")
    rows = audit.set_index("dataset")

    assert rows.loc["A", "rail_family_wins"]
    assert rows.loc["A", "best_method"] == "rail_gated"
    assert rows.loc["A", "rail_margin_to_best"] == 0.0

    assert not rows.loc["B", "rail_family_wins"]
    assert rows.loc["B", "best_method"] == "always"
    assert rows.loc["B", "rail_margin_to_best"] < 0.0


def test_winner_audit_supports_lower_is_better_metrics():
    df = pd.DataFrame(
        [
            {"dataset": "A", "method": "always", "contaminated": 10.0},
            {"dataset": "A", "method": "rail_gated", "contaminated": 4.0},
            {"dataset": "A", "method": "rail_weighted", "contaminated": 9.0},
        ]
    )

    audit = build_winner_audit(df, metric_col="contaminated", greater_is_better=False)

    assert bool(audit.loc[0, "rail_family_wins"])
    assert audit.loc[0, "best_method"] == "rail_gated"
    assert audit.loc[0, "rail_margin_to_best"] == 0.0
