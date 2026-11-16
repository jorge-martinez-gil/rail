"""Report observed label errors by adjusted deliberation zone, including withheld trials."""
import argparse
import csv
from collections import Counter
from pathlib import Path

from experiments.human_study import load_sessions, params_from_session
from experiments.rail_core import complexity_bonus


def aps_trials(seed):
    from experiments.rail_paper import (
        make_aps_like_dataset, make_default_policies, policy_admits, rail_components,
    )

    policy = next(p for p in make_default_policies() if p.name == "rail_gated")
    for event in make_aps_like_dataset(seed).replay_events:
        diag = rail_components(event.telemetry, policy)
        adjusted = policy.w_delta * diag["delta"] - diag["beta"]
        yield (adjusted, policy.tau_min, policy.tau_max, event.y_human != event.y_true,
               policy_admits(policy, diag["score"]), event.y_true == 1, "unrecorded")


def human_trials(inputs):
    for session in load_sessions(inputs):
        p = params_from_session(session["rail_defaults"])
        for r in session["records"]:
            beta = complexity_bonus(r["n_features"], r["edits"], r["focus_s"], p)
            adjusted = p.w_delta * r["delta_t_s"] - beta
            interruptions = r.get("interruptions")
            status = "unrecorded" if interruptions is None else ("yes" if interruptions > 0 else "no")
            yield (adjusted, p.tau_min, p.tau_max, r["operator_label"] != r["truth"],
                   bool(r["admitted"]), r["truth"] == "FAULT", status)


def summarize(trials, interruption_groups):
    groups = {(zone, "all"): Counter() for zone in ("fast", "central", "slow")}
    for status in interruption_groups:
        groups["slow", status] = Counter()
    for adjusted, lower, upper, error, admitted, positive, status in trials:
        zone = "fast" if adjusted < lower else "slow" if adjusted > upper else "central"
        keys = [(zone, "all")]
        if zone == "slow":
            keys.append((zone, status))
        for key in keys:
            counts = groups.setdefault(key, Counter())
            counts["responses"] += 1
            counts["errors"] += int(error)
            counts["withheld"] += int(not admitted)
            counts["correct_withheld"] += int(not error and not admitted)
            counts["correct_positive_withheld"] += int(positive and not error and not admitted)
    for (zone, interruption), c in groups.items():
        yield [zone, interruption, c["responses"], c["errors"],
               c["errors"] / c["responses"] if c["responses"] else None,
               c["withheld"], c["correct_withheld"], c["correct_positive_withheld"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["aps", "human"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sessions", nargs="+", type=Path, default=[Path("real tests")])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.source == "aps":
        rows = list(summarize(aps_trials(args.seed), ["unrecorded"]))
    else:
        for path in args.sessions:
            if not path.exists():
                parser.error(f"Session path does not exist: {path}")
        rows = list(summarize(human_trials(args.sessions), ["yes", "no"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["zone", "interrupted", "responses", "errors", "error_rate",
                         "withheld", "correct_withheld", "correct_positive_withheld"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
