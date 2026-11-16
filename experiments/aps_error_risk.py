"""Estimate APS-like operator error risk from observable interaction telemetry."""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from experiments.rail_paper import make_aps_like_dataset, make_default_policies, rail_components


def features(events):
    rows = []
    for event in events:
        t = event.telemetry
        rows.append([t.decision_time_s - t.anchor_time_s, t.num_features_shown,
                     t.edit_count, t.focus_time_s])
    return np.asarray(rows)


def errors(events):
    return np.array([e.y_human != e.y_true for e in events])


def fit_risk(events, seed):
    model = CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=40, max_depth=1, min_samples_leaf=20,
                                   random_state=seed),
        method="sigmoid", cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        ensemble=True,
    )
    return model.fit(features(events), errors(events))


def evaluate(seed, error_cost, reject_cost):
    data = make_aps_like_dataset(seed)
    model = fit_risk(data.validation_events, seed)
    events = data.replay_events
    risk = model.predict_proba(features(events))[:, 1]
    wrong = errors(events)
    positive = np.array([e.y_true == 1 for e in events])
    gate = next(p for p in make_default_policies() if p.name == "rail_gated")
    diagnostics = [rail_components(e.telemetry, gate) for e in events]
    adjusted = np.array([gate.w_delta * d["delta"] - d["beta"] for d in diagnostics])
    zone = np.select([adjusted < gate.tau_min, adjusted > gate.tau_max],
                     ["fast", "slow"], default="central")
    baseline = np.array([d["score"] >= gate.theta for d in diagnostics])
    prior = errors(data.validation_events).mean()
    for name, admit in (("band", baseline), ("risk", risk * error_cost <= reject_cost)):
        for region in ("fast", "central", "slow", "all"):
            take = np.ones(len(events), dtype=bool) if region == "all" else zone == region
            accepted = take & admit
            withheld = take & ~admit
            yield dict(seed=seed, policy=name, zone=region, responses=take.sum(),
                       errors=wrong[take].sum(), observed_error=wrong[take].mean(),
                       predicted_risk=risk[take].mean(), admitted=accepted.sum(),
                       coverage=accepted.sum()/take.sum(), admitted_errors=wrong[accepted].sum(),
                       admitted_error=wrong[accepted].mean() if accepted.any() else None,
                       correct_withheld=(withheld & ~wrong).sum(),
                       correct_positive_withheld=(withheld & ~wrong & positive).sum(),
                       cost_per_response=(error_cost*wrong[accepted].sum()
                                          + reject_cost*withheld.sum())/take.sum(),
                       risk_brier=np.mean((risk[take]-wrong[take])**2),
                       prior_brier=np.mean((prior-wrong[take])**2),
                       error_cost=error_cost, reject_cost=reject_cost)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    parser.add_argument("--error-cost", type=float, default=5)
    parser.add_argument("--reject-cost", type=float, default=1)
    parser.add_argument("--output", type=Path, default=Path("publication_outputs/aps_risk/zones.csv"))
    args = parser.parse_args()
    if not (math.isfinite(args.error_cost) and args.error_cost > 0 and
            math.isfinite(args.reject_cost) and args.reject_cost >= 0):
        parser.error("Costs must be finite, with error cost > 0 and rejection cost >= 0")
    rows = [row for seed in args.seeds for row in evaluate(seed, args.error_cost, args.reject_cost)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
