"""Calibrate APS-like timing gates at matched validation coverage and evaluate on replay."""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from experiments.rail_paper import (
    SklearnOnlineClassifier, make_aps_like_dataset, make_default_policies,
    rail_components, threshold_for_target_yield,
)


def factors(events, policy):
    pairs = []
    for event in events:
        diag = rail_components(event.telemetry, policy)
        pairs.append((diag["s_fast"], diag["s_slow"]))
    return np.asarray(pairs).T


def evaluate(seed, coverages):
    data = make_aps_like_dataset(seed)
    gate = next(p for p in make_default_policies() if p.name == "rail_gated")
    val_fast, val_slow = factors(data.validation_events, gate)
    fast, slow = factors(data.replay_events, gate)
    truth = np.array([e.y_true for e in data.replay_events])
    errors = np.array([e.y_true != e.y_human for e in data.replay_events])
    base = SklearnOnlineClassifier(classes=[0, 1], random_state=seed)
    base.fit_initial(data.X_warmup, data.y_warmup)
    reference = np.mean(val_fast * val_slow >= gate.theta)
    targets = coverages or [reference]
    for exponent in (1.0, 0.5, 0.2, 0.0):
        validation = val_fast * val_slow**exponent
        replay = fast * slow**exponent
        settings = [("calibrated", target, threshold_for_target_yield(validation, target))
                    for target in targets]
        if exponent == 1:
            settings.insert(0, ("fixed", reference, gate.theta))
        for method, target, threshold in settings:
            mask = replay >= threshold
            model = base.clone()
            for i in np.flatnonzero(mask):
                event = data.replay_events[i]
                model.update(event.x, event.y_human)
            yield dict(seed=seed, method=method, slow_exponent=exponent,
                       target_coverage=target, threshold=threshold,
                       validation_coverage=np.mean(validation >= threshold),
                       replay_coverage=mask.mean(), admitted=int(mask.sum()),
                       errors=int(errors[mask].sum()),
                       admitted_error=errors[mask].mean() if mask.any() else None,
                       coverage_negative=mask[truth == 0].mean(),
                       coverage_positive=mask[truth == 1].mean(),
                       macro_f1=model.evaluate_macro_f1(data.X_test, data.y_test))


def paired_summary(rows):
    """Compare each variant with the calibrated two-sided gate on the same seed."""
    frame = pd.DataFrame(rows)
    calibrated = frame[frame.method == "calibrated"].copy()
    # In default mode, the matched reference coverage can differ between seeds.
    calibrated["target_index"] = calibrated.groupby(["seed", "slow_exponent"]).cumcount()
    metrics = ["admitted_error", "replay_coverage", "coverage_negative",
               "coverage_positive", "macro_f1"]
    reference = calibrated[calibrated.slow_exponent == 1].set_index(["seed", "target_index"])
    paired = calibrated.join(reference[metrics], on=["seed", "target_index"], rsuffix="_ref")
    summary = []
    for (_, exponent), group in paired.groupby(["target_index", "slow_exponent"]):
        for metric in metrics:
            delta = (group[metric] - group[metric + "_ref"]).dropna()
            n = len(delta)
            mean = delta.mean()
            half = t.ppf(0.975, n - 1) * delta.sem() if n > 1 else np.nan
            summary.append(dict(mean_target=group.target_coverage.mean(), slow_exponent=exponent,
                                metric=metric, mean_value=group[metric].mean(), paired_seeds=n,
                                mean_delta=mean, ci95_low=mean-half, ci95_high=mean+half))
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--coverage", nargs="+", type=float,
                        help="Target fractions (default: fixed gate's validation coverage)")
    parser.add_argument("--output", type=Path,
                        default=Path("publication_outputs/aps_coverage/results.csv"))
    args = parser.parse_args()
    if args.coverage and any(not 0 < c <= 1 for c in args.coverage):
        parser.error("coverage must be in (0, 1]")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct for paired intervals")
    rows = [row for seed in args.seeds for row in evaluate(seed, args.coverage)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    paired_summary(rows).to_csv(args.output.with_suffix(".paired.csv"), index=False)


if __name__ == "__main__":
    main()
