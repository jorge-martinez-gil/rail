"""Analyse real-operator pilot telemetry exported by the RAIL console.

Input: ``rail-human-telemetry-v1`` JSON session files (one per participant
and condition), as exported by the console's study mode. Each file carries
the session's admission parameters (``rail_defaults``) and a per-trial
record with raw telemetry (anchored deliberation, focus time, edits,
features inspected), the recorded vigilance score ``V``, the gate decision,
the operator's label, the ground-truth label, and the contamination flag.

The script

1. re-computes ``beta``, ``V``, and the admission decision for every trial
   from the *raw* telemetry through :mod:`experiments.rail_core` (the same
   reference implementation the paper uses) and cross-checks them against
   the values the console recorded -- a full client/reference consistency
   audit;
2. estimates the contamination-contract quantities
   (:func:`experiments.rail_core.contamination_contract`) on the pooled
   trials: base rate, false-admission rate, clean retention, admitted-stream
   contamination, plug-in Bayes bound, and admission efficiency;
3. compares vigilance on contaminated vs. clean trials (rank AUC plus an
   exact permutation test on the mean difference); and
4. writes per-trial and per-participant CSVs, a JSON summary, the LaTeX
   table used in the manuscript, and a two-panel figure.

Usage::

    python -m experiments.human_study "real tests" \
        --output-dir publication_outputs/human_study

Participants are pseudonymised in the console; the script additionally
assigns P1, P2, ... by session start time for use in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from experiments.rail_core import (
    AdmissionParams,
    admission_diagnostics,
    contamination_contract,
)

SCHEMA = "rail-human-telemetry-v1"

#: Maximum number of label permutations for the exact test before falling
#: back to seeded Monte Carlo sampling.
EXACT_PERMUTATION_CAP = 500_000
MC_PERMUTATIONS = 100_000
MC_SEED = 20260710


def params_from_session(defaults: dict[str, Any]) -> AdmissionParams:
    """Map the console's ``rail_defaults`` block onto :class:`AdmissionParams`."""

    return AdmissionParams(
        tau_min=float(defaults["tauMin"]),
        tau_max=float(defaults["tauMax"]),
        k=float(defaults["k"]),
        theta=float(defaults["theta"]),
        w_delta=float(defaults.get("wDelta", 1.0)),
        w_features=float(defaults.get("wf", 0.0)),
        w_edits=float(defaults.get("we", 0.0)),
        w_focus=float(defaults.get("ws", 0.0)),
    )


@dataclass
class ParticipantSummary:
    participant: str
    alias: str
    condition: str
    started: str
    n_trials: int
    duration_s: float
    n_admitted: int
    n_contaminated: int
    n_contaminated_admitted: int
    operator_accuracy: float
    operator_accuracy_admitted: float | None
    operator_accuracy_withheld: float | None
    model_accuracy: float
    median_delta_s: float
    mean_focus_s: float
    mean_vigilance: float
    trials_with_interruptions: int
    max_queue_depth: int
    v_mismatches: int
    admit_mismatches: int
    max_abs_v_deviation: float


def load_sessions(inputs: list[Path]) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.glob("**/*.json")))
        else:
            paths.append(input_path)
    sessions = []
    for path in sorted(dict.fromkeys(paths)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            continue
        data["_source"] = str(path)
        sessions.append(data)
    if not sessions:
        raise SystemExit(f"no {SCHEMA} files found under {[str(p) for p in inputs]}")
    sessions.sort(key=lambda s: s.get("started", ""))
    for idx, session in enumerate(sessions, start=1):
        session["_alias"] = f"P{idx}"
    return sessions


def audit_session(session: dict[str, Any], tolerance: float) -> ParticipantSummary:
    """Cross-check recorded scores against the reference implementation."""

    params = params_from_session(session["rail_defaults"])
    records = session["records"]
    v_mismatches = 0
    admit_mismatches = 0
    max_dev = 0.0
    for record in records:
        diag = admission_diagnostics(
            delta_sec=float(record["delta_t_s"]),
            num_features=int(record["n_features"]),
            edit_count=int(record["edits"]),
            focus_seconds=float(record["focus_s"]),
            params=params,
        )
        dev = abs(diag["score"] - float(record["V"]))
        max_dev = max(max_dev, dev)
        if dev > tolerance:
            v_mismatches += 1
        if int(diag["eligible"]) != int(record["admitted"]):
            admit_mismatches += 1
        record["_recomputed_v"] = diag["score"]
        record["_recomputed_beta"] = diag["beta"]

    admitted = [r for r in records if r["admitted"]]
    withheld = [r for r in records if not r["admitted"]]

    def _acc(rows: list[dict[str, Any]]) -> float | None:
        if not rows:
            return None
        return mean(1.0 if r["operator_label"] == r["truth"] else 0.0 for r in rows)

    duration_s = (
        (records[-1]["t_decision_ms"] - records[0]["t_render_ms"]) / 1000.0 if records else 0.0
    )
    return ParticipantSummary(
        participant=str(session["participant"]),
        alias=str(session["_alias"]),
        condition=str(session["condition"]),
        started=str(session.get("started", "")),
        n_trials=len(records),
        duration_s=round(duration_s, 1),
        n_admitted=len(admitted),
        n_contaminated=sum(int(r["contaminated"]) for r in records),
        n_contaminated_admitted=sum(int(r["contaminated"]) for r in admitted),
        operator_accuracy=_acc(records) or 0.0,
        operator_accuracy_admitted=_acc(admitted),
        operator_accuracy_withheld=_acc(withheld),
        model_accuracy=mean(1.0 if r["model_flag"] == r["truth"] else 0.0 for r in records),
        median_delta_s=median(float(r["delta_t_s"]) for r in records),
        mean_focus_s=mean(float(r["focus_s"]) for r in records),
        mean_vigilance=mean(float(r["V"]) for r in records),
        trials_with_interruptions=sum(1 for r in records if int(r["interruptions"]) > 0),
        max_queue_depth=max(int(r["queue_depth"]) for r in records),
        v_mismatches=v_mismatches,
        admit_mismatches=admit_mismatches,
        max_abs_v_deviation=max_dev,
    )


def permutation_test(values: list[float], flags: list[bool]) -> dict[str, float | str]:
    """One-sided test that flagged (contaminated) trials have lower values.

    Exact over all label placements when feasible, otherwise seeded Monte
    Carlo. Also reports the rank AUC (P(clean > contaminated) + 0.5 ties).
    """

    n = len(values)
    n_flagged = sum(flags)
    flagged = [v for v, f in zip(values, flags, strict=True) if f]
    clean = [v for v, f in zip(values, flags, strict=True) if not f]
    if not flagged or not clean:
        return {"method": "undefined", "p_value": float("nan"), "auc": float("nan")}
    observed = mean(clean) - mean(flagged)
    pairs = len(flagged) * len(clean)
    auc = (
        sum(1.0 for c in flagged for cl in clean if cl > c)
        + 0.5 * sum(1.0 for c in flagged for cl in clean if cl == c)
    ) / pairs

    total_sum = sum(values)
    n_combos = math.comb(n, n_flagged)
    hits = 0
    if n_combos <= EXACT_PERMUTATION_CAP:
        for combo in itertools.combinations(range(n), n_flagged):
            combo_sum = sum(values[i] for i in combo)
            diff = (total_sum - combo_sum) / (n - n_flagged) - combo_sum / n_flagged
            if diff >= observed - 1e-12:
                hits += 1
        return {
            "method": "exact",
            "n_permutations": n_combos,
            "observed_mean_difference": observed,
            "p_value": hits / n_combos,
            "auc": auc,
        }
    import random

    rng = random.Random(MC_SEED)
    idx = list(range(n))
    for _ in range(MC_PERMUTATIONS):
        sample = rng.sample(idx, n_flagged)
        combo_sum = sum(values[i] for i in sample)
        diff = (total_sum - combo_sum) / (n - n_flagged) - combo_sum / n_flagged
        if diff >= observed - 1e-12:
            hits += 1
    return {
        "method": "monte-carlo",
        "n_permutations": MC_PERMUTATIONS,
        "observed_mean_difference": observed,
        "p_value": (hits + 1) / (MC_PERMUTATIONS + 1),
        "auc": auc,
    }


def pooled_records(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for session in sessions:
        for record in session["records"]:
            row = dict(record)
            row["participant"] = session["participant"]
            row["alias"] = session["_alias"]
            rows.append(row)
    return rows


def write_trials_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "alias",
        "participant",
        "condition",
        "trial",
        "alertId",
        "delta_t_s",
        "focus_s",
        "edits",
        "n_features",
        "interruptions",
        "queue_depth",
        "beta",
        "V",
        "_recomputed_v",
        "admitted",
        "operator_label",
        "model_flag",
        "truth",
        "contaminated",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_participants_csv(summaries: list[ParticipantSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def latex_table(summaries: list[ParticipantSummary], pooled: dict[str, Any]) -> str:
    def pct(x: float | None) -> str:
        return "--" if x is None else f"{100.0 * x:.0f}"

    lines = [
        r"% Auto-generated by experiments/human_study.py -- do not edit by hand.",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Participant & Events & Admitted & Contam. & Contam.\ admitted"
        r" & Op.\ acc.\ (\%) & Median $\Delta$ (s) & Mean $V$ \\",
        r"\midrule",
    ]
    for s in summaries:
        lines.append(
            f"{s.alias} & {s.n_trials} & {s.n_admitted} & {s.n_contaminated} & "
            f"{s.n_contaminated_admitted} & {pct(s.operator_accuracy)} & "
            f"{s.median_delta_s:.2f} & {s.mean_vigilance:.3f} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        f"Pooled & {pooled['n_trials']} & {pooled['n_admitted']} & "
        f"{pooled['n_contaminated']} & {pooled['n_contaminated_admitted']} & "
        f"{pct(pooled['operator_accuracy'])} & {pooled['median_delta_s']:.2f} & "
        f"{pooled['mean_vigilance']:.3f} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def make_figure(
    rows: list[dict[str, Any]],
    params: AdmissionParams,
    out_stub: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return []

    from experiments.rail_core import sigmoid

    admitted = [r for r in rows if r["admitted"]]
    withheld = [r for r in rows if not r["admitted"]]
    contaminated = [r for r in rows if r["contaminated"]]
    mean_beta = mean(float(r["beta"]) for r in rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"width_ratios": [2.0, 1.0]})

    ax = axes[0]
    grid = [i * 0.02 for i in range(0, 601)]
    # Reference curve at the pooled mean complexity bonus beta-bar.
    curve = [
        sigmoid(params.k * (params.w_delta * d - (params.tau_min + mean_beta)))
        * sigmoid(params.k * ((params.tau_max + mean_beta) - params.w_delta * d))
        for d in grid
    ]
    ax.plot(
        grid, curve, color="#444444", lw=1.2, label=rf"$V(\Delta)$ at $\bar\beta$={mean_beta:.2f}"
    )
    ax.axhline(params.theta, color="#444444", lw=0.8, ls="--")
    ax.annotate(
        rf"$\vartheta$ = {params.theta:g}",
        xy=(0.15, params.theta),
        xytext=(0.15, params.theta + 0.04),
        fontsize=8,
        color="#444444",
    )
    ax.scatter(
        [r["delta_t_s"] for r in admitted],
        [r["V"] for r in admitted],
        s=22,
        facecolor="#0072B2",
        edgecolor="none",
        alpha=0.85,
        label=f"admitted (n={len(admitted)})",
    )
    ax.scatter(
        [r["delta_t_s"] for r in withheld],
        [r["V"] for r in withheld],
        s=22,
        facecolor="none",
        edgecolor="#7F7F7F",
        linewidth=1.0,
        label=f"withheld (n={len(withheld)})",
    )
    ax.scatter(
        [r["delta_t_s"] for r in contaminated],
        [r["V"] for r in contaminated],
        marker="x",
        s=55,
        color="#D55E00",
        linewidth=1.6,
        label=f"contaminated (n={len(contaminated)})",
    )
    ax.set_xlabel(r"anchored deliberation $\Delta$ (s)")
    ax.set_ylabel(r"vigilance $V$")
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    ax.set_title("(a) Per-trial gate operating points", fontsize=9)

    ax = axes[1]
    clean_v = [r["V"] for r in rows if not r["contaminated"]]
    cont_v = [r["V"] for r in rows if r["contaminated"]]
    import random

    rng = random.Random(MC_SEED)
    ax.scatter(
        [0 + rng.uniform(-0.12, 0.12) for _ in clean_v],
        clean_v,
        s=16,
        facecolor="#0072B2",
        edgecolor="none",
        alpha=0.55,
    )
    ax.scatter(
        [1 + rng.uniform(-0.06, 0.06) for _ in cont_v],
        cont_v,
        marker="x",
        s=55,
        color="#D55E00",
        linewidth=1.6,
    )
    for pos, vals in ((0, clean_v), (1, cont_v)):
        if vals:
            med = median(vals)
            ax.hlines(med, pos - 0.2, pos + 0.2, color="#222222", lw=1.4)
    ax.axhline(params.theta, color="#444444", lw=0.8, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["clean", "contam."])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(r"vigilance $V$")
    ax.set_title("(b) $V$ by contamination status", fontsize=9)

    fig.tight_layout()
    written = []
    for suffix, dpi in ((".pdf", None), (".png", 600)):
        target = out_stub.with_suffix(suffix)
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(str(target))
    plt.close(fig)
    return written


def analyse(inputs: list[Path], output_dir: Path, tolerance: float) -> dict[str, Any]:
    sessions = load_sessions(inputs)
    summaries = [audit_session(session, tolerance) for session in sessions]
    rows = pooled_records(sessions)

    conditions = sorted({s.condition for s in summaries})
    param_sets = {json.dumps(s["rail_defaults"], sort_keys=True) for s in sessions}
    if len(param_sets) != 1:
        raise SystemExit("sessions were recorded with different admission parameters")
    params = params_from_session(sessions[0]["rail_defaults"])

    contract = contamination_contract(
        feedback_is_correct=[not bool(r["contaminated"]) for r in rows],
        admitted=[bool(r["admitted"]) for r in rows],
    )
    vig_test = permutation_test(
        [float(r["V"]) for r in rows],
        [bool(r["contaminated"]) for r in rows],
    )

    admitted = [r for r in rows if r["admitted"]]
    withheld = [r for r in rows if not r["admitted"]]
    slow_withheld = sum(
        1 for r in withheld if float(r["delta_t_s"]) > params.tau_max + float(r["beta"])
    )
    pooled = {
        "n_participants": len(summaries),
        "n_trials": len(rows),
        "conditions": conditions,
        "n_admitted": len(admitted),
        "n_contaminated": sum(int(r["contaminated"]) for r in rows),
        "n_contaminated_admitted": sum(int(r["contaminated"]) for r in admitted),
        "operator_accuracy": mean(1.0 if r["operator_label"] == r["truth"] else 0.0 for r in rows),
        "operator_accuracy_admitted": mean(
            1.0 if r["operator_label"] == r["truth"] else 0.0 for r in admitted
        )
        if admitted
        else None,
        "operator_accuracy_withheld": mean(
            1.0 if r["operator_label"] == r["truth"] else 0.0 for r in withheld
        )
        if withheld
        else None,
        "model_accuracy": mean(1.0 if r["model_flag"] == r["truth"] else 0.0 for r in rows),
        "median_delta_s": median(float(r["delta_t_s"]) for r in rows),
        "mean_focus_s": mean(float(r["focus_s"]) for r in rows),
        "mean_vigilance": mean(float(r["V"]) for r in rows),
        "trials_with_interruptions": sum(1 for r in rows if int(r["interruptions"]) > 0),
        "max_queue_depth": max(int(r["queue_depth"]) for r in rows),
        "withheld_slow_side": slow_withheld,
        "v_mismatches": sum(s.v_mismatches for s in summaries),
        "admit_mismatches": sum(s.admit_mismatches for s in summaries),
        "max_abs_v_deviation": max(s.max_abs_v_deviation for s in summaries),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_trials_csv(rows, output_dir / "human_pilot_trials.csv")
    write_participants_csv(summaries, output_dir / "human_pilot_participants.csv")
    (output_dir / "table_human_pilot.tex").write_text(
        latex_table(summaries, pooled), encoding="utf-8"
    )
    figure_paths = make_figure(rows, params, output_dir / "fig_human_pilot")

    summary = {
        "schema": "RAIL.human_pilot_summary.v1",
        "sources": [s["_source"] for s in sessions],
        "admission_params": asdict(params),
        "participants": [asdict(s) for s in summaries],
        "pooled": pooled,
        "contamination_contract": contract,
        "vigilance_separation": vig_test,
        "figures": figure_paths,
    }
    (output_dir / "human_pilot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse rail-human-telemetry-v1 session exports (real-operator pilot)."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Telemetry JSON files or directories.")
    parser.add_argument("--output-dir", type=Path, default=Path("publication_outputs/human_study"))
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Max |recorded V - recomputed V| before a trial counts as a mismatch.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = analyse(args.inputs, args.output_dir, args.tolerance)
    pooled = summary["pooled"]
    contract = summary["contamination_contract"]
    print(
        f"{pooled['n_participants']} participants, {pooled['n_trials']} trials "
        f"({', '.join(pooled['conditions'])}); "
        f"score audit: {pooled['v_mismatches']} V mismatches, "
        f"{pooled['admit_mismatches']} admission mismatches "
        f"(max |dev| {pooled['max_abs_v_deviation']:.2e})"
    )
    print(
        f"base contamination {contract['base_contamination_rate']:.3f} -> "
        f"admitted-stream {contract['admitted_contamination_rate']:.3f} "
        f"(bound {contract['contamination_bound']:.3f}); "
        f"alpha {contract['false_admission_rate']:.3f}, "
        f"rho {contract['clean_admission_rate']:.3f}, "
        f"AE {contract['admission_efficiency']:.3f}"
    )
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
