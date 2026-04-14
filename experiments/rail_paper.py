
"""
rail_four_datasets_three_strong_baselines_autocontained.py

Self-contained evaluation script with four built-in dataset generators and
three stronger baselines:

- confidence_gated
- loss_gated
- margin_gated

Included policies:
- static
- always
- confidence_gated
- loss_gated
- margin_gated
- rail_gated
- rail_weighted

Important:
This script is fully self-contained and does not require external dataset files.
The four datasets are benchmark-like generators meant to stress the same kinds
of conditions described in the paper:
- Synthetic: drift + workload effects
- SECOM-like: high-dimensional, missing values, class imbalance
- APS-like: large-scale, extreme imbalance, heterogeneous noise
- ATC-like: multiclass intent stream with correlated features

If you want the exact generators from your own script, paste that script and
the integration can be made exact.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

EPS = 1e-12


@dataclass
class Telemetry:
    anchor_time_s: float
    decision_time_s: float
    focus_time_s: float
    edit_count: int
    num_features_shown: int


@dataclass
class ReplayEvent:
    x: np.ndarray
    y_true: int
    y_human: int
    human_feedback_is_correct: bool
    telemetry: Telemetry


@dataclass
class PolicyConfig:
    name: str
    threshold: float = 0.60
    confidence_variant: str = "maxprob"
    loss_temperature: float = 1.0
    tau_min: float = 0.9
    tau_max: float = 6.0
    k: float = 1.4
    theta: float = 0.60
    w_delta: float = 1.0
    w_f: float = 0.02
    w_e: float = 0.12
    w_s: float = 0.03


@dataclass
class DatasetBundle:
    name: str
    X_warmup: np.ndarray
    y_warmup: np.ndarray
    validation_events: List[ReplayEvent]
    replay_events: List[ReplayEvent]
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class RunMetrics:
    dataset: str
    method: str
    run_id: int
    final_macro_f1: float
    contaminated_admissions: int
    admitted_feedback: int
    total_feedback: int
    admitted_yield: float
    ae: float


@dataclass
class SummaryRow:
    dataset: str
    method: str
    final_macro_f1_mean: float
    final_macro_f1_std: float
    contaminated_admissions_mean: float
    contaminated_admissions_std: float
    admitted_yield_mean: float
    admitted_yield_std: float
    ae_mean: float
    ae_std: float


class SklearnOnlineClassifier:
    def __init__(self, classes: Sequence[int], random_state: int = 0):
        self.classes_ = np.array(sorted(set(classes)))
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            learning_rate="optimal",
            random_state=random_state,
        )
        self._is_initialized = False

    def clone(self) -> "SklearnOnlineClassifier":
        return copy.deepcopy(self)

    def fit_initial(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.partial_fit(np.asarray(X), np.asarray(y), classes=self.classes_)
        self._is_initialized = True

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(np.asarray(x).reshape(1, -1))[0]
        return np.clip(p, EPS, 1.0)

    def update(self, x: np.ndarray, y: int, sample_weight: float = 1.0) -> None:
        self.model.partial_fit(
            np.asarray(x).reshape(1, -1),
            np.asarray([y]),
            sample_weight=np.asarray([float(sample_weight)]),
        )

    def evaluate_macro_f1(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        preds = self.model.predict(np.asarray(X_test))
        return float(f1_score(np.asarray(y_test), preds, average="macro"))


def normalized_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, EPS, 1.0)
    probs = probs / probs.sum()
    k = len(probs)
    ent = -np.sum(probs * np.log(probs))
    return float(ent / max(np.log(k), EPS))


def score_confidence(probs: np.ndarray, variant: str = "maxprob") -> float:
    if variant == "maxprob":
        return float(np.max(probs))
    if variant == "entropy":
        return float(1.0 - normalized_entropy(probs))
    raise ValueError(f"Unknown confidence variant: {variant}")


def score_loss_gated(probs: np.ndarray, y_human: int, temperature: float = 1.0) -> float:
    p = float(np.clip(np.asarray(probs)[int(y_human)], EPS, 1.0))
    return float(p ** (1.0 / max(temperature, EPS)))


def score_margin_gated(probs: np.ndarray) -> float:
    p = np.sort(np.asarray(probs))[::-1]
    if len(p) == 1:
        return 1.0
    margin = float(p[0] - p[1])
    return float(np.clip(margin, 0.0, 1.0))


def rail_components(telemetry: Telemetry, cfg: PolicyConfig) -> Dict[str, float]:
    delta = float(telemetry.decision_time_s - telemetry.anchor_time_s)
    beta = (
        cfg.w_f * float(telemetry.num_features_shown)
        + cfg.w_e * float(telemetry.edit_count)
        + cfg.w_s * float(telemetry.focus_time_s)
    )
    s_fast = 1.0 / (1.0 + math.exp(-cfg.k * (cfg.w_delta * delta - (cfg.tau_min + beta))))
    s_slow = 1.0 / (1.0 + math.exp(-cfg.k * ((cfg.tau_max + beta) - cfg.w_delta * delta)))
    return {
        "score": float(s_fast * s_slow),
        "delta": delta,
        "beta": beta,
        "s_fast": s_fast,
        "s_slow": s_slow,
    }


def compute_policy_score(policy: PolicyConfig, probs: np.ndarray, y_human: int, telemetry: Telemetry) -> float:
    name = policy.name.lower()
    if name == "confidence_gated":
        return score_confidence(probs, variant=policy.confidence_variant)
    if name == "loss_gated":
        return score_loss_gated(probs, y_human=y_human, temperature=policy.loss_temperature)
    if name == "margin_gated":
        return score_margin_gated(probs)
    if name in {"rail_gated", "rail_weighted"}:
        return rail_components(telemetry, policy)["score"]
    if name == "always":
        return 1.0
    if name == "static":
        return 0.0
    raise ValueError(f"Unknown policy: {policy.name}")


def policy_admits(policy: PolicyConfig, score: float) -> bool:
    name = policy.name.lower()
    if name == "static":
        return False
    if name == "always":
        return True
    if name in {"confidence_gated", "loss_gated", "margin_gated"}:
        return score >= policy.threshold
    if name == "rail_gated":
        return score >= policy.theta
    if name == "rail_weighted":
        return True
    raise ValueError(f"Unknown policy: {policy.name}")


def policy_weight(policy: PolicyConfig, score: float) -> float:
    name = policy.name.lower()
    if name == "static":
        return 0.0
    if name == "always":
        return 1.0
    if name in {"confidence_gated", "loss_gated", "margin_gated", "rail_gated"}:
        return 1.0 if policy_admits(policy, score) else 0.0
    if name == "rail_weighted":
        return float(np.clip(score, 0.0, 1.0))
    raise ValueError(f"Unknown policy: {policy.name}")


def scores_for_events(model: SklearnOnlineClassifier, events: Sequence[ReplayEvent], policy: PolicyConfig) -> np.ndarray:
    scores = []
    for ev in events:
        probs = model.predict_proba(ev.x)
        scores.append(compute_policy_score(policy, probs, ev.y_human, ev.telemetry))
    return np.asarray(scores, dtype=float)


def threshold_for_target_yield(scores: np.ndarray, target_yield: float) -> float:
    scores = np.asarray(scores, dtype=float)
    target_yield = float(np.clip(target_yield, 0.0, 1.0))
    if target_yield <= 0.0:
        return float(np.max(scores) + 1e-9)
    if target_yield >= 1.0:
        return float(np.min(scores) - 1e-9)
    sorted_scores = np.sort(scores)[::-1]
    idx = int(math.ceil(target_yield * len(sorted_scores))) - 1
    idx = max(0, min(idx, len(sorted_scores) - 1))
    return float(sorted_scores[idx])


def calibrate_gated_baselines_to_rail_yield(
    base_model: SklearnOnlineClassifier,
    validation_events: Sequence[ReplayEvent],
    policies: Sequence[PolicyConfig],
) -> List[PolicyConfig]:
    pols = [copy.deepcopy(p) for p in policies]
    by_name = {p.name.lower(): p for p in pols}
    rail_scores = scores_for_events(base_model, validation_events, by_name["rail_gated"])
    rail_target_yield = float(np.mean(rail_scores >= by_name["rail_gated"].theta))
    for baseline_name in ["confidence_gated", "loss_gated", "margin_gated"]:
        sc = scores_for_events(base_model, validation_events, by_name[baseline_name])
        by_name[baseline_name].threshold = threshold_for_target_yield(sc, rail_target_yield)
    return [by_name[p.name.lower()] for p in pols]


def replay_once(
    dataset_name: str,
    base_model: SklearnOnlineClassifier,
    events: Sequence[ReplayEvent],
    X_test: np.ndarray,
    y_test: np.ndarray,
    policy: PolicyConfig,
    run_id: int,
    always_reference_counts: Tuple[int, int] | None = None,
) -> RunMetrics:
    model = base_model.clone()
    contaminated_admissions = 0
    admitted_feedback = 0
    total_feedback = 0

    for ev in events:
        probs = model.predict_proba(ev.x)
        score = compute_policy_score(policy, probs, ev.y_human, ev.telemetry)
        admit = policy_admits(policy, score)
        weight = policy_weight(policy, score)
        total_feedback += 1
        if admit:
            admitted_feedback += 1
            if not ev.human_feedback_is_correct:
                contaminated_admissions += 1
        if weight > 0.0:
            model.update(ev.x, ev.y_human, sample_weight=weight)

    final_macro_f1 = model.evaluate_macro_f1(X_test, y_test)
    admitted_yield = admitted_feedback / max(total_feedback, 1)

    if always_reference_counts is None:
        c_always, y_always = contaminated_admissions, admitted_feedback
    else:
        c_always, y_always = always_reference_counts

    if y_always == admitted_feedback:
        ae = 0.0
    else:
        ae = float((c_always - contaminated_admissions) / (y_always - admitted_feedback + EPS))

    return RunMetrics(
        dataset=dataset_name,
        method=policy.name,
        run_id=run_id,
        final_macro_f1=final_macro_f1,
        contaminated_admissions=contaminated_admissions,
        admitted_feedback=admitted_feedback,
        total_feedback=total_feedback,
        admitted_yield=admitted_yield,
        ae=ae,
    )


def run_all_policies_once(
    dataset_name: str,
    base_model: SklearnOnlineClassifier,
    events: Sequence[ReplayEvent],
    X_test: np.ndarray,
    y_test: np.ndarray,
    policies: Sequence[PolicyConfig],
    run_id: int,
) -> List[RunMetrics]:
    by_name = {p.name.lower(): p for p in policies}
    always_metrics = replay_once(
        dataset_name, base_model, events, X_test, y_test, by_name["always"], run_id=run_id
    )
    ref = (always_metrics.contaminated_admissions, always_metrics.admitted_feedback)
    out = [always_metrics]
    for p in policies:
        if p.name.lower() == "always":
            continue
        out.append(
            replay_once(
                dataset_name, base_model, events, X_test, y_test, p, run_id=run_id,
                always_reference_counts=ref
            )
        )
    order = [p.name for p in policies]
    return sorted(out, key=lambda r: order.index(r.method))


def summarize_runs(rows: Sequence[RunMetrics]) -> List[SummaryRow]:
    keys = sorted(set((r.dataset, r.method) for r in rows))
    out = []
    for dataset, method in keys:
        subset = [r for r in rows if r.dataset == dataset and r.method == method]
        f1 = np.array([r.final_macro_f1 for r in subset], dtype=float)
        cont = np.array([r.contaminated_admissions for r in subset], dtype=float)
        yld = np.array([r.admitted_yield for r in subset], dtype=float)
        ae = np.array([r.ae for r in subset], dtype=float)
        out.append(
            SummaryRow(
                dataset=dataset,
                method=method,
                final_macro_f1_mean=float(np.mean(f1)),
                final_macro_f1_std=float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
                contaminated_admissions_mean=float(np.mean(cont)),
                contaminated_admissions_std=float(np.std(cont, ddof=1)) if len(cont) > 1 else 0.0,
                admitted_yield_mean=float(np.mean(yld)),
                admitted_yield_std=float(np.std(yld, ddof=1)) if len(yld) > 1 else 0.0,
                ae_mean=float(np.mean(ae)),
                ae_std=float(np.std(ae, ddof=1)) if len(ae) > 1 else 0.0,
            )
        )
    return out


def _fmt(mean: float, std: float, decimals: int = 3) -> str:
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def latex_main_table(summary_rows: Sequence[SummaryRow], datasets: Sequence[str], methods: Sequence[str]) -> str:
    rows_by_key = {(r.dataset, r.method): r for r in summary_rows}
    best_per_dataset = {}
    for ds in datasets:
        vals = [rows_by_key[(ds, m)].final_macro_f1_mean for m in methods if (ds, m) in rows_by_key]
        best_per_dataset[ds] = max(vals)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Final Macro-F1 ($\\uparrow$) over 20 runs. Stronger gated baselines were calibrated to match the validation-window admitted yield of RAIL-Gated.}")
    lines.append("\\label{tab:main_f1}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Method & Synthetic & SECOM-like & APS-like & ATC-like \\\\")
    lines.append("\\midrule")
    for m in methods:
        cells = [m.replace("_", "-")]
        for ds in datasets:
            row = rows_by_key[(ds, m)]
            txt = _fmt(row.final_macro_f1_mean, row.final_macro_f1_std)
            if abs(row.final_macro_f1_mean - best_per_dataset[ds]) < 1e-12:
                txt = "\\textbf{" + txt + "}"
            cells.append(txt)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def latex_tradeoff_table(summary_rows: Sequence[SummaryRow], datasets: Sequence[str], methods: Sequence[str]) -> str:
    rows_by_key = {(r.dataset, r.method): r for r in summary_rows}
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Admission trade-off over 20 runs. Lower contaminated admissions is better; higher yield and AE are better.}")
    lines.append("\\label{tab:tradeoff}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Method & Contaminated admissions ($\\downarrow$) & Yield ($\\uparrow$) & AE ($\\uparrow$) \\\\")
    lines.append("\\midrule")
    for i, ds in enumerate(datasets):
        subset = [rows_by_key[(ds, m)] for m in methods]
        best_cont = min(r.contaminated_admissions_mean for r in subset)
        best_ae = max(r.ae_mean for r in subset)
        for j, m in enumerate(methods):
            row = rows_by_key[(ds, m)]
            ds_cell = f"\\multirow{{{len(methods)}}}{{*}}{{{ds}}}" if j == 0 else ""
            cont = _fmt(row.contaminated_admissions_mean, row.contaminated_admissions_std, 2)
            yld = _fmt(row.admitted_yield_mean, row.admitted_yield_std, 3)
            ae = _fmt(row.ae_mean, row.ae_std, 3)
            if abs(row.contaminated_admissions_mean - best_cont) < 1e-12:
                cont = "\\textbf{" + cont + "}"
            if abs(row.ae_mean - best_ae) < 1e-12:
                ae = "\\textbf{" + ae + "}"
            lines.append(f"{ds_cell} & {m.replace('_', '-')} & {cont} & {yld} & {ae} \\\\")
        if i < len(datasets) - 1:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def make_default_policies() -> List[PolicyConfig]:
    return [
        PolicyConfig(name="static"),
        PolicyConfig(name="always"),
        PolicyConfig(name="confidence_gated", threshold=0.60, confidence_variant="maxprob"),
        PolicyConfig(name="loss_gated", threshold=0.60, loss_temperature=1.0),
        PolicyConfig(name="margin_gated", threshold=0.20),
        PolicyConfig(name="rail_gated", theta=0.60, tau_min=0.9, tau_max=6.0, k=1.4, w_delta=1.0, w_f=0.02, w_e=0.12, w_s=0.03),
        PolicyConfig(name="rail_weighted", theta=0.60, tau_min=0.9, tau_max=6.0, k=1.4, w_delta=1.0, w_f=0.02, w_e=0.12, w_s=0.03),
    ]


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def make_multiclass_data(
    rng: np.random.Generator,
    n: int,
    n_features: int,
    n_classes: int,
    drift: float,
    class_bias: np.ndarray,
    informative_scale: float = 1.0,
    noise_scale: float = 1.0,
    missing_prob: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    W = rng.normal(0.0, informative_scale, size=(n_features, n_classes))
    X = rng.normal(0.0, noise_scale, size=(n, n_features))
    X[:, : min(5, n_features)] += drift
    logits = X @ W + class_bias
    P = softmax_rows(logits)
    y = np.array([rng.choice(n_classes, p=P[i]) for i in range(n)], dtype=int)

    if missing_prob > 0.0:
        mask = rng.random(X.shape) < missing_prob
        X = X.copy()
        X[mask] = np.nan

    return X, y


def make_binary_data(
    rng: np.random.Generator,
    n: int,
    n_features: int,
    drift: float,
    intercept: float,
    informative_scale: float = 1.0,
    noise_scale: float = 1.0,
    missing_prob: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    w = rng.normal(0.0, informative_scale, size=(n_features,))
    X = rng.normal(0.0, noise_scale, size=(n, n_features))
    X[:, : min(6, n_features)] += drift
    logits = X @ w + intercept
    p1 = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p1).astype(int)
    if missing_prob > 0.0:
        mask = rng.random(X.shape) < missing_prob
        X = X.copy()
        X[mask] = np.nan
    return X, y


def preprocess_splits(X_warmup, X_val, X_rep, X_test):
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    X_warmup = scl.fit_transform(imp.fit_transform(X_warmup))
    X_val = scl.transform(imp.transform(X_val))
    X_rep = scl.transform(imp.transform(X_rep))
    X_test = scl.transform(imp.transform(X_test))
    return X_warmup, X_val, X_rep, X_test


def simulate_events(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
    overload_prob: float,
    correct_prob_normal: float,
    correct_prob_overload: float,
    fast_band: Tuple[float, float],
    good_band: Tuple[float, float],
    slow_band: Tuple[float, float],
    focus_normal: Tuple[float, float],
    focus_overload: Tuple[float, float],
    edits_normal: Tuple[int, int],
    edits_overload: Tuple[int, int],
) -> List[ReplayEvent]:
    events = []
    n_features = X.shape[1]
    for i in range(len(X)):
        y_true = int(y[i])
        overload = rng.random() < overload_prob
        if overload:
            is_correct = bool(rng.random() < correct_prob_overload)
            anchor = float(rng.uniform(0.0, 1.0))
            if rng.random() < 0.5:
                delta = float(rng.uniform(*fast_band))
            else:
                delta = float(rng.uniform(*slow_band))
            focus = float(rng.uniform(*focus_overload))
            edits = int(rng.integers(edits_overload[0], edits_overload[1] + 1))
        else:
            is_correct = bool(rng.random() < correct_prob_normal)
            anchor = float(rng.uniform(0.0, 1.0))
            delta = float(rng.uniform(*good_band))
            focus = float(rng.uniform(*focus_normal))
            edits = int(rng.integers(edits_normal[0], edits_normal[1] + 1))

        if is_correct:
            y_human = y_true
        else:
            wrong = [c for c in range(n_classes) if c != y_true]
            y_human = int(rng.choice(wrong))

        telemetry = Telemetry(
            anchor_time_s=anchor,
            decision_time_s=anchor + delta,
            focus_time_s=focus,
            edit_count=edits,
            num_features_shown=n_features,
        )
        events.append(
            ReplayEvent(
                x=X[i],
                y_true=y_true,
                y_human=y_human,
                human_feedback_is_correct=is_correct,
                telemetry=telemetry,
            )
        )
    return events


def make_synthetic_dataset(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_features = 12
    n_classes = 4
    bias = np.array([0.1, -0.1, 0.0, 0.0])
    X_warmup, y_warmup = make_multiclass_data(rng, 500, n_features, n_classes, 0.0, bias, 0.9, 1.0)
    X_val, y_val = make_multiclass_data(rng, 250, n_features, n_classes, 0.25, bias, 0.9, 1.0)
    X_rep, y_rep = make_multiclass_data(rng, 3200, n_features, n_classes, 0.55, bias, 0.9, 1.0)
    X_test, y_test = make_multiclass_data(rng, 700, n_features, n_classes, 0.55, bias, 0.9, 1.0)
    X_warmup, X_val, X_rep, X_test = preprocess_splits(X_warmup, X_val, X_rep, X_test)
    return DatasetBundle(
        name="Synthetic",
        X_warmup=X_warmup,
        y_warmup=y_warmup,
        validation_events=simulate_events(
            X_val, y_val, n_classes, rng,
            overload_prob=0.35,
            correct_prob_normal=0.90,
            correct_prob_overload=0.68,
            fast_band=(0.1, 0.8),
            good_band=(1.2, 5.5),
            slow_band=(7.0, 12.0),
            focus_normal=(1.5, 4.0),
            focus_overload=(0.1, 1.3),
            edits_normal=(1, 3),
            edits_overload=(0, 1),
        ),
        replay_events=simulate_events(
            X_rep, y_rep, n_classes, rng,
            overload_prob=0.38,
            correct_prob_normal=0.89,
            correct_prob_overload=0.66,
            fast_band=(0.1, 0.8),
            good_band=(1.2, 5.5),
            slow_band=(7.0, 12.0),
            focus_normal=(1.5, 4.0),
            focus_overload=(0.1, 1.3),
            edits_normal=(1, 3),
            edits_overload=(0, 1),
        ),
        X_test=X_test,
        y_test=y_test,
    )


def make_secom_like_dataset(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 100)
    n_features = 590
    n_classes = 2
    X_warmup, y_warmup = make_binary_data(rng, 700, n_features, 0.0, intercept=-1.3, informative_scale=0.18, noise_scale=1.0, missing_prob=0.12)
    X_val, y_val = make_binary_data(rng, 260, n_features, 0.15, intercept=-1.3, informative_scale=0.18, noise_scale=1.0, missing_prob=0.12)
    X_rep, y_rep = make_binary_data(rng, 1000, n_features, 0.25, intercept=-1.25, informative_scale=0.18, noise_scale=1.0, missing_prob=0.12)
    X_test, y_test = make_binary_data(rng, 400, n_features, 0.25, intercept=-1.25, informative_scale=0.18, noise_scale=1.0, missing_prob=0.12)
    X_warmup, X_val, X_rep, X_test = preprocess_splits(X_warmup, X_val, X_rep, X_test)
    return DatasetBundle(
        name="SECOM-like",
        X_warmup=X_warmup,
        y_warmup=y_warmup,
        validation_events=simulate_events(
            X_val, y_val, n_classes, rng,
            overload_prob=0.30,
            correct_prob_normal=0.91,
            correct_prob_overload=0.70,
            fast_band=(0.15, 0.9),
            good_band=(1.0, 5.8),
            slow_band=(6.8, 11.0),
            focus_normal=(1.8, 4.3),
            focus_overload=(0.15, 1.4),
            edits_normal=(1, 3),
            edits_overload=(0, 1),
        ),
        replay_events=simulate_events(
            X_rep, y_rep, n_classes, rng,
            overload_prob=0.32,
            correct_prob_normal=0.90,
            correct_prob_overload=0.69,
            fast_band=(0.15, 0.9),
            good_band=(1.0, 5.8),
            slow_band=(6.8, 11.0),
            focus_normal=(1.8, 4.3),
            focus_overload=(0.15, 1.4),
            edits_normal=(1, 3),
            edits_overload=(0, 1),
        ),
        X_test=X_test,
        y_test=y_test,
    )


def make_aps_like_dataset(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 200)
    n_features = 170
    n_classes = 2
    X_warmup, y_warmup = make_binary_data(rng, 2000, n_features, 0.0, intercept=-2.4, informative_scale=0.22, noise_scale=1.15, missing_prob=0.04)
    X_val, y_val = make_binary_data(rng, 600, n_features, 0.18, intercept=-2.35, informative_scale=0.22, noise_scale=1.15, missing_prob=0.04)
    X_rep, y_rep = make_binary_data(rng, 50000, n_features, 0.30, intercept=-2.30, informative_scale=0.22, noise_scale=1.15, missing_prob=0.04)
    X_test, y_test = make_binary_data(rng, 2000, n_features, 0.30, intercept=-2.30, informative_scale=0.22, noise_scale=1.15, missing_prob=0.04)
    X_warmup, X_val, X_rep, X_test = preprocess_splits(X_warmup, X_val, X_rep, X_test)
    return DatasetBundle(
        name="APS-like",
        X_warmup=X_warmup,
        y_warmup=y_warmup,
        validation_events=simulate_events(
            X_val, y_val, n_classes, rng,
            overload_prob=0.34,
            correct_prob_normal=0.92,
            correct_prob_overload=0.72,
            fast_band=(0.12, 0.9),
            good_band=(1.1, 5.8),
            slow_band=(6.5, 11.5),
            focus_normal=(1.7, 4.0),
            focus_overload=(0.15, 1.2),
            edits_normal=(1, 3),
            edits_overload=(0, 1),
        ),
        replay_events=simulate_events(
            X_rep, y_rep, n_classes, rng,
            overload_prob=0.36,
            correct_prob_normal=0.91,
            correct_prob_overload=0.70,
            fast_band=(0.12, 0.9),
            good_band=(1.1, 5.8),
            slow_band=(6.5, 11.5),
            focus_normal=(1.7, 4.0),
            focus_overload=(0.15, 1.2),
            edits_normal=(1, 3),
            edits_overload=(0, 1),
        ),
        X_test=X_test,
        y_test=y_test,
    )


def make_atc_like_dataset(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed + 300)
    n_features = 48
    n_classes = 4
    bias = np.array([0.2, 0.1, -0.1, -0.2])
    X_warmup, y_warmup = make_multiclass_data(rng, 700, n_features, n_classes, 0.0, bias, 0.45, 0.9)
    X_val, y_val = make_multiclass_data(rng, 260, n_features, n_classes, 0.10, bias, 0.45, 0.9)
    X_rep, y_rep = make_multiclass_data(rng, 1400, n_features, n_classes, 0.18, bias, 0.45, 0.9)
    X_test, y_test = make_multiclass_data(rng, 600, n_features, n_classes, 0.18, bias, 0.45, 0.9)
    X_warmup, X_val, X_rep, X_test = preprocess_splits(X_warmup, X_val, X_rep, X_test)
    return DatasetBundle(
        name="ATC-like",
        X_warmup=X_warmup,
        y_warmup=y_warmup,
        validation_events=simulate_events(
            X_val, y_val, n_classes, rng,
            overload_prob=0.28,
            correct_prob_normal=0.88,
            correct_prob_overload=0.74,
            fast_band=(0.10, 0.85),
            good_band=(1.0, 5.2),
            slow_band=(6.0, 10.0),
            focus_normal=(1.6, 4.5),
            focus_overload=(0.2, 1.4),
            edits_normal=(1, 4),
            edits_overload=(0, 2),
        ),
        replay_events=simulate_events(
            X_rep, y_rep, n_classes, rng,
            overload_prob=0.30,
            correct_prob_normal=0.87,
            correct_prob_overload=0.75,
            fast_band=(0.10, 0.85),
            good_band=(1.0, 5.2),
            slow_band=(6.0, 10.0),
            focus_normal=(1.6, 4.5),
            focus_overload=(0.2, 1.4),
            edits_normal=(1, 4),
            edits_overload=(0, 2),
        ),
        X_test=X_test,
        y_test=y_test,
    )


def build_all_datasets(seed: int) -> List[DatasetBundle]:
    return [
        make_synthetic_dataset(seed),
        make_secom_like_dataset(seed),
        make_aps_like_dataset(seed),
        make_atc_like_dataset(seed),
    ]


def run_benchmark(runs: int = 20, output_dir: str = "rail_autocontained_results", seed: int = 7) -> Dict[str, object]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_run_rows: List[RunMetrics] = []
    policies = make_default_policies()

    for run_id in range(runs):
        datasets = build_all_datasets(seed + run_id * 17)
        for ds in datasets:
            all_classes = np.unique(np.concatenate([ds.y_warmup, ds.y_test]))
            base_model = SklearnOnlineClassifier(classes=all_classes, random_state=seed + run_id)
            base_model.fit_initial(ds.X_warmup, ds.y_warmup)

            calibrated = calibrate_gated_baselines_to_rail_yield(
                base_model=base_model,
                validation_events=ds.validation_events,
                policies=policies,
            )

            rows = run_all_policies_once(
                dataset_name=ds.name,
                base_model=base_model,
                events=ds.replay_events,
                X_test=ds.X_test,
                y_test=ds.y_test,
                policies=calibrated,
                run_id=run_id,
            )
            all_run_rows.extend(rows)

    summary = summarize_runs(all_run_rows)
    methods = [p.name for p in policies]
    datasets = ["Synthetic", "SECOM-like", "APS-like", "ATC-like"]

    df_runs = pd.DataFrame([asdict(r) for r in all_run_rows])
    df_summary = pd.DataFrame([asdict(r) for r in summary])

    df_runs.to_csv(outdir / "run_metrics.csv", index=False)
    df_summary.to_csv(outdir / "summary_metrics.csv", index=False)
    (outdir / "summary_metrics.json").write_text(json.dumps([asdict(r) for r in summary], indent=2), encoding="utf-8")

    main_tex = latex_main_table(summary, datasets=datasets, methods=methods)
    trade_tex = latex_tradeoff_table(summary, datasets=datasets, methods=methods)
    (outdir / "table_main_f1.tex").write_text(main_tex, encoding="utf-8")
    (outdir / "table_tradeoff.tex").write_text(trade_tex, encoding="utf-8")

    return {
        "output_dir": str(outdir),
        "summary": [asdict(r) for r in summary],
        "latex_main_table": main_tex,
        "latex_tradeoff_table": trade_tex,
    }


if __name__ == "__main__":
    result = run_benchmark(runs=20, output_dir="rail_autocontained_results", seed=7)
    print("Results written to:", result["output_dir"])
    print("\nMain table:\n")
    print(result["latex_main_table"])
    print("\nTrade-off table:\n")
    print(result["latex_tradeoff_table"])
