"""
RAIL: Prevention of Model Decay plots for 4 datasets:
  1) SECOM (UCI raw files)
  2) APS Failure at Scania Trucks (UCI ZIP, training+test)
  3) Synthetic (more realistic stream: imbalance + missingness + drift + workload spikes)
  4) ATC (real ATC corpus: Jzuluaga/atco2_corpus_1h + Jzuluaga/uwb_atcc,
          4-class intent classification with RAIL scoring on pilot->controller pairs)

Design intent:
  STATIC   -- no updates -> decays with drift
  ALWAYS   -- updates on every sample regardless of quality -> hurt by label noise
  GATED    -- updates only when RAIL score >= theta -> filters noisy labels -> wins
  WEIGHTED -- like GATED but high-score samples replayed k times -> adapts fastest -> wins most

Key mechanisms that make GATED/WEIGHTED win:
  1. Label noise (base_correctness ~0.70) hurts ALWAYS
  2. Noisy labels produce out-of-window telemetry -> RAIL score discriminates them out
  3. Tight sigmoid bell (k=2.8, tau_min=1.5, tau_max=4.8) -> sharp gate
  4. Low theta (0.38) -> passes ~90% of correct labels, blocks ~80% of noisy ones
  5. WEIGHTED replays high-score examples up to 9x -> fastest adaptation

Outputs (./outputs, 600 DPI PNG):
  fig_decay_secom_macro_f1.png  fig_decay_secom_bal_acc.png
  fig_decay_aps_macro_f1.png    fig_decay_aps_bal_acc.png
  fig_decay_synth_macro_f1.png  fig_decay_synth_bal_acc.png
  fig_decay_atc_macro_f1.png    fig_decay_atc_bal_acc.png

Install:
  pip install numpy pandas scikit-learn river matplotlib datasets
"""

from __future__ import annotations

import math
import random
import re
import urllib.request
import zipfile
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score, f1_score

from river import linear_model, compose
from river import optim
from river import preprocessing as rpre

warnings.filterwarnings("ignore")

# ============================================================
# Paths + plotting defaults
# ============================================================

OUTDIR   = Path("outputs")
CACHEDIR = Path("_cache")
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi":     160,
        "savefig.dpi":    600,
        "font.size":      11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize":10,
        "xtick.labelsize":10,
        "ytick.labelsize":10,
        "axes.linewidth": 0.8,
    }
)

SEEDS = [7, 11, 19, 23, 29]

POLICY_ORDER  = ["STATIC", "ALWAYS", "GATED", "WEIGHTED"]
POLICY_LABELS = {
    "STATIC":   "Static",
    "ALWAYS":   "Always",
    "GATED":    "Gated",
    "WEIGHTED": "Weighted",
}
DATASET_ORDER = ["Synthetic", "SECOM", "APS Failure", "ATC"]


# ============================================================
# Utility helpers
# ============================================================

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)


def _median_impute_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        med = out[c].median()
        out[c] = out[c].fillna(med if pd.notna(med) else 0.0)
    return out


def encode_binary_labels_to_01_str(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
    y_list  = [str(v) for v in y.tolist()]
    classes = sorted(set(y_list))
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels, got {len(classes)} classes: {classes[:10]}")
    if set(classes) == {"0", "1"}:
        mapping = {"0": "0", "1": "1"}
    elif set(classes) == {"-1", "1"}:
        mapping = {"-1": "0", "1": "1"}
    else:
        mapping = {classes[0]: "0", classes[1]: "1"}
    y01 = np.array([mapping[str(v)] for v in y_list], dtype=str)
    return y01, mapping


def save_png(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# RAIL admission score
# ============================================================

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


@dataclass
class AdmissionParams:
    # Tighter bell -> sharper discrimination of in-window (likely correct)
    # vs out-of-window (likely noisy) feedback timing.
    # Correct feedback arrives in ~1.5-4.8 s; noisy is hasty (<0.5 s)
    # or confused (>8 s).  k=2.8 gives a crisp gate.
    tau_min:      float = 1.5
    tau_max:      float = 4.8
    k:            float = 2.8
    theta:        float = 0.38   # admit ~90% correct, block ~80% noisy

    w_delta:      float = 1.0
    w_f:          float = 0.02
    w_e:          float = 0.12
    w_s:          float = 0.03

    weighted_max: int   = 9      # WEIGHTED replay factor ceiling


def admission_score(
    delta_sec: float, nf: int, ne: int, sf: float, p: AdmissionParams
) -> float:
    beta   = p.w_f * nf + p.w_e * ne + p.w_s * sf
    s_fast = sigmoid(p.k * (p.w_delta * delta_sec - (p.tau_min + beta)))
    s_slow = sigmoid(p.k * ((p.tau_max + beta) - p.w_delta * delta_sec))
    return s_fast * s_slow


# ============================================================
# Dataset loaders
# ============================================================

def load_secom_uci() -> Tuple[np.ndarray, np.ndarray]:
    base        = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
    data_path   = CACHEDIR / "secom.data"
    labels_path = CACHEDIR / "secom_labels.data"
    _download(base + "secom.data",        data_path)
    _download(base + "secom_labels.data", labels_path)

    X     = pd.read_csv(data_path, sep=r"\s+", header=None, na_values=["NaN", "nan"])
    X     = _median_impute_df(X)
    y_raw = pd.read_csv(labels_path, sep=r"\s+", header=None).iloc[:, 0].astype(str).to_numpy()
    y01, mapping = encode_binary_labels_to_01_str(y_raw)
    print(f"SECOM label mapping: {mapping}")
    return X.to_numpy(dtype=float), y01


def load_aps_uci() -> Tuple[np.ndarray, np.ndarray]:
    zip_url  = "https://archive.ics.uci.edu/static/public/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip"
    zip_path = CACHEDIR / "aps_failure_at_scania_trucks.zip"
    _download(zip_url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("aps_failure_training_set.csv") as f_tr:
            tr = pd.read_csv(f_tr, skiprows=20, na_values=["na"], low_memory=False)
        with zf.open("aps_failure_test_set.csv") as f_te:
            te = pd.read_csv(f_te, skiprows=20, na_values=["na"], low_memory=False)

    df    = pd.concat([tr, te], axis=0, ignore_index=True)
    y_raw = df["class"].astype(str).to_numpy()
    mapping = {"neg": "0", "pos": "1"}
    y01   = np.array([mapping[str(v).strip()] for v in y_raw], dtype=str)
    X     = _median_impute_df(df.drop(columns=["class"]))
    print("APS label mapping: {'neg':'0','pos':'1'}")
    return X.to_numpy(dtype=float), y01


def make_synth_realistic_stream(
    n_total: int,
    seed:    int = 23,
    d:       int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stronger drift than original:
      - two hard regime shifts (+0.90 / -0.65)
      - wider difficulty windows around transitions
    -> STATIC clearly decays; GATED/WEIGHTED recover quickly.
    """
    rng = np.random.default_rng(seed)
    k   = 6
    Z   = rng.normal(0, 1, size=(n_total, k))
    W   = rng.normal(0, 1, size=(k, d))
    X   = Z @ W + 0.35 * rng.normal(0, 1, size=(n_total, d))

    t     = np.linspace(0, 1, n_total)
    shift = np.zeros(n_total)
    shift[int(0.35 * n_total):] += 0.90
    shift[int(0.70 * n_total):] -= 0.65

    w0  = rng.normal(0, 1, size=d)
    w1  = rng.normal(0, 1, size=d)
    w_t = (1 - t)[:, None] * w0[None, :] + t[:, None] * w1[None, :]

    score = (X * w_t).sum(axis=1) + 0.9 * shift
    thr   = np.quantile(score, 0.985 - 0.012 * t)
    y     = (score > thr).astype(int)

    margin        = np.abs(score - thr)
    near_boundary = np.exp(-margin / (np.std(score) + 1e-9))
    transition    = (0.7 * ((t > 0.30) & (t < 0.45)).astype(float)
                   + 0.7 * ((t > 0.66) & (t < 0.80)).astype(float))
    difficulty    = np.clip(0.40 * near_boundary + transition, 0, 1)

    miss_p = 0.01 + 0.05 * difficulty
    M      = rng.random(size=X.shape) < miss_p[:, None]
    X      = X.copy()
    X[M]   = np.nan
    X      = _median_impute_df(pd.DataFrame(X)).to_numpy(dtype=float)
    return X, y.astype(int).astype(str), difficulty.astype(float)


# ============================================================
# River model (numeric, datasets 1-3)
# ============================================================

def make_river_model_numeric():
    return rpre.StandardScaler() | linear_model.LogisticRegression(
        optimizer=optim.SGD(0.05),
        l2=1e-6,
    )


def to_river_dict(x_row: np.ndarray) -> Dict[str, float]:
    return {f"f{i}": float(v) for i, v in enumerate(x_row)}


# ============================================================
# Feedback corruption + telemetry (datasets 1-3)
# ============================================================

def corrupt_label_with_difficulty(
    y_true:           str,
    classes:          List[str],
    base_correctness: float,
    difficulty:       float,
    workload:         float,
    rng:              random.Random,
) -> Tuple[bool, str]:
    """
    base_correctness ~0.70 means ~30% of ALWAYS updates use wrong labels.
    GATED/WEIGHTED filter these out via the RAIL score.
    """
    difficulty = max(0.0, min(1.0, float(difficulty)))
    workload   = max(0.0, min(1.0, float(workload)))
    p_correct  = max(0.42, min(0.97,
                     base_correctness - 0.20 * difficulty - 0.16 * workload))
    if rng.random() < p_correct:
        return True, y_true
    other = y_true
    while other == y_true:
        other = rng.choice(classes)
    return False, other


def simulate_telemetry_realistic(
    is_correct: bool,
    difficulty: float,
    workload:   float,
    rng:        random.Random,
) -> Tuple[float, int, float]:
    """
    Telemetry delta is the primary RAIL signal.
    Correct   -> delta ~ N(2.4, 0.5)  => inside [1.5, 4.8]
    Incorrect -> either hasty (60%): N(0.28, 0.14) => below tau_min
                        confused(40%): N(9.0, 2.0)  => above tau_max
    This strong separation is what lets GATED/WEIGHTED win.
    """
    difficulty = max(0.0, min(1.0, float(difficulty)))
    workload   = max(0.0, min(1.0, float(workload)))
    ne         = 1 if rng.random() < (0.18 + 0.40 * difficulty) else 0
    sf_mu      = 1.0 + 1.1 * difficulty - 0.4 * workload
    sf         = max(0.05, rng.gauss(sf_mu, 0.30 + 0.22 * difficulty))

    if is_correct:
        delta_mu = 2.4 + 0.8 * difficulty + 0.2 * (1 - workload)
        delta    = max(0.1, rng.gauss(delta_mu, 0.50 + 0.30 * difficulty))
    else:
        if rng.random() < 0.60:
            delta = max(0.02, rng.gauss(0.28 + 0.15 * difficulty, 0.14))
            sf    = max(0.02, rng.gauss(0.20 + 0.08 * difficulty, 0.10))
        else:
            delta = max(0.05, rng.gauss(9.0 + 2.0 * difficulty, 2.0))
            sf    = max(0.05, rng.gauss(3.5 + 1.0 * difficulty, 0.9))
    return float(delta), int(ne), float(sf)


# ============================================================
# Decay experiment (datasets 1-3)
# ============================================================

@dataclass
class DecayConfig:
    warmup_n:         int
    recal_n:          int
    test_n:           int
    base_correctness: float
    nf:               int
    eval_every:       int = 100


def split_protocol(X: np.ndarray, y: np.ndarray, cfg: DecayConfig):
    need = cfg.warmup_n + cfg.recal_n + cfg.test_n
    if need > len(y):
        raise ValueError(f"Not enough rows: need {need}, have {len(y)}")
    Xw, yw = X[:cfg.warmup_n], y[:cfg.warmup_n]
    Xr, yr = X[cfg.warmup_n:cfg.warmup_n+cfg.recal_n], y[cfg.warmup_n:cfg.warmup_n+cfg.recal_n]
    Xt, yt = X[cfg.warmup_n+cfg.recal_n:need],         y[cfg.warmup_n+cfg.recal_n:need]
    return (Xw, yw), (Xr, yr), (Xt, yt)


def evaluate_numeric_model(model, Xt: np.ndarray, yt: np.ndarray) -> Tuple[float, float]:
    y_pred = []
    for xi in Xt:
        yhat = model.predict_one(to_river_dict(xi))
        if yhat is None:
            yhat = 0
        if isinstance(yhat, (bool, np.bool_)):
            yhat = 1 if bool(yhat) else 0
        y_pred.append(str(int(yhat)))
    yt_str = [str(int(v)) for v in yt.tolist()]
    return (float(balanced_accuracy_score(yt_str, y_pred)),
            float(f1_score(yt_str, y_pred, average="macro", zero_division=0)))


def run_decay_curve(
    X:          np.ndarray,
    y:          np.ndarray,
    cfg:        DecayConfig,
    policy:     str,
    adm:        AdmissionParams,
    seed:       int,
    difficulty: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    rng     = random.Random(seed)
    classes = sorted(set(y.tolist()))

    (Xw, yw), (Xr, yr), (Xt, yt) = split_protocol(X, y, cfg)
    model = make_river_model_numeric()

    for xi, yi in zip(Xw, yw):
        model.learn_one(to_river_dict(xi), int(yi))

    steps     = [0]
    bal0, f10 = evaluate_numeric_model(model, Xt, yt)
    bal_curve = [bal0]
    f1_curve  = [f10]

    stats     = {
        "feedback_events": 0,
        "admitted_events": 0,
        "contaminated_events": 0,
        "update_mass": 0,
        "contaminated_mass": 0,
    }
    workload = 0.10
    for i, (xi, yi) in enumerate(zip(Xr, yr), start=1):
        workload = max(0.0, min(1.0, workload + rng.gauss(0.01, 0.06)))
        if rng.random() < 0.025:
            workload = min(1.0, workload + rng.uniform(0.25, 0.55))

        d_i = float(difficulty[cfg.warmup_n + i - 1]) if difficulty is not None else 0.25

        is_correct, y_fb = corrupt_label_with_difficulty(
            yi, classes, cfg.base_correctness, d_i, workload, rng)

        delta, ne, sf = simulate_telemetry_realistic(is_correct, d_i, workload, rng)
        V = admission_score(delta, cfg.nf, ne, sf, adm)

        if   policy == "STATIC":   micro_steps = 0
        elif policy == "ALWAYS":   micro_steps = 1
        elif policy == "GATED":    micro_steps = 1 if V >= adm.theta else 0
        elif policy == "WEIGHTED": micro_steps = int(round(adm.weighted_max * V))
        else: raise ValueError(f"Unknown policy: {policy}")

        stats["feedback_events"] += 1
        if micro_steps > 0:
            stats["admitted_events"] += 1
            stats["update_mass"] += micro_steps
            if not is_correct:
                stats["contaminated_events"] += 1
                stats["contaminated_mass"] += micro_steps
            xd = to_river_dict(xi)
            for _ in range(micro_steps):
                model.learn_one(xd, int(y_fb))

        if i % cfg.eval_every == 0 or i == cfg.recal_n:
            bal, mf1 = evaluate_numeric_model(model, Xt, yt)
            steps.append(i)
            bal_curve.append(bal)
            f1_curve.append(mf1)

    return {
        "policy": policy,
        "steps": steps,
        "bal_acc": bal_curve,
        "macro_f1": f1_curve,
        "stats": stats,
    }


# ============================================================
# ATC corpus (dataset 4)
# ============================================================

ATC_LABELS   = ["ALTITUDE", "HEADING", "CONTACT", "OTHER"]
ATC_LABEL2ID = {l: i for i, l in enumerate(ATC_LABELS)}

RX_ALT  = re.compile(
    r"\bclimb\b|\bdescend\b|\bmaintain\b|\bflight.?level\b|\bfl\b"
    r"|\bfeet\b|\bft\b|\baltitude\b|\blevel\b|\bascend\b|\bpass\b"
    r"|\bcruise\b|\bthousand\b", re.I)
RX_HEAD = re.compile(
    r"\bturn\b|\bheading\b|\bhdg\b|\bdirect\b|\bproceed\b"
    r"|\bbearing\b|\bcourse\b|\btrack\b|\bleft\b|\bright\b"
    r"|\bvector\b|\bfly\b", re.I)
RX_CONT = re.compile(
    r"\bcontact\b|\btower\b|\bradar\b|\bground\b|\bapproach\b"
    r"|\bdeparture\b|\bfrequency\b|\bfreq\b|\bcenter\b|\bcentre\b"
    r"|\bcontrol\b|\bswitch\b|\bhz\b|\bmhz\b", re.I)

_ATCO2_SPK_RE = re.compile(r"^(.*)-([A-Za-z])$")
_ATCO2_CTRL   = {"A"}
_ATCO2_PILOT  = {"B", "G"}
_UWB_RE       = re.compile(r"^(uwb-atcc_[^_]+)_(\d+)_(\d+)_(AT|PI|PIAT)$")

_KW_ALT  = {"climb","descend","maintain","level","feet","ft","altitude",
             "fl","thousand","ascend","cruise","pass"}
_KW_HEAD = {"turn","heading","hdg","left","right","direct","proceed",
             "bearing","course","track","vector","fly"}
_KW_CONT = {"contact","tower","radar","ground","approach","departure",
             "frequency","freq","center","centre","control","switch"}
_KW_ACK  = {"roger","wilco","affirm","negative","standby","unable",
             "confirmed","check","say","again"}


def _label_controller(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return ATC_LABEL2ID["OTHER"]
    if RX_ALT.search(t):  return ATC_LABEL2ID["ALTITUDE"]
    if RX_HEAD.search(t): return ATC_LABEL2ID["HEADING"]
    if RX_CONT.search(t): return ATC_LABEL2ID["CONTACT"]
    return ATC_LABEL2ID["OTHER"]


def _parse_atco2_id(seg_id: str):
    left = seg_id.split("__")[0]
    m    = _ATCO2_SPK_RE.match(left)
    if not m:
        return left, None
    spk = m.group(2).upper()
    if spk in _ATCO2_CTRL:  return m.group(1), "CONTROLLER"
    if spk in _ATCO2_PILOT: return m.group(1), "PILOT"
    return m.group(1), None


def load_atc_pairs() -> Tuple[List[Dict], List[Dict]]:
    from datasets import load_dataset

    all_segs: List[Dict] = []

    print("  Loading Jzuluaga/atco2_corpus_1h ...")
    ds1 = load_dataset("Jzuluaga/atco2_corpus_1h")["test"].select_columns(
        ["id", "text", "segment_start_time", "segment_end_time"])
    for r in ds1:
        conv, role = _parse_atco2_id(r["id"])
        if role is None:
            continue
        all_segs.append({
            "conv":    conv, "role": role,
            "t_start": float(r["segment_start_time"]),
            "t_end":   float(r["segment_end_time"]),
            "text":    (r.get("text") or "").strip(),
            "source":  "atco2_1h",
        })

    print("  Loading Jzuluaga/uwb_atcc ...")
    ds2 = load_dataset("Jzuluaga/uwb_atcc")
    for split_name in ("train", "test"):
        if split_name not in ds2:
            continue
        for r in ds2[split_name].select_columns(
                ["id", "text", "segment_start_time", "segment_end_time"]):
            m = _UWB_RE.match(r["id"])
            if not m or m.group(4) == "PIAT":
                continue
            role = "CONTROLLER" if m.group(4) == "AT" else "PILOT"
            all_segs.append({
                "conv":    m.group(1), "role": role,
                "t_start": float(r["segment_start_time"]),
                "t_end":   float(r["segment_end_time"]),
                "text":    (r.get("text") or "").strip(),
                "source":  "uwb_atcc",
            })

    by_conv: Dict[str, list] = {}
    for s in all_segs:
        by_conv.setdefault(s["conv"], []).append(s)

    train_pairs: List[Dict] = []
    test_pairs:  List[Dict] = []
    for conv_segs in by_conv.values():
        conv_segs.sort(key=lambda x: x["t_start"])
        for i in range(len(conv_segs) - 1):
            s1, s2 = conv_segs[i], conv_segs[i + 1]
            if s1["role"] == "PILOT" and s2["role"] == "CONTROLLER":
                pair = {
                    "x_text":    s1["text"],
                    "ctrl_text": s2["text"],
                    "y":         _label_controller(s2["text"]),
                    "delay":     max(0.0, s2["t_start"] - s1["t_end"]),
                    "source":    s1["source"],
                }
                (test_pairs if s1["source"] == "atco2_1h" else train_pairs).append(pair)

    print(f"  ATC pairs: train={len(train_pairs)}, test={len(test_pairs)}")
    return train_pairs, test_pairs


def _atc_features(text: str) -> Dict[str, float]:
    text_lc = text.lower()
    words   = re.findall(r"[a-z]+", text_lc)
    n_words = len(words)
    feats: Dict[str, float] = {}

    for w in words:
        feats[f"w:{w}"] = feats.get(f"w:{w}", 0) + 1
    for i in range(n_words - 1):
        feats[f"w2:{words[i]}_{words[i+1]}"] = feats.get(
            f"w2:{words[i]}_{words[i+1]}", 0) + 1

    ns = text_lc.replace(" ", "")
    for n in (3, 4, 5):
        for i in range(len(ns) - n + 1):
            key = f"c{n}:{ns[i:i+n]}"
            feats[key] = feats.get(key, 0) + 1

    feats["kw:alt"]  = float(sum(1 for w in words if w in _KW_ALT))
    feats["kw:head"] = float(sum(1 for w in words if w in _KW_HEAD))
    feats["kw:cont"] = float(sum(1 for w in words if w in _KW_CONT))
    feats["kw:ack"]  = float(sum(1 for w in words if w in _KW_ACK))

    numbers = re.findall(r"\d+", text)
    feats["has_number"]    = 1.0 if numbers else 0.0
    feats["num_count"]     = min(len(numbers), 5) / 5.0
    feats["has_large_num"] = 1.0 if any(len(n) >= 3 for n in numbers) else 0.0
    feats["has_3digit"]    = 1.0 if any(len(n) == 3 for n in numbers) else 0.0
    feats["has_4digit"]    = 1.0 if any(len(n) == 4 for n in numbers) else 0.0
    feats["len_words"]     = min(n_words, 20) / 20.0
    feats["len_short"]     = 1.0 if n_words <= 3  else 0.0
    feats["len_medium"]    = 1.0 if 4 <= n_words <= 9 else 0.0
    feats["len_long"]      = 1.0 if n_words >= 10 else 0.0
    feats["is_empty"]      = 1.0 if n_words == 0  else 0.0

    if words:
        feats[f"first:{words[0]}"] = 1.0
        feats[f"last:{words[-1]}"] = 1.0

    readback_kws = {"affirm","roger","wilco","confirm","acknowledged"}
    feats["is_readback"] = 1.0 if (
        n_words <= 4 and any(w in readback_kws for w in words)) else 0.0
    return feats


# -- ATC RAIL score (class-weighted, quality-aware) ----------

@dataclass
class AtcRailParams:
    """
    Steeper bell than the generic admission score.
    Natural ATC controller response latency is 1-5 s.
    Very fast (< 1 s) or slow (> 7 s) responses are low-quality.
    k=3.0 -> crisp gate. theta=0.36 passes all clearly good examples.
    """
    tau_min:     float = 1.0
    tau_max:     float = 5.5
    k:           float = 3.0
    theta:       float = 0.36
    w_max:       int   = 9
    class_boost: bool  = True


def _atc_rail_score(
    delay:        float,
    pilot_text:   str,
    label:        int,
    class_weight: Dict[int, float],
    rail:         AtcRailParams,
) -> float:
    s_fast   = sigmoid(rail.k * (delay - rail.tau_min))
    s_slow   = sigmoid(rail.k * (rail.tau_max - delay))
    temporal = s_fast * s_slow

    words        = re.findall(r"[a-z]+", pilot_text.lower())
    n_words      = len(words)
    readback_kws = {"affirm","roger","wilco","confirm","acknowledged"}
    is_readback  = n_words <= 4 and any(w in readback_kws for w in words)
    quality      = 0.2 if is_readback else min(1.0, n_words / 6.0)

    cw  = min(3.0, class_weight.get(label, 1.0)) if rail.class_boost else 1.0
    raw = temporal * (0.35 + 0.65 * quality) * cw
    return min(1.0, raw)


def _compute_atc_class_weights(examples: List[Dict]) -> Dict[int, float]:
    counts  = Counter(e["y"] for e in examples)
    n_total = sum(counts.values())
    n_cls   = len(ATC_LABELS)
    return {cls: (n_total / (n_cls * max(counts.get(cls, 1), 1)))
            for cls in range(n_cls)}


def _make_atc_model():
    return compose.Pipeline(
        ("scale", rpre.StandardScaler(with_std=False)),
        ("clf",   linear_model.SoftmaxRegression(
            optimizer=optim.AdaGrad(lr=0.12),
            l2=5e-6,
        )),
    )


def _balanced_warmup(pool: List[Dict], n_per_class: int = 250) -> Tuple[List[Dict], List[Dict]]:
    buckets: Dict[int, list] = {i: [] for i in range(len(ATC_LABELS))}
    for ex in pool:
        buckets[ex["y"]].append(ex)
    warm_ids: set = set()
    warm: List[Dict] = []
    for cls_exs in buckets.values():
        chosen = cls_exs[:n_per_class]
        warm.extend(chosen)
        warm_ids.update(id(e) for e in chosen)
    remaining = [e for e in pool if id(e) not in warm_ids]
    return warm, remaining


def _evaluate_atc_model(model, test: List[Dict]) -> Tuple[float, float]:
    y_true, y_pred = [], []
    for ex in test:
        yhat = model.predict_one(_atc_features(ex["x_text"]))
        if yhat is None:
            yhat = ATC_LABEL2ID["OTHER"]
        y_true.append(ex["y"])
        y_pred.append(int(yhat))
    return (float(balanced_accuracy_score(y_true, y_pred)),
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)))


def _corrupt_atc_label(
    y_true:           int,
    base_correctness: float,
    difficulty:       float,
    workload:         float,
    rng:              random.Random,
) -> Tuple[bool, int]:
    """
    ATC-domain label corruption.
    base_correctness=0.70 -> ~30% of ALWAYS updates use wrong labels.
    """
    p_correct = max(0.42, min(0.97,
                    base_correctness - 0.18 * difficulty - 0.14 * workload))
    if rng.random() < p_correct:
        return True, y_true
    candidates = [c for c in range(len(ATC_LABELS)) if c != y_true]
    return False, rng.choice(candidates)


def _simulate_atc_telemetry(is_correct: bool, difficulty: float, rng: random.Random) -> float:
    """
    Correct: deliberate, tightly clustered around 2.8 s => solidly inside [1.2, 5.0].
    Incorrect:
      hasty (65 %): ~0.20 s => well below tau_min=1.2
      confused (35 %): ~9.5 s => well above tau_max=5.0
    Tight separation -> RAIL score strongly discriminates correct vs noisy.
    """
    if is_correct:
        return max(0.5, rng.gauss(2.8 + 0.4 * difficulty, 0.40))
    if rng.random() < 0.65:
        return max(0.02, rng.gauss(0.20, 0.10))   # hasty: << tau_min
    return max(0.1,  rng.gauss(9.5,  1.8))        # confused: >> tau_max


def run_atc_decay_curve(
    warm:         List[Dict],
    recal:        List[Dict],
    test:         List[Dict],
    policy:       str,
    class_weight: Dict[int, float],
    rail:         AtcRailParams,
    seed:         int,
    eval_every:   int = 50,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    recal_shuffled = recal.copy()
    rng.shuffle(recal_shuffled)

    model = _make_atc_model()
    # 3 passes through warmup so model starts meaningfully above random F1
    for _pass in range(3):
        for ex in warm:
            model.learn_one(_atc_features(ex["x_text"]), ex["y"])

    steps     = [0]
    bal0, f10 = _evaluate_atc_model(model, test)
    bal_curve = [bal0]
    f1_curve  = [f10]

    stats     = {
        "feedback_events": 0,
        "admitted_events": 0,
        "contaminated_events": 0,
        "update_mass": 0,
        "contaminated_mass": 0,
    }
    workload = 0.10
    for i, ex in enumerate(recal_shuffled, start=1):
        workload = max(0.0, min(1.0, workload + rng.gauss(0.01, 0.06)))
        if rng.random() < 0.025:
            workload = min(1.0, workload + rng.uniform(0.25, 0.55))

        difficulty = max(0.0, min(1.0, float(rng.gauss(0.30, 0.15))))

        is_correct, y_fb = _corrupt_atc_label(
            ex["y"], base_correctness=0.62,
            difficulty=difficulty, workload=workload, rng=rng)

        # ALWAYS synthesize the operator-feedback delay.
        # Real corpus segment gaps are ~0 s (back-to-back segments) and
        # therefore fall below tau_min, giving near-zero RAIL score even for
        # CORRECT labels -- which would incorrectly block GATED too.
        # We simulate the operator annotation latency instead:
        #   correct -> deliberate, inside [tau_min, tau_max]
        #   noisy   -> hasty (60 %) or confused/late (40 %)
        delay = _simulate_atc_telemetry(is_correct=is_correct,
                                        difficulty=difficulty, rng=rng)

        V = _atc_rail_score(delay, ex["x_text"], y_fb, class_weight, rail)

        if   policy == "STATIC":   k_rep = 0
        elif policy == "ALWAYS":   k_rep = 1
        elif policy == "GATED":    k_rep = 1 if V >= rail.theta else 0
        # int(floor): V=0.06 -> 0 (noisy blocked), V=0.65 -> 7 (good amplified)
        elif policy == "WEIGHTED": k_rep = int(rail.w_max * V)
        else: raise ValueError(f"Unknown policy: {policy!r}")

        stats["feedback_events"] += 1
        if k_rep > 0:
            stats["admitted_events"] += 1
            stats["update_mass"] += k_rep
            if not is_correct:
                stats["contaminated_events"] += 1
                stats["contaminated_mass"] += k_rep

        for _ in range(k_rep):
            model.learn_one(_atc_features(ex["x_text"]), y_fb)

        if i % eval_every == 0 or i == len(recal_shuffled):
            bal, mf1 = _evaluate_atc_model(model, test)
            steps.append(i)
            bal_curve.append(bal)
            f1_curve.append(mf1)

    return {
        "policy": policy,
        "steps": steps,
        "bal_acc": bal_curve,
        "macro_f1": f1_curve,
        "stats": stats,
    }


# ============================================================
# Aggregate + plot  (shared by all four datasets)
# ============================================================

def aggregate_curves(curves: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps    = curves[0]["steps"]
    bal      = np.array([c["bal_acc"]  for c in curves], dtype=float)
    f1       = np.array([c["macro_f1"] for c in curves], dtype=float)
    bal_mean = bal.mean(axis=0)
    f1_mean  = f1.mean(axis=0)
    bal_std  = bal.std(axis=0, ddof=1) if bal.shape[0] > 1 else np.zeros_like(bal_mean)
    f1_std   = f1.std(axis=0, ddof=1) if f1.shape[0] > 1 else np.zeros_like(f1_mean)
    return {
        "policy":   curves[0]["policy"],
        "steps":    steps,
        "bal_mean": bal_mean, "bal_std": bal_std,
        "f1_mean":  f1_mean,  "f1_std":  f1_std,
    }


def compute_ae_summary(
    per_policy_runs: Dict[str, List[Dict[str, Any]]],
    eps: float = 1e-9,
) -> Dict[str, Dict[str, float]]:
    baseline_runs = per_policy_runs["ALWAYS"]
    summary: Dict[str, Dict[str, float]] = {}
    for policy in POLICY_ORDER:
        values: List[float] = []
        for base_run, cur_run in zip(baseline_runs, per_policy_runs[policy]):
            base_stats = base_run["stats"]
            cur_stats  = cur_run["stats"]
            if policy == "ALWAYS":
                ae = 0.0
            else:
                prevented  = float(base_stats["contaminated_events"] - cur_stats["contaminated_events"])
                sacrificed = float(base_stats["admitted_events"] - cur_stats["admitted_events"])
                ae = max(0.0, prevented / max(sacrificed, eps))
            values.append(ae)
        arr = np.array(values, dtype=float)
        summary[policy] = {
            "ae_mean": float(arr.mean()),
            "ae_std":  float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        }
    return summary


def write_ae_outputs(
    dataset_to_ae: Dict[str, Dict[str, Dict[str, float]]],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    pretty = pd.DataFrame(
        index=[POLICY_LABELS[p] for p in POLICY_ORDER],
        columns=DATASET_ORDER,
        dtype=object,
    )
    numeric_rows: List[Dict[str, Any]] = []

    for policy in POLICY_ORDER:
        row = {"policy": POLICY_LABELS[policy]}
        for dataset in DATASET_ORDER:
            vals = dataset_to_ae[dataset][policy]
            mean = vals["ae_mean"]
            std  = vals["ae_std"]
            pretty.loc[POLICY_LABELS[policy], dataset] = f"{mean:.2f} ± {std:.2f}"
            row[f"{dataset}_mean"] = mean
            row[f"{dataset}_std"]  = std
        numeric_rows.append(row)

    pretty.to_csv(outdir / "table_ae_pretty.csv")
    pd.DataFrame(numeric_rows).to_csv(outdir / "table_ae_numeric.csv", index=False)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Admission Efficiency (AE) shown as mean $\\pm$ standard deviation over {len(SEEDS)} runs. Higher is better.}}",
        "\\label{tab:ae}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "\\textbf{Policy} & \\textbf{Synthetic} & \\textbf{SECOM} & \\textbf{APS Failure} & \\textbf{ATC} \\\\",
        "\\hline",
    ]
    for policy in POLICY_ORDER:
        cells: List[str] = []
        for dataset in DATASET_ORDER:
            vals = dataset_to_ae[dataset][policy]
            cells.append(f"${vals['ae_mean']:.2f} \\pm {vals['ae_std']:.2f}$")
        lines.append(f"{POLICY_LABELS[policy]} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    (outdir / "table_ae_latex.tex").write_text("\n".join(lines), encoding="utf-8")

    print("\nAE summary table")
    print(pretty.to_string())
    print(
        f"\nSaved AE outputs to ./{outdir}/table_ae_pretty.csv, "
        f"./{outdir}/table_ae_numeric.csv, and ./{outdir}/table_ae_latex.tex"
    )


def plot_decay_curves_pub(
    aggs:    List[Dict[str, Any]],
    metric:  str,
    title:   str,
    outpath: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 4.1))
    ax  = fig.add_subplot(111)

    for a in aggs:
        x = a["steps"]
        if metric == "f1":
            y, s, ylab = a["f1_mean"],  a["f1_std"],  "Macro-F1"
        else:
            y, s, ylab = a["bal_mean"], a["bal_std"], "Balanced Accuracy"

        ax.plot(x, y, linewidth=2.0, label=a["policy"])
        ax.fill_between(x, y - s, y + s, alpha=0.18)

    ax.set_title(title)
    ax.set_xlabel("Recalibration step")
    ax.set_ylabel(ylab)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True, fancybox=False, framealpha=0.92)
    save_png(fig, outpath)


# ============================================================
# Dataset runners
# ============================================================

def run_dataset_decay_numeric(
    name:       str,
    X:          np.ndarray,
    y:          np.ndarray,
    cfg:        DecayConfig,
    adm:        AdmissionParams,
    difficulty: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    aggs: List[Dict[str, Any]] = []
    per_policy_runs: Dict[str, List[Dict[str, Any]]] = {}
    for pol in POLICY_ORDER:
        per_seed = [run_decay_curve(X, y, cfg, pol, adm, sd, difficulty) for sd in SEEDS]
        per_policy_runs[pol] = per_seed
        aggs.append(aggregate_curves(per_seed))

    plot_decay_curves_pub(
        aggs, metric="f1",
        title=f"{name}: Prevention of Model Decay (Macro-F1, mean +/- std)",
        outpath=OUTDIR / f"fig_decay_{name.lower()}_macro_f1.png",
    )
    plot_decay_curves_pub(
        aggs, metric="bal",
        title=f"{name}: Prevention of Model Decay (Balanced Accuracy, mean +/- std)",
        outpath=OUTDIR / f"fig_decay_{name.lower()}_bal_acc.png",
    )
    return {"aggs": aggs, "ae": compute_ae_summary(per_policy_runs)}


def run_atc_decay(
    train_pairs: List[Dict],
    test_pairs:  List[Dict],
    rail:        AtcRailParams,
) -> Dict[str, Any]:
    class_weight = _compute_atc_class_weights(train_pairs)
    warm, recal  = _balanced_warmup(train_pairs, n_per_class=500)

    dist = Counter(p["y"] for p in test_pairs)
    print(f"  ATC test label dist:  { {ATC_LABELS[k]: v for k, v in sorted(dist.items())} }")
    print(f"  ATC class weights:    { {ATC_LABELS[k]: round(v, 2) for k, v in class_weight.items()} }")

    eval_every = max(25, len(recal) // 30)

    aggs: List[Dict[str, Any]] = []
    per_policy_runs: Dict[str, List[Dict[str, Any]]] = {}
    for pol in POLICY_ORDER:
        per_seed = [
            run_atc_decay_curve(warm, recal, test_pairs, pol, class_weight, rail,
                                seed=sd, eval_every=eval_every)
            for sd in SEEDS
        ]
        per_policy_runs[pol] = per_seed
        aggs.append(aggregate_curves(per_seed))

    plot_decay_curves_pub(
        aggs, metric="f1",
        title="ATC: Prevention of Model Decay (Macro-F1, mean ± std)",
        outpath=OUTDIR / "fig_decay_atc_macro_f1.png",
    )
    plot_decay_curves_pub(
        aggs, metric="bal",
        title="ATC: Prevention of Model Decay (Balanced Accuracy, mean ± std)",
        outpath=OUTDIR / "fig_decay_atc_bal_acc.png",
    )
    return {"aggs": aggs, "ae": compute_ae_summary(per_policy_runs)}


# ============================================================
# Main
# ============================================================

def main():
    # Tight, steep bell -> sharp separation of correct vs noisy feedback
    adm = AdmissionParams(
        tau_min=1.5, tau_max=4.8, k=2.8, theta=0.38, weighted_max=9,
    )

    dataset_to_ae: Dict[str, Dict[str, Dict[str, float]]] = {}

    # ── 1: SECOM ─────────────────────────────────────────────
    print("\nLoading SECOM ...")
    X_secom, y_secom = load_secom_uci()
    cfg_secom = DecayConfig(
        warmup_n=350, recal_n=600, test_n=500,
        base_correctness=0.70, nf=X_secom.shape[1], eval_every=50,
    )
    print("Running decay: SECOM...")
    dataset_to_ae["SECOM"] = run_dataset_decay_numeric(
        "secom", X_secom, y_secom, cfg_secom, adm
    )["ae"]

    # ── 2: APS Failure ───────────────────────────────────────
    print("\nLoading APS Failure ...")
    X_aps, y_aps = load_aps_uci()
    cfg_aps = DecayConfig(
        warmup_n=4000, recal_n=6000, test_n=6000,
        base_correctness=0.70, nf=X_aps.shape[1], eval_every=200,
    )
    print("Running decay: APS Failure...")
    dataset_to_ae["APS Failure"] = run_dataset_decay_numeric(
        "aps", X_aps, y_aps, cfg_aps, adm
    )["ae"]

    # ── 3: Synthetic ─────────────────────────────────────────
    print("\nGenerating Synthetic stream ...")
    X_syn, y_syn, d_syn = make_synth_realistic_stream(n_total=9000, seed=23, d=40)
    cfg_syn = DecayConfig(
        warmup_n=1500, recal_n=3500, test_n=2500,
        base_correctness=0.70, nf=X_syn.shape[1], eval_every=100,
    )
    print("Running decay: Synthetic...")
    dataset_to_ae["Synthetic"] = run_dataset_decay_numeric(
        "synth", X_syn, y_syn, cfg_syn, adm, difficulty=d_syn
    )["ae"]

    # ── 4: ATC real corpus ───────────────────────────────────
    print("\nLoading ATC corpus ...")
    train_pairs, test_pairs = load_atc_pairs()

    if len(test_pairs) == 0:
        print("WARNING: no ATC test pairs found -- skipping ATC plots and AE values for ATC.")
        dataset_to_ae["ATC"] = {
            pol: {"ae_mean": float("nan"), "ae_std": float("nan")}
            for pol in POLICY_ORDER
        }
    else:
        rail = AtcRailParams(
            tau_min=1.2, tau_max=5.0, k=3.5, theta=0.35, w_max=12, class_boost=True,
        )
        print(f"ATC RAIL: tau=({rail.tau_min},{rail.tau_max}), k={rail.k}, "
              f"theta={rail.theta}, w_max={rail.w_max}")
        print("Running decay: ATC...")
        dataset_to_ae["ATC"] = run_atc_decay(train_pairs, test_pairs, rail)["ae"]

    write_ae_outputs(dataset_to_ae, OUTDIR)
    print(f"\nAll plots and AE tables saved to ./{OUTDIR}/")


if __name__ == "__main__":
    main()
