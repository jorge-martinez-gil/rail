# RAIL

**Trust-aware Human-in-the-Loop validation for streaming predictions — with empirical evidence that it prevents model decay.**

RAIL is a framework for gating online model recalibration through operator attention quality. It has two components: a **live operator console** for real-time prediction validation, and an **experiment harness** that demonstrates RAIL-gated updates outperform both static models and naïve update-everything strategies across four datasets.

---

## Overview

Machine learning models deployed on live streams decay as the data distribution shifts. The standard fix — continuous retraining on operator feedback — backfires when labels are noisy: hurried or confused operators introduce errors that compound over time. RAIL solves this by computing a **Vigilance Index (V)** for every operator decision, measuring *how carefully* the label was produced, and only admitting high-quality feedback into the retraining loop.

The project ships two artifacts:

| Component | File | Purpose |
|---|---|---|
| **RAIL Console** | `app.html` | Browser-based HITL dashboard for live validation, vigilance scoring, export, and one-click recalibration over MQTT |
| **Decay Experiments** | `decay_experiments.py` | Reproducible simulation comparing the update policies across four datasets |

---

## Vigilance Index

Both components share the same core admission score — a sigmoid bell curve over deliberation time, adjusted for task complexity:

```
Δ  = t_decision − t_anchor
β  = w_features · n_features + w_edits · n_edits + w_focus · focus_seconds
s_fast = σ(k · (w_latency · Δ − (τ_min + β)))
s_slow = σ(k · ((τ_max + β) − w_latency · Δ))
V  = s_fast · s_slow
```

The intuition is a Goldilocks window: decisions that are too fast (hasty, likely rubber-stamped) or too slow (confused, likely unreliable) receive low V scores. Only decisions made within the sweet spot — adjusted per-row for complexity — pass the gate.

| Parameter | Meaning |
|---|---|
| **τ\_min / τ\_max** | Goldilocks window boundaries (seconds) |
| **k** | Sigmoid steepness — higher = sharper gate |
| **β** | Per-row complexity bonus (features inspected, edits made, focus time) |
| **θ** | Eligibility threshold — only rows with V ≥ θ are admitted for retraining |

---

## RAIL Console (`app.html`)

A single-file browser application that connects to any MQTT broker and provides a full operator workstation for streaming predictions.

### Features

- **Live MQTT streaming** — Subscribes to prediction and feature topics; displays results in real time.
- **CSV batch mode** — Load a CSV and publish rows one at a time through the broker for offline review.
- **Per-row validation** — Mark each prediction OK or Non-OK. Target values become editable only for Non-OK rows.
- **Deliberation window visualization** — Each row has an expandable timing panel showing a live progress bar, the V(Δ) admission curve, and full diagnostic breakdown.
- **Real-time monitoring** — Plotly-powered charts for Predicted vs Target, absolute error, decision counts, and vigilance over time. All charts are expandable to full-screen.
- **Filtered export** — Export only Non-OK rows meeting the vigilance threshold as structured JSON (`RAIL.non_ok_export.v4` schema).
- **One-click recalibration** — Publish eligible Non-OK rows back to the broker on a `/recalibrate` topic.
- **KPI dashboard** — Rows in queue, eligible rejections, average V, median Δ.
- **Tunable parameters** — All vigilance parameters adjustable live, with three presets (Conservative, Balanced, Aggressive).

### Running the Console

1. Open `app.html` in any modern browser — no build step required.
2. Click **Load Config (auto-connect)** and select a JSON config file.
3. Once connected, load a CSV or wait for live data.

### Configuration File

```json
{
  "brokerURL": "ws://localhost:9001",
  "inputTopic": "model/input",
  "outputTopic": "model/output",
  "inputs": [
    { "name": "feature_1" },
    { "name": "feature_2" },
    { "name": "feature_3" }
  ]
}
```

| Field | Description |
|---|---|
| `brokerURL` | WebSocket URL of the MQTT broker |
| `inputTopic` | Topic to publish feature rows to (the model consumes this) |
| `outputTopic` | Topic the model publishes predictions on (expects `{ "median": … }`) |
| `inputs` | Array of feature descriptors with `name` fields matching CSV columns |

### CSV Format

```
<ignored>, <ignored>, feature_1, feature_2, ..., feature_N, target
```

The first two columns are skipped. Feature columns map to the `inputs` array in the config. The last column is the target value.

### Interface Overview

| Area | Description |
|---|---|
| **Sidebar — Connection** | Load config and connect/disconnect from the MQTT broker |
| **Sidebar — Data** | Load CSV, export eligible Non-OK rows, trigger recalibration |
| **Sidebar — Vigilance** | Tune all V parameters and presets; formula reference |
| **Top bar** | Connection status, OK/Non-OK counts, eligible count, broker info |
| **KPI cards** | Rows in queue, eligible rejections, avg V, median Δ |
| **Validation queue** | Filterable, searchable table with per-row status, notes, and V |
| **Monitoring charts** | Predicted vs Target + error, decision counts bar chart, V series |

### Export Schema

Exported JSON follows the `RAIL.non_ok_export.v4` schema:

```json
{
  "schema": "RAIL.non_ok_export.v4",
  "exportedAt": "2026-03-20T10:30:00.000Z",
  "theta": 0.60,
  "params": { "tauMin": 0.8, "tauMax": 6.0, "k": 1.2 },
  "counts": { "non_ok_total": 42, "non_ok_eligible": 28 },
  "rows": [
    {
      "Timestamp": "...",
      "Status": "non-ok",
      "Features": { "feature_1": "0.5" },
      "Target": "1.23",
      "Predicted": "1.18",
      "OperatorNote": "drift suspected",
      "Vigilance": 0.847,
      "Diagnostics": {
        "anchor_reason": "click",
        "delta_s": 2.31,
        "beta": 0.14,
        "z_fast": 1.81,
        "z_slow": 4.43,
        "s_fast": 0.859,
        "s_slow": 0.988,
        "edits": 1,
        "focus_ms": 1200
      }
    }
  ]
}
```

---

## Decay Experiments (`decay_experiments.py`)

A reproducible simulation that demonstrates RAIL-gated recalibration prevents model decay across four datasets and two metrics (Macro-F1, Balanced Accuracy), averaged over five random seeds with standard-deviation bands.

### Datasets

**SECOM** — UCI semiconductor manufacturing process data. Binary classification (pass/fail), 590 features, high missingness. Median-imputed.

**APS Failure at Scania Trucks** — UCI predictive maintenance dataset. Binary classification (neg/pos), 170 features, combined training+test splits.

**Synthetic Stream** — A purpose-built stream with controlled concept drift (two hard regime shifts at 35% and 70%), class imbalance, varying missingness, and workload spikes. 40 features, latent factor structure. Designed to make STATIC visibly decay and GATED/WEIGHTED visibly recover.

**ATC (Air Traffic Control)** — Real spoken ATC corpus from Hugging Face (`Jzuluaga/atco2_corpus_1h` + `Jzuluaga/uwb_atcc`). Pilot→Controller pairs are extracted and labelled into four intent classes: ALTITUDE, HEADING, CONTACT, OTHER. Features are bag-of-words, character n-grams, keyword indicators, and structural cues. A domain-specific RAIL score includes class weighting and utterance quality.

### Running the Experiments

```bash
pip install numpy pandas scikit-learn river matplotlib datasets
python decay_experiments.py
```

Outputs are saved as 600 DPI PNGs to `./outputs/`:

```
fig_decay_secom_macro_f1.png    fig_decay_secom_bal_acc.png
fig_decay_aps_macro_f1.png      fig_decay_aps_bal_acc.png
fig_decay_synth_macro_f1.png    fig_decay_synth_bal_acc.png
fig_decay_atc_macro_f1.png      fig_decay_atc_bal_acc.png
```

### Simulated Operator Telemetry

Since the experiment runs offline without a real operator, it simulates telemetry that mirrors the patterns observed in live HITL workflows:

**Correct labels** receive deliberate timing — delta drawn from a Gaussian centred around 2.4–2.8 s, landing squarely inside the [τ\_min, τ\_max] window. This produces high V scores.

**Noisy labels** receive one of two timing signatures: hasty (60–65% of errors) with delta around 0.2 s, far below τ\_min; or confused (35–40% of errors) with delta around 9 s, far above τ\_max. Both produce near-zero V scores.

Additional realism includes workload spikes (random bursts that depress correctness), per-sample difficulty (based on proximity to decision boundaries or drift transitions), and variable feature inspection depth and edit counts.

### Default Admission Parameters

| Parameter | Numeric datasets | ATC dataset |
|---|---|---|
| τ\_min | 1.5 s | 1.2 s |
| τ\_max | 4.8 s | 5.0 s |
| k | 2.8 | 3.5 |
| θ | 0.38 | 0.35 |
| Replay max | 9 | 12 |

### Online Learning Model

All experiments use [River](https://riverml.xyz/) for online (single-pass) learning. Numeric datasets use `LogisticRegression` with SGD behind a `StandardScaler`. The ATC dataset uses `SoftmaxRegression` with AdaGrad for multiclass classification. Models are warmed up on a held-out portion before the recalibration stream begins.

---

## Project Structure

```
.
├── app/                      # RAIL Console (browser, zero-build)
├── experiments/              # Decay prevention experiment harness
├── outputs/                  # Generated plots (600 DPI PNG)
├── _cache/                   # Downloaded dataset cache (auto-created)
└── README.md
```

---

## Technology

| Layer | Stack |
|---|---|
| **Console** | Single HTML file — MQTT.js, Plotly.js, Inter typeface (all from CDN). No server, no framework, no bundler. |
| **Experiments** | Python 3.8+ — NumPy, pandas, scikit-learn, River, Matplotlib, Hugging Face Datasets |

---

## License

See the project license file for terms.
