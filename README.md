<div align="center">

<h1>
  <br/>
  🛤️ RAIL
  <br/>
  <sub><sup>Reliability-Aware Ingress Layer for Human Feedback in Stream Analytics</sup></sub>
</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/MQTT-Enabled-660066?style=for-the-badge&logo=eclipse-mosquitto&logoColor=white" alt="MQTT"/>
  <img src="https://img.shields.io/badge/Online%20Learning-River-00BFA5?style=for-the-badge" alt="River"/>
  <img src="https://img.shields.io/badge/License-See%20File-lightgrey?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/github/last-commit/jorge-martinez-gil/rail?style=for-the-badge&color=informational" alt="Last commit"/>
</p>

<blockquote>
<strong>Trust-aware Human-in-the-Loop validation for streaming predictions —<br/>with empirical evidence that it prevents model decay.</strong>
</blockquote>

<p>
RAIL is a framework for gating online model recalibration through <em>operator attention quality</em>.<br/>
It ships a <strong>live browser console</strong> for real-time prediction validation and a <strong>reproducible experiment harness</strong> that quantifies decay prevention across four real-world datasets.
</p>

</div>

---

## 📋 Table of Contents

- [Motivation](#-motivation)
- [Architecture at a Glance](#-architecture-at-a-glance)
- [The Vigilance Index](#-the-vigilance-index)
- [RAIL Console](#-rail-console-apphtml)
  - [Features](#features)
  - [Quick Start](#quick-start)
  - [Configuration](#configuration-file)
  - [CSV Format](#csv-format)
  - [Interface Overview](#interface-overview)
  - [Export Schema](#export-schema)
- [Decay Experiments](#-decay-experiments-decay_experimentspy)
  - [Datasets](#datasets)
  - [Running the Experiments](#running-the-experiments)
  - [Simulated Operator Telemetry](#simulated-operator-telemetry)
  - [Admission Parameters](#default-admission-parameters)
  - [Online Learning Model](#online-learning-model)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [License](#-license)

---

## 💡 Motivation

Machine learning models deployed on live streams **decay** as data distributions shift. The standard remedy — continuous retraining on operator feedback — backfires when labels are noisy: hurried or inattentive operators introduce corrupted ground-truth, accelerating the very decay the retraining was supposed to cure.

**RAIL solves this by treating the operator's attention as a first-class signal.** Before any label is admitted for retraining, RAIL measures how the operator deliberated — too fast suggests rubber-stamping; too slow suggests confusion. Only labels whose decision timing falls inside a per-row *Goldilocks window* pass the admission gate. The result is a high-quality, low-noise recalibration signal even under realistic human workloads.

---

## 🏗️ Architecture at a Glance

```
╔══════════════════════════════════════════════════════════════════╗
║                       Live Data Stream                           ║
║            (MQTT topics: model/input ↔ model/output)            ║
╚══════════════════════╤═══════════════════════════════════════════╝
                       │
              ┌────────▼─────────┐
              │   RAIL Console   │  app.html — zero-build browser UI
              │  (Operator HMI)  │  Plotly charts · MQTT.js · Inter
              └────────┬─────────┘
                       │  Per-row deliberation telemetry
              ┌────────▼─────────┐
              │  Vigilance Gate  │  V(Δ) sigmoid admission function
              │  V ≥ θ  →  ✓    │  Complexity-adjusted Goldilocks window
              └────────┬─────────┘
                       │  Eligible Non-OK rows only
              ┌────────▼─────────┐
              │  /recalibrate    │  Structured JSON export (v4 schema)
              │  MQTT publish    │  → River online learner
              └──────────────────┘
```

| Component | File | Purpose |
|:---|:---|:---|
| **RAIL Console** | `app.html` | Browser-based HITL dashboard for live validation, vigilance scoring, export, and one-click recalibration over MQTT |
| **Decay Experiments** | `decay_experiments.py` | Reproducible simulation comparing update policies across four datasets |

---

## 📐 The Vigilance Index

Both components share a single admission score — a **sigmoid bell curve over deliberation time**, adjusted for task complexity:

$$V = \sigma\!\left(k\cdot(w_\lambda\,\Delta - (\tau_\min + \beta))\right) \;\cdot\; \sigma\!\left(k\cdot((\tau_\max + \beta) - w_\lambda\,\Delta)\right)$$

In plain notation:

```
Δ  = t_decision − t_anchor
β  = w_features · n_features + w_edits · n_edits + w_focus · focus_seconds

s_fast = σ( k · (w_λ · Δ − (τ_min + β)) )
s_slow = σ( k · ((τ_max + β) − w_λ · Δ) )
V      = s_fast · s_slow          ∈ (0, 1)
```

The intuition is a **Goldilocks window**: decisions that are too fast (hasty, likely rubber-stamped) or too slow (confused, likely unreliable) receive low V scores. Only decisions made within the sweet spot — deliberate, engaged, proportionate to task complexity — score near 1.

| Parameter | Meaning |
|:---|:---|
| **τ_min / τ_max** | Goldilocks window boundaries (seconds) |
| **k** | Sigmoid steepness — higher = sharper admission gate |
| **β** | Per-row complexity bonus (features inspected, edits made, focus time) |
| **θ** | Eligibility threshold — only rows with V ≥ θ are admitted for retraining |

> **Key property:** β widens the window proportionally to row complexity, rewarding operators who spend more time on harder items rather than penalising them for it.

---

## 🖥️ RAIL Console (`app.html`)

A **single-file browser application** that connects to any MQTT broker and provides a complete operator workstation for streaming prediction validation. No server, no build step, no dependencies to install.

### Features

| Feature | Description |
|:---|:---|
| 📡 **Live MQTT streaming** | Subscribes to prediction and feature topics; displays results in real time |
| 📂 **CSV batch mode** | Load a CSV and publish rows one at a time through the broker for offline review |
| ✅ **Per-row validation** | Mark each prediction OK or Non-OK; target values become editable for Non-OK rows |
| ⏱️ **Deliberation panel** | Expandable timing panel per row: live progress bar, V(Δ) admission curve, full diagnostics |
| 📊 **Real-time monitoring** | Plotly-powered charts — Predicted vs Target, absolute error, decision counts, V over time |
| 📤 **Filtered export** | Export Non-OK rows above the vigilance threshold as structured JSON (`RAIL.non_ok_export.v4`) |
| 🔁 **One-click recalibration** | Publish eligible Non-OK rows back to the broker on `/recalibrate` |
| 📈 **KPI dashboard** | Rows in queue · eligible rejections · average V · median Δ |
| ⚙️ **Tunable parameters** | All vigilance parameters adjustable live, with three presets: *Conservative*, *Balanced*, *Aggressive* |

### Quick Start

```bash
# 1. Open the console — no install needed
open app.html          # or double-click in your file manager

# 2. Click "Load Config (auto-connect)" and select a JSON config file
# 3. Once connected, load a CSV or wait for live MQTT data
```

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

| Field | Type | Description |
|:---|:---:|:---|
| `brokerURL` | `string` | WebSocket URL of the MQTT broker |
| `inputTopic` | `string` | Topic to publish feature rows to (consumed by the model) |
| `outputTopic` | `string` | Topic the model publishes predictions on — expects `{ "median": … }` |
| `inputs` | `array` | Feature descriptors; `name` fields must match CSV column headers |

### CSV Format

```
<ignored>, <ignored>, feature_1, feature_2, ..., feature_N, target
```

> The first two columns are skipped. Feature columns map to the `inputs` array in the config. The last column is the target value.

### Interface Overview

| Area | Description |
|:---|:---|
| **Sidebar — Connection** | Load config and connect / disconnect from the MQTT broker |
| **Sidebar — Data** | Load CSV, export eligible Non-OK rows, trigger recalibration |
| **Sidebar — Vigilance** | Tune all V parameters and presets; formula reference |
| **Top bar** | Connection status · OK/Non-OK counts · eligible count · broker info |
| **KPI cards** | Rows in queue · eligible rejections · avg V · median Δ |
| **Validation queue** | Filterable, searchable table with per-row status, notes, and V |
| **Monitoring charts** | Predicted vs Target + error · decision counts bar chart · V series |

### Export Schema

Exported JSON follows the `RAIL.non_ok_export.v4` schema:

<details>
<summary><strong>Show full schema example</strong></summary>

```json
{
  "schema": "RAIL.non_ok_export.v4",
  "exportedAt": "2026-03-20T10:30:00.000Z",
  "theta": 0.60,
  "params": { "tauMin": 0.8, "tauMax": 6.0, "k": 1.2 },
  "counts": { "non_ok_total": 42, "non_ok_eligible": 28 },
  "rows": [
    {
      "Timestamp": "2026-03-20T10:28:12.000Z",
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

</details>

---

## 🔬 Decay Experiments (`decay_experiments.py`)

A reproducible simulation demonstrating that RAIL-gated recalibration **prevents model decay** across four datasets and two metrics (Macro-F1, Balanced Accuracy), averaged over five random seeds with confidence intervals.

### Datasets

<table>
<thead>
<tr>
  <th>Dataset</th>
  <th>Domain</th>
  <th>Task</th>
  <th>Features</th>
  <th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
  <td><strong>SECOM</strong></td>
  <td>Semiconductor manufacturing</td>
  <td>Binary (pass/fail)</td>
  <td>590</td>
  <td>High missingness; median-imputed</td>
</tr>
<tr>
  <td><strong>APS Failure</strong></td>
  <td>Predictive maintenance (Scania trucks)</td>
  <td>Binary (neg/pos)</td>
  <td>170</td>
  <td>Combined train + test splits</td>
</tr>
<tr>
  <td><strong>Synthetic Stream</strong></td>
  <td>Controlled concept drift</td>
  <td>Binary</td>
  <td>40</td>
  <td>Two hard regime shifts at 35 % and 70 %; class imbalance; workload spikes</td>
</tr>
<tr>
  <td><strong>ATC (Air Traffic Control)</strong></td>
  <td>Spoken language / NLP</td>
  <td>Multi-class intent (4 classes)</td>
  <td>TF-IDF</td>
  <td>Hugging Face corpora; pilot → controller utterance pairs</td>
</tr>
</tbody>
</table>

### Running the Experiments

```bash
# Install dependencies
pip install numpy pandas scikit-learn river matplotlib datasets

# Run all experiments — outputs saved to ./outputs/
python decay_experiments.py
```

Results are saved as **600 DPI PNGs** to `./outputs/`:

```
fig_decay_secom_macro_f1.png      fig_decay_secom_bal_acc.png
fig_decay_aps_macro_f1.png        fig_decay_aps_bal_acc.png
fig_decay_synth_macro_f1.png      fig_decay_synth_bal_acc.png
fig_decay_atc_macro_f1.png        fig_decay_atc_bal_acc.png
```

### Simulated Operator Telemetry

Since the experiment runs offline without a real operator, it simulates telemetry that mirrors patterns observed in live HITL workflows:

| Label Quality | Timing Signature | V Outcome |
|:---|:---|:---|
| ✅ **Correct** | Δ ~ N(2.4–2.8 s, σ=0.3) — inside [τ_min, τ_max] | **High V** |
| ❌ **Hasty error** (60–65 % of errors) | Δ ~ 0.2 s, far below τ_min | **Low V** — rejected |
| ❌ **Confused error** (35–40 % of errors) | Δ ~ 9 s, far above τ_max | **Low V** — rejected |

Additional realism includes **workload spikes** (random bursts that depress correctness), **per-sample difficulty** (based on proximity to decision boundaries or drift transitions), and **variable feature inspection counts**.

### Default Admission Parameters

| Parameter | Numeric datasets | ATC dataset |
|:---:|:---:|:---:|
| τ_min | 1.5 s | 1.2 s |
| τ_max | 4.8 s | 5.0 s |
| k | 2.8 | 3.5 |
| θ | 0.38 | 0.35 |
| Replay max | 9 | 12 |

### Online Learning Model

All experiments use [**River**](https://riverml.xyz/) for true online (single-pass) learning:

- **Numeric datasets** — `LogisticRegression` with SGD behind a `StandardScaler`
- **ATC dataset** — `SoftmaxRegression` over TF-IDF features, accounting for class imbalance

Three update policies are compared in each run:

| Policy | Description |
|:---|:---|
| **No update** | Baseline — model frozen at deployment time |
| **Unfiltered update** | Retrain on all operator labels regardless of quality |
| **RAIL-gated update** | Retrain only on labels with V ≥ θ — the proposed approach |

---

## 📁 Project Structure

```
rail/
├── app/                      # RAIL Console (browser, zero-build)
│   └── app.html              #   ↳ single-file operator dashboard
├── experiments/              # Decay prevention experiment harness
│   └── decay_experiments.py  #   ↳ reproducible simulation script
├── outputs/                  # Generated plots (600 DPI PNG, git-ignored)
├── _cache/                   # Downloaded dataset cache (auto-created)
└── README.md
```

---

## 🛠️ Technology Stack

<table>
<thead>
<tr><th>Layer</th><th>Technology</th><th>Notes</th></tr>
</thead>
<tbody>
<tr>
  <td><strong>Console — UI</strong></td>
  <td>Vanilla HTML/CSS/JS</td>
  <td>No framework, no bundler, zero build step</td>
</tr>
<tr>
  <td><strong>Console — Charts</strong></td>
  <td>Plotly.js (CDN)</td>
  <td>Real-time time-series and bar charts</td>
</tr>
<tr>
  <td><strong>Console — Messaging</strong></td>
  <td>MQTT.js (CDN)</td>
  <td>WebSocket MQTT — any broker compatible</td>
</tr>
<tr>
  <td><strong>Experiments — Core</strong></td>
  <td>Python 3.8+, NumPy, pandas</td>
  <td>Data wrangling and simulation engine</td>
</tr>
<tr>
  <td><strong>Experiments — ML</strong></td>
  <td>scikit-learn, River</td>
  <td>Batch pre-processing + true online learning</td>
</tr>
<tr>
  <td><strong>Experiments — Data</strong></td>
  <td>Hugging Face Datasets</td>
  <td>Automatic download and caching of ATC corpora</td>
</tr>
<tr>
  <td><strong>Experiments — Visualisation</strong></td>
  <td>Matplotlib</td>
  <td>Publication-quality 600 DPI PNG figures</td>
</tr>
</tbody>
</table>

---

## 📄 License

See the [LICENSE](./LICENSE) file for full terms and conditions.

---

<div align="center">

<sub>
Built with care for the stream analytics and HITL research community.<br/>
If RAIL is useful in your work, consider citing or starring the repository ⭐
</sub>

</div>
