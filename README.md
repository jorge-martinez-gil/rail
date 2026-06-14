<div align="center">

<h1>
  <br/>
  🛤️ RAIL
  <br/>
  <sub><sup>Reliability-Aware Ingress Layer for Human Feedback in Stream Analytics</sup></sub>
</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
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
It ships a <strong>live browser console</strong> for real-time prediction validation and a <strong>publication-oriented experiment harness</strong> that separates real-data evidence from self-contained benchmark-like stress tests.
</p>

</div>

---

## 📋 Table of Contents

- [Motivation](#-motivation)
- [Architecture at a Glance](#-architecture-at-a-glance)
- [The Vigilance Index](#-the-vigilance-index)
  - [Contamination Contract](#contamination-contract)
- [RAIL Console](#-rail-console-apphtml)
  - [Features](#features)
  - [Quick Start](#quick-start)
  - [Configuration](#configuration-file)
  - [CSV Format](#csv-format)
  - [Interface Overview](#interface-overview)
  - [Export Schema](#export-schema)
- [Reproducing the Paper](#-reproducing-the-paper-experimentsreproduce_paperpy)
  - [Benchmark streams](#benchmark-streams)
  - [Running the experiments](#running-the-experiments)
  - [Simulated Operator Telemetry](#simulated-operator-telemetry)
  - [Admission Parameters](#default-admission-parameters)
  - [Online Learning Model](#online-learning-model)
- [Publication Artifact](#-publication-artifact)
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
| **Reproduction driver** | `experiments/reproduce_paper.py` | Single-command runner that produces every table, figure, and statistic in the paper from the frozen seed manifest |

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

### Contamination Contract

The sigmoid product is the telemetry score; the contamination claim is a separate, testable admission contract. Let:

$$A_\theta = \mathbf{1}[V \ge \theta] \qquad C = \mathbf{1}[\text{operator feedback is contaminated}]$$

Define the validation-window rates:

$$\pi = P(C=1), \qquad \alpha_\theta = P(A_\theta=1 \mid C=1), \qquad \rho_\theta = P(A_\theta=1 \mid C=0)$$

Here, \(\alpha_\theta\) is the contaminated-feedback false admission rate and \(\rho_\theta\) is the clean-feedback retention rate. By Bayes' rule:

$$P(C=1 \mid A_\theta=1) =
\frac{\pi \alpha_\theta}
     {\pi \alpha_\theta + (1-\pi)\rho_\theta}$$

Therefore, if calibration data or a validation panel gives \(\alpha_\theta \le \bar{\alpha}\) and \(\rho_\theta \ge \underline{\rho}\), then the admitted recalibration stream is bounded by:

$$P(C=1 \mid A_\theta=1) \le
\frac{\pi \bar{\alpha}}
     {\pi \bar{\alpha} + (1-\pi)\underline{\rho}}$$

RAIL chooses or reports \(\theta\) against this bound rather than treating vigilance as a purely qualitative principle. The discrete experiment tables estimate the same contract with contaminated admissions, admitted yield, and Admission Efficiency:

$$AE_\theta =
\frac{\text{contaminated feedback prevented}}
     {\text{all feedback withheld}}$$

The reference implementation `experiments/rail_core.py::contamination_contract` computes these quantities from validation labels and gate decisions, and the tests verify that the empirical bound matches the Bayes expression.

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
| **Publication activity report** | Locally record opt-in user activity and export aggregate timing, focus, edit, decision, and vigilance summaries (`RAIL.activity_report.v1`) |
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

## 🔬 Reproducing the Paper (`experiments/reproduce_paper.py`)

A single driver produces every table, figure, and statistic in the paper
from the frozen seeds in `SEED_MANIFEST.json`. All four benchmark streams are
generated **self-contained**, so the full artifact runs offline with no
external dataset download. Each preserves the dimensionality, class
imbalance, and workload structure of its real-world reference.

### Benchmark streams

<table>
<thead>
<tr>
  <th>Stream</th>
  <th>Reference domain</th>
  <th>Task</th>
  <th>Features</th>
  <th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
  <td><strong>Synthetic</strong></td>
  <td>Controlled drift + workload</td>
  <td>4-class</td>
  <td>12</td>
  <td>Drift-and-workload generator used to stress the contamination mechanism</td>
</tr>
<tr>
  <td><strong>SECOM-like</strong></td>
  <td>Semiconductor manufacturing</td>
  <td>Binary (pass/fail)</td>
  <td>590</td>
  <td>High-dimensional, severe class imbalance</td>
</tr>
<tr>
  <td><strong>APS-like</strong></td>
  <td>Predictive maintenance (Scania trucks)</td>
  <td>Binary (neg/pos)</td>
  <td>170</td>
  <td>Sparse industrial-fault proxy, rare positives</td>
</tr>
<tr>
  <td><strong>ATC-like</strong></td>
  <td>Air-traffic-control communications</td>
  <td>Multi-class intent (4 classes)</td>
  <td>—</td>
  <td>Intent stream inspired by ATC pilot → controller exchanges</td>
</tr>
</tbody>
</table>

### Running the experiments

```bash
# Install dependencies
python -m pip install -e ".[experiments,dev]"

# Quick sanity check (~30 s)
python -m experiments.reproduce_paper --tier smoke --workers 2 --fast

# Headline tier: 30 seeds, all stages (paper numbers)
python -m experiments.reproduce_paper --tier headline --workers 8 --slow

# Regenerate the two main-text figures from a completed headline run
python -m experiments.make_main_figures \
    --data publication_outputs/self_contained_v3 \
    --out publication_outputs/main_figures

# Run tests for the shared vigilance score and statistics
pytest

# Aggregate exported activity reports from user studies
python -m experiments.activity_report path/to/activity-reports \
    --output-dir publication_outputs/activity
```

See `RUN.md` for every tier, the timing breakdown, and the backend
validation note.

Results are saved below `./publication_outputs/` with a `run_manifest.json` containing the run mode, seed, Python version, platform, and output locations. The canonical runner also writes `JOURNAL_ARTIFACTS.md`, a reviewer-facing index of generated figures, tables, and metadata. Figures are emitted as 600 DPI PNG files plus PDF/SVG vector masters where possible.

```
publication_outputs/
  real_data/
    fig_decay_secom_macro_f1.png      fig_decay_secom_bal_acc.png
    fig_decay_aps_macro_f1.png        fig_decay_aps_bal_acc.png
    fig_decay_synth_macro_f1.png      fig_decay_synth_bal_acc.png
    fig_decay_atc_macro_f1.png        fig_decay_atc_bal_acc.png
  self_contained/
    run_metrics.csv
    summary_metrics.csv
    fig_self_contained_macro_f1.png/.pdf/.svg
    fig_self_contained_admission_tradeoff.png/.pdf/.svg
    table_main_f1.tex
    table_main_f1.csv
    table_main_f1.md
    table_tradeoff.tex
    table_tradeoff.csv
    table_tradeoff.md
    table_rail_macro_f1_audit.csv/.md
    table_rail_ae_audit.csv/.md
  activity/
    activity_session_summary.csv
    activity_condition_summary.csv
    activity_summary.json
```

The RAIL audit tables identify the best method and the best RAIL-family method
per dataset for Macro-F1 and Admission Efficiency. They are intended as an
honest claim check for manuscript wording; the pipeline does not overwrite or
alter empirical outcomes.

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

The real-data pipeline compares these core update policies:

| Policy | Description |
|:---|:---|
| **No update** | Baseline — model frozen at deployment time |
| **Unfiltered update** | Retrain on all operator labels regardless of quality |
| **RAIL-gated update** | Retrain only on labels with V ≥ θ — the proposed approach |
| **RAIL-weighted update** | Replay high-vigilance examples with larger weight for faster adaptation |

The self-contained pipeline additionally compares confidence-gated, loss-gated, and margin-gated baselines calibrated to the RAIL admission yield.

---

## 📦 Publication Artifact

Publication-readiness files are included for reproducibility and review:

| File | Purpose |
|:---|:---|
| `ARTIFACT.md` | Reviewer-facing setup, commands, output layout, and data availability notes |
| `pyproject.toml` | Python metadata, optional dependency groups, and test configuration |
| `requirements.txt` | Flat experiment dependency list |
| `CITATION.cff` | Citation metadata for repository archiving |
| `schemas/` | JSON Schemas for export, recalibration, and activity-report payloads |
| `experiments/activity_report.py` | Aggregates exported browser activity reports into session and condition summaries |
| `.github/workflows/ci.yml` | CI smoke test for the shared RAIL scoring contract |

Generated outputs and downloaded datasets are ignored by Git. For submission, archive `publication_outputs/` and deposit the code/results bundle in a persistent repository.

---

## 📁 Project Structure

```
rail/
├── app/                      # RAIL Console (browser, zero-build)
│   ├── app.html              #   ↳ single-file operator dashboard
│   ├── config.json           #   ↳ demo MQTT configuration
│   └── data.csv              #   ↳ demo CSV stream
├── experiments/              # Reproduction harness
│   ├── rail_core.py          #   ↳ shared admission-score reference
│   ├── reproduce_paper.py    #   ↳ single-command reproduction driver
│   ├── rail_paper.py         #   ↳ self-contained streams + policies
│   ├── baselines_integrated.py #  ↳ noise-robust baselines (ITLM, GCE, ...)
│   ├── regime_sweep.py       #   ↳ 24-cell phase-diagram sweep
│   ├── rail_stats.py         #   ↳ Friedman/Nemenyi/Wilcoxon + BCa/Cliff's δ
│   ├── theory.py             #   ↳ closed-form contract/risk grid
│   └── make_main_figures.py  #   ↳ main-text figure generator
├── tests/                    # Test suite (pytest)
├── schemas/                  # JSON payload contracts
├── paper/                    # LaTeX manuscript and figures
├── SEED_MANIFEST.json        # Frozen seeds and run protocol
├── publication_outputs/      # Generated plots/tables/manifests (git-ignored)
├── ARTIFACT.md
├── RUN.md
├── CITATION.cff
├── pyproject.toml
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
  <td>Python 3.10+, NumPy, pandas</td>
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
