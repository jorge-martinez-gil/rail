"""Auditable, session-clustered analysis for the September paper release.

Run from the repository root: python -m experiments.study_release
Only JSON exports are counted; CSV exports are duplicate representations.
No threshold is fitted to these outcomes. Session IDs are clustering units,
not independently verified identities. Requires NumPy; no other dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.human_study import audit_session, load_sessions


def counts(records):
    return np.array([
        len(records), sum(r['admitted'] for r in records),
        sum(r['operator_label'] != r['truth'] for r in records),
        sum(r['admitted'] and r['operator_label'] != r['truth'] for r in records),
    ], dtype=float)


def metrics(x):
    n, a, e, ae = np.moveaxis(np.asarray(x), -1, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return {
            'yield': a / n, 'base_error': e / n, 'admitted_error': ae / a,
            'withheld_error': (e - ae) / (n - a),
            'clean_retention': (a - ae) / (n - e),
            'false_admission': ae / e,
            'admission_efficiency': (e - ae) / (n - a),
            'error_reduction': e / n - ae / a,
        }


def cluster_bootstrap(groups, replicates=20000, seed=20260917):
    """Resample complete sessions within condition, retaining condition sizes."""
    rng = np.random.default_rng(seed)
    samples = np.zeros((replicates, 4))
    for group in groups:
        x = np.asarray(group)
        indices = rng.integers(0, len(x), size=(replicates, len(x)))
        samples += x[indices].sum(axis=1)
    result = {}
    for name, values in metrics(samples).items():
        valid = values[np.isfinite(values)]
        result[name] = {
            'ci95': np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None,
            'defined_replicates': len(valid),
        }
    return result


def analyse_release(inputs, output, replicates=20000):
    sessions = load_sessions([Path(inputs)])
    ids = [s['participant'] for s in sessions]
    if len(set(ids)) != len(ids):
        raise ValueError('Repeated participant IDs: cluster sessions by person before analysis.')
    defaults = {json.dumps(s['rail_defaults'], sort_keys=True) for s in sessions}
    if len(defaults) != 1:
        raise ValueError('Mixed gate defaults require separate analysis.')
    manifest, audited = [], []
    for s in sessions:
        records = s['records']
        if not records or len({r['trial'] for r in records}) != len(records):
            raise ValueError('Empty session or repeated trial index.')
        for r in records:
            if r['condition'] != s['condition']:
                raise ValueError('Condition mismatch.')
            if int(r['contaminated']) != int(r['operator_label'] != r['truth']):
                raise ValueError('Contamination flag disagrees with labels.')
            if r['admitted'] not in (0, 1) or not 0 <= r['V'] <= 1:
                raise ValueError('Invalid admission or score.')
        a = audit_session(s, tolerance=0.00025)
        if a.v_mismatches or a.admit_mismatches:
            raise ValueError('Score/admission audit failed.')
        audited.append(a)
        path = Path(s['_source'])
        manifest.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'alias': s['_alias'], 'condition': s['condition'], 'trials': len(records)})
    conditions = sorted({s['condition'] for s in sessions})
    groups = [[counts(s['records']) for s in sessions if s['condition'] == c] for c in conditions]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for condition in conditions + ['pooled']:
        subset = [s for s in sessions if condition == 'pooled' or s['condition'] == condition]
        records = [r for s in subset for r in s['records']]
        x = counts(records)
        applicable = groups if condition == 'pooled' else [groups[conditions.index(condition)]]
        summaries[condition] = {
            'sessions': len(subset), 'events': int(x[0]), 'admitted': int(x[1]),
            'errors': int(x[2]), 'admitted_errors': int(x[3]),
            'correct_withheld': int(x[0] - x[1] - x[2] + x[3]),
            'point': {k: float(v) for k, v in metrics(x).items()},
            'bootstrap': cluster_bootstrap(applicable, replicates),
            'interrupted_events': sum(r['interruptions'] > 0 for r in records),
            'median_delta_s': float(np.median([r['delta_t_s'] for r in records])),
            'model_accuracy': float(np.mean([r['truth'] == r['model_flag'] for r in records])),
            'error_sessions': sum(any(r['contaminated'] for r in s['records']) for s in subset),
        }
    loo = []
    for omit in range(len(sessions)):
        x = sum((counts(s['records']) for i, s in enumerate(sessions) if i != omit), np.zeros(4))
        loo.append(float(metrics(x)['error_reduction']))
    result = {
        'seed': 20260917, 'replicates': replicates, 'method': 'condition-stratified session-cluster percentile bootstrap',
        'notes': ['Exploratory intervals; sparse errors limit coverage.',
                  'Zero observed errors in a subset do not establish zero population risk.',
                  'No causal condition comparison; assignment and identity verification unavailable.',
                  'Same-sample Bayes substitution is an identity, not an independent certificate.'],
        'conditions': summaries, 'sources': manifest,
        'audit': {'max_v_deviation': max(a.max_abs_v_deviation for a in audited),
                  'admission_mismatches': sum(a.admit_mismatches for a in audited),
                  'duration_s_range': [min(a.duration_s for a in audited), max(a.duration_s for a in audited)],
                  'date_range': [min(s['started'] for s in sessions), max(s['started'] for s in sessions)],
                  'distinct_item_sequences': len({tuple((r['truth'], r['model_flag'], r['n_features']) for r in s['records']) for s in sessions})},
        'leave_one_session_out_error_reduction_range': [min(loo), max(loo)],
    }
    september = [s for s in sessions if s['started'].startswith('2026-09')]
    if september:
        september_counts = sum((counts(s['records']) for s in september), np.zeros(4))
        result['september_only'] = {
            'sessions': len(september), 'counts_n_admitted_errors_admitted_errors': september_counts.astype(int).tolist(),
            'point': {k: float(v) for k, v in metrics(september_counts).items()},
            'bootstrap': cluster_bootstrap([
                [counts(s['records']) for s in september if s['condition'] == c]
                for c in conditions if any(s['condition'] == c for s in september)
            ], replicates),
        }
    (output / 'study_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output / 'session_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['session', 'condition', 'events', 'admitted', 'errors', 'admitted_errors'])
        for s in sessions:
            writer.writerow([s['_alias'], s['condition'], *map(int, counts(s['records']))])
    lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Condition & Sessions & Events & Admitted & Errors & Errors adm. & Clean withheld \\', r'\midrule']
    for name, s in summaries.items():
        lines.append(f"{name.title()} & {s['sessions']} & {s['events']} & {s['admitted']} & {s['errors']} & {s['admitted_errors']} & {s['correct_withheld']} " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (output / 'table_study.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'sources'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='?', default='real tests')
    parser.add_argument('--output', default='paper_release/study')
    parser.add_argument('--replicates', type=int, default=20000)
    args = parser.parse_args()
    analyse_release(args.inputs, args.output, args.replicates)
