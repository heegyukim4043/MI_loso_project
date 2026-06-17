from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS = PROJECT / 'results'
RUNS = RESULTS / 'runs'
CROSS = ROOT / 'cross_dataset.py'
SUMMARY = PROJECT / 'cross_dataset_sourceweight_backbones_20260616.md'
PREP = PROJECT / 'preprocessed_sfreq100'

EXPERIMENTS = [
    dict(
        name='EEGNet + DSA+SEA+SourceWeight tau=5',
        run_id='20260616_sfreq100_eegnet_datasetea_subjectea_sourceweight_tau5',
        model='eegnet',
        args=['--both', '--model', 'eegnet', '--dataset_ea', '--ea', '--source_weighting', '--source_weight_tau', '5.0'],
    ),
    dict(
        name='Conformer + DSA+SEA+SourceWeight tau=5',
        run_id='20260616_sfreq100_conformer_datasetea_subjectea_sourceweight_tau5',
        model='conformer',
        args=['--both', '--model', 'conformer', '--dataset_ea', '--ea', '--source_weighting', '--source_weight_tau', '5.0'],
    ),
]

BASELINES = [
    ('CSPNet + DSA+SEA+SourceWeight tau=5',
     RESULTS/'loso_results_20260612_sfreq100_datasetea_ea_sourceweight_tau5_cross_cho2017_to_lee2019_cspnet.csv',
     RESULTS/'loso_results_20260612_sfreq100_datasetea_ea_sourceweight_tau5_cross_lee2019_to_cho2017_cspnet.csv'),
    ('CSPNet + DSA+SEA+SessionEA+SourceWeight tau=5',
     RESULTS/'loso_results_20260615_sfreq100_datasetea_subjectea_sessionea_sourceweight_tau5_cross_cho2017_to_lee2019_cspnet.csv',
     RESULTS/'loso_results_20260615_sfreq100_datasetea_subjectea_sessionea_sourceweight_tau5_cross_lee2019_to_cho2017_cspnet.csv'),
]

def csv_path(exp, src, tgt):
    return RESULTS / f"loso_results_{exp['run_id']}_cross_{src}_to_{tgt}_{exp['model']}.csv"

def read_metric(path: Path, acc_col='acc', kap_col='kappa'):
    if not path.exists():
        return 0, None, None
    accs, kaps = [], []
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            if row.get(acc_col) not in ('', None):
                accs.append(float(row[acc_col]))
            if row.get(kap_col) not in ('', None):
                kaps.append(float(row[kap_col]))
    return len(accs), (sum(accs)/len(accs) if accs else None), (sum(kaps)/len(kaps) if kaps else None)

def fmt(v):
    n, acc, kap = v
    if acc is None:
        return f'n={n}'
    return f'{acc:.2f}% / k={kap:.3f} (n={n})'

def avg_pair(v1, v2):
    if v1[1] is None or v2[1] is None:
        return ''
    return f'{(v1[1] + v2[1]) / 2:.2f}%'

def write_summary(note=''):
    lines = [
        '# Cross-Dataset SourceWeight Backbone Check (2026-06-16)', '',
        f'Last updated: `{datetime.now():%Y-%m-%d %H:%M}` KST', '',
        '- Input: `/home/hkim/MI_test/preprocessed_sfreq100`, `MI_N_TIMES=201`',
        '- DSA+SEA = `--dataset_ea --ea`; SourceWeight = covariance-similarity weighted source sampler with `tau=5.0`.', '',
        '| Method | Status | Cho->Lee | Lee->Cho | Avg Acc |',
        '|---|---|---:|---:|---:|',
    ]
    for name, cho, lee in BASELINES:
        v1, v2 = read_metric(cho), read_metric(lee)
        status = 'completed' if v1[0] and v2[0] else 'missing'
        lines.append(f'| {name} | {status} | {fmt(v1)} | {fmt(v2)} | {avg_pair(v1, v2)} |')
    for exp in EXPERIMENTS:
        v1 = read_metric(csv_path(exp, 'cho2017', 'lee2019'))
        v2 = read_metric(csv_path(exp, 'lee2019', 'cho2017'))
        status = 'completed' if v1[0] and v2[0] else ('partial' if v1[0] or v2[0] else 'pending')
        lines.append(f"| {exp['name']} | {status} | {fmt(v1)} | {fmt(v2)} | {avg_pair(v1, v2)} |")
    if note:
        lines += ['', '## Queue Status', '', note]
    SUMMARY.write_text('\n'.join(lines) + '\n')
    print(SUMMARY, flush=True)

def run_exp(exp):
    out1 = csv_path(exp, 'cho2017', 'lee2019')
    out2 = csv_path(exp, 'lee2019', 'cho2017')
    if out1.exists() and out2.exists():
        write_summary(f"- Skipped `{exp['name']}` because both CSV files already exist.")
        return 0
    log = RUNS / f"cross_dataset_sourceweight_backbone_{exp['run_id']}.log"
    env = os.environ.copy()
    env['MI_PREPROCESSED_DIR'] = str(PREP)
    env['MI_N_TIMES'] = '201'
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    cmd = [sys.executable, '-u', str(CROSS), *exp['args'], '--run_id', exp['run_id']]
    print(f"[run] {exp['name']} -> {log}", flush=True)
    with log.open('a') as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    write_summary(f"- Last completed `{exp['name']}` exit={rc}")
    return rc

def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    write_summary('- Queue started.')
    for exp in EXPERIMENTS:
        rc = run_exp(exp)
        if rc != 0:
            write_summary(f"- Queue stopped at `{exp['name']}` exit={rc}.")
            return rc
    write_summary('- Queue completed.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
