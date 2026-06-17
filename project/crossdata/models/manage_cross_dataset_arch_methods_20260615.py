from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS = PROJECT / 'results'
RUNS = RESULTS / 'runs'
CROSS = ROOT / 'cross_dataset.py'
SUMMARY = PROJECT / 'cross_dataset_arch_methods_20260615.md'
PREP = PROJECT / 'preprocessed_sfreq100'
WAIT_UNIT = 'mi-cross-dataset-arch-baselines-20260615.service'

EXPERIMENTS = [
    dict(name='EEGNet + DSA+SEA', run_id='20260615_sfreq100_eegnet_datasetea_subjectea', model='eegnet', metric='base', args=['--both', '--model', 'eegnet', '--dataset_ea', '--ea']),
    dict(name='EEGNet + EA+TENT', run_id='20260615_sfreq100_eegnet_ea_tent', model='eegnet', metric='tent', args=['--both', '--model', 'eegnet', '--ea', '--tent']),
    dict(name='Conformer + DSA+SEA', run_id='20260615_sfreq100_conformer_datasetea_subjectea', model='conformer', metric='base', args=['--both', '--model', 'conformer', '--dataset_ea', '--ea']),
    dict(name='Conformer + EA+TENT', run_id='20260615_sfreq100_conformer_ea_tent', model='conformer', metric='tent', args=['--both', '--model', 'conformer', '--ea', '--tent']),
]

METRIC_COLUMNS = {
    'base': ('acc', 'kappa'),
    'tent': ('tent_acc', 'tent_kappa'),
    'snap': ('snap_acc', 'snap_kappa'),
    'snap_adabn': ('snap_adabn_acc', 'snap_adabn_kappa'),
}

EXISTING = [
    dict(name='CSPNet + EA+Snapshot', model='cspnet', metric='snap',
         cho=RESULTS/'loso_results_20260613_sfreq100_ea_adabn_snapx6_cross_cho2017_to_lee2019_cspnet.csv',
         lee=RESULTS/'loso_results_20260613_sfreq100_ea_adabn_snapx6_cross_lee2019_to_cho2017_cspnet.csv'),
    dict(name='CSPNet + EA+TENT', model='cspnet', metric='tent',
         cho=RESULTS/'loso_results_20260612_sfreq100_ea_tent_cross_cho2017_to_lee2019_cspnet.csv',
         lee=RESULTS/'loso_results_20260612_sfreq100_ea_tent_cross_lee2019_to_cho2017_cspnet.csv'),
]

def csv_path(exp, src, tgt):
    return RESULTS / f"loso_results_{exp['run_id']}_cross_{src}_to_{tgt}_{exp['model']}.csv"

def read_metric(path: Path, metric: str):
    if not path.exists():
        return 0, None, None
    acc_col, kap_col = METRIC_COLUMNS[metric]
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

def write_summary(note=''):
    lines = [
        '# Cross-Dataset Architecture Method Cells (2026-06-15)', '',
        f'Last updated: `{datetime.now():%Y-%m-%d %H:%M}` KST', '',
        '- Input: `/home/hkim/MI_test/preprocessed_sfreq100`, `MI_N_TIMES=201`',
        '- DSA+SEA means `--dataset_ea --ea` in `cross_dataset.py`.',
        '- TENT results use `tent_acc/tent_kappa`; Snapshot uses `snap_acc/snap_kappa`.', '',
        '| Method | Status | Cho->Lee | Lee->Cho |',
        '|---|---|---:|---:|',
    ]
    for item in EXISTING:
        v1 = read_metric(item['cho'], item['metric'])
        v2 = read_metric(item['lee'], item['metric'])
        status = 'completed' if v1[0] and v2[0] else 'missing'
        lines.append(f"| {item['name']} | {status} | {fmt(v1)} | {fmt(v2)} |")
    for exp in EXPERIMENTS:
        v1 = read_metric(csv_path(exp, 'cho2017', 'lee2019'), exp['metric'])
        v2 = read_metric(csv_path(exp, 'lee2019', 'cho2017'), exp['metric'])
        status = 'completed' if v1[0] and v2[0] else ('partial' if v1[0] or v2[0] else 'pending')
        lines.append(f"| {exp['name']} | {status} | {fmt(v1)} | {fmt(v2)} |")
    if note:
        lines += ['', '## Queue Status', '', note]
    SUMMARY.write_text('\n'.join(lines) + '\n')
    print(SUMMARY, flush=True)

def wait_for_baseline_unit():
    write_summary(f'- Waiting for `{WAIT_UNIT}` to finish before using GPU0.')
    while True:
        rc = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', WAIT_UNIT]).returncode
        if rc != 0:
            break
        time.sleep(60)
    write_summary(f'- `{WAIT_UNIT}` is no longer active; starting missing method cells.')

def run_exp(exp):
    out1 = csv_path(exp, 'cho2017', 'lee2019')
    out2 = csv_path(exp, 'lee2019', 'cho2017')
    if out1.exists() and out2.exists():
        write_summary(f"- Skipped `{exp['name']}` because both CSV files already exist.")
        return 0
    log = RUNS / f"cross_dataset_arch_methods_{exp['run_id']}.log"
    cmd = [sys.executable, '-u', str(CROSS), *exp['args'], '--run_id', exp['run_id']]
    env = os.environ.copy()
    env['MI_PREPROCESSED_DIR'] = str(PREP)
    env['MI_N_TIMES'] = '201'
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    print(f"[run] {exp['name']} -> {log}", flush=True)
    with log.open('a') as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    write_summary(f"- Last completed `{exp['name']}` exit={rc}")
    return rc

def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    wait_for_baseline_unit()
    for exp in EXPERIMENTS:
        rc = run_exp(exp)
        if rc != 0:
            write_summary(f"- Queue stopped at `{exp['name']}` exit={rc}.")
            return rc
    write_summary('- Queue completed.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
