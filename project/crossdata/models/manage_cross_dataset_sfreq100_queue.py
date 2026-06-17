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
SUMMARY = PROJECT / 'cross_dataset_sfreq100_results_summary.md'
PREP = PROJECT / 'preprocessed_sfreq100'

EXPERIMENTS = [
    {
        'priority': 1,
        'name': 'EA+CSPNet',
        'run_id': '20260612_sfreq100_ea_cspnet',
        'model': 'cspnet',
        'metric': 'base',
        'args': ['--both', '--model', 'cspnet', '--ea'],
    },
    {
        'priority': 2,
        'name': 'EA+AdaBN+Con',
        'run_id': '20260612_sfreq100_ea_adabn_con',
        'model': 'cspnetcontrastive',
        'metric': 'adabn',
        'args': ['--both', '--model', 'cspnetcontrastive', '--ea', '--adabn'],
    },
    {
        'priority': 3,
        'name': 'EA+Snapx6',
        'run_id': '20260612_sfreq100_ea_snapx6',
        'model': 'cspnet',
        'metric': 'unsupported',
        'args': None,
        'note': 'Not executed: cross_dataset.py has no snapshot ensemble evaluation path yet.',
    },
    {
        'priority': 4,
        'name': 'EA+AdaBN',
        'run_id': '20260612_sfreq100_ea_adabn',
        'model': 'cspnet',
        'metric': 'adabn',
        'args': ['--both', '--model', 'cspnet', '--ea', '--adabn'],
    },
    {
        'priority': 5,
        'name': 'EA+TENT',
        'run_id': '20260612_sfreq100_ea_tent',
        'model': 'cspnet',
        'metric': 'tent',
        'args': ['--both', '--model', 'cspnet', '--ea', '--tent'],
    },
    {
        'priority': 6,
        'name': 'SubjClust tau=5 analogue',
        'run_id': '20260612_sfreq100_datasetea_ea_sourceweight_tau5',
        'model': 'cspnet',
        'metric': 'base',
        'args': ['--both', '--model', 'cspnet', '--dataset_ea', '--ea', '--source_weighting', '--source_weight_tau', '5.0'],
    },
]

METRIC_COLUMNS = {
    'base': ('acc', 'kappa'),
    'adabn': ('adabn_acc', 'adabn_kappa'),
    'tent': ('tent_acc', 'tent_kappa'),
}

def csv_path(exp, src, tgt):
    return RESULTS / f"loso_results_{exp['run_id']}_cross_{src}_to_{tgt}_{exp['model']}.csv"

def read_metric(path, metric):
    if metric not in METRIC_COLUMNS or not path.exists():
        return 0, None, None
    acc_col, kappa_col = METRIC_COLUMNS[metric]
    accs, kaps = [], []
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            if row.get(acc_col) not in ('', None):
                accs.append(float(row[acc_col]))
            if row.get(kappa_col) not in ('', None):
                kaps.append(float(row[kappa_col]))
    return len(accs), (sum(accs)/len(accs) if accs else None), (sum(kaps)/len(kaps) if kaps else None)

def write_summary(status_note=''):
    lines = [
        '# Cross-Dataset srate-100Hz Performance Summary', '',
        f'Last updated: `{datetime.now():%Y-%m-%d %H:%M}` KST', '',
        '- Preprocessed dir: `/home/hkim/MI_test/preprocessed_sfreq100`',
        '- Cho2017: resampled from 128Hz/T=257 to 100Hz/T=201',
        '- Lee2019: native/current 100Hz/T=201 copy',
        '- Evaluation script: `/home/hkim/MI_test/MI_loso_project/cross_dataset.py`',
        '- Environment: `MI_PREPROCESSED_DIR=/home/hkim/MI_test/preprocessed_sfreq100`, `MI_N_TIMES=201`', '',
        '| Priority | Method | Status | Cho->Lee | Lee->Cho | Note |',
        '|---:|---|---|---:|---:|---|',
    ]
    for exp in EXPERIMENTS:
        if exp['args'] is None:
            lines.append(f"| {exp['priority']} | {exp['name']} | not run |  |  | {exp.get('note','')} |")
            continue
        vals = []
        status = 'pending'
        for src, tgt in [('cho2017','lee2019'), ('lee2019','cho2017')]:
            n, acc, kap = read_metric(csv_path(exp, src, tgt), exp['metric'])
            vals.append((n, acc, kap))
        if vals[0][0] and vals[1][0]:
            status = 'completed' if vals[0][0] >= 54 and vals[1][0] >= 52 else 'partial'
        elif vals[0][0] or vals[1][0]:
            status = 'partial'
        def fmt(v):
            n, acc, kap = v
            return f'{acc:.2f}% / k={kap:.3f} (n={n})' if acc is not None else f'n={n}'
        lines.append(f"| {exp['priority']} | {exp['name']} | {status} | {fmt(vals[0])} | {fmt(vals[1])} |  |")
    if status_note:
        lines += ['', '## Queue Status', '', status_note]
    SUMMARY.write_text('\n'.join(lines) + '\n')
    print(SUMMARY)

def run_exp(exp):
    if exp['args'] is None:
        print(f"[skip] {exp['name']}: {exp.get('note','')}", flush=True)
        write_summary(f"- Skipped `{exp['name']}`: {exp.get('note','')}")
        return 0
    log_path = RUNS / f"cross_dataset_sfreq100_{exp['run_id']}.log"
    cmd = [sys.executable, '-u', str(CROSS), *exp['args'], '--run_id', exp['run_id']]
    env = os.environ.copy()
    env['MI_PREPROCESSED_DIR'] = str(PREP)
    env['MI_N_TIMES'] = '201'
    print(f"[run] {exp['priority']} {exp['name']} -> {log_path}", flush=True)
    with log_path.open('a') as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    write_summary(f"- Last completed command: `{exp['name']}` exit={rc}")
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
