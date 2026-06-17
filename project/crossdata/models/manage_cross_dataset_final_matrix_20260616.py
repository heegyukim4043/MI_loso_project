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
SUMMARY = PROJECT / 'cross_dataset_final_matrix_20260616.md'
PREP = PROJECT / 'preprocessed_sfreq100'

EXPERIMENTS = [
    # Priority 1: LOSO matrix symmetry, subject EA + TTA/snapshot for EEGNet/Conformer.
    dict(priority=1, group='SubjEA TTA/Snapshot', name='EEGNet + SubjEA+AdaBN', run_id='20260616_sfreq100_eegnet_ea_adabn', model='eegnet', metric='adabn', args=['--both', '--model', 'eegnet', '--ea', '--adabn']),
    dict(priority=1, group='SubjEA TTA/Snapshot', name='EEGNet + SubjEA+Snapshotx6', run_id='20260616_sfreq100_eegnet_ea_snapx6', model='eegnet', metric='snap', args=['--both', '--model', 'eegnet', '--ea', '--snapshot_ensemble', '--snapshot_T0', '50']),
    dict(priority=1, group='SubjEA TTA/Snapshot', name='Conformer + SubjEA+AdaBN', run_id='20260616_sfreq100_conformer_ea_adabn', model='conformer', metric='adabn', args=['--both', '--model', 'conformer', '--ea', '--adabn']),
    dict(priority=1, group='SubjEA TTA/Snapshot', name='Conformer + SubjEA+Snapshotx6', run_id='20260616_sfreq100_conformer_ea_snapx6', model='conformer', metric='snap', args=['--both', '--model', 'conformer', '--ea', '--snapshot_ensemble', '--snapshot_T0', '50']),

    # Priority 2: cross-dataset TTA on top of 2-stage EA.
    dict(priority=2, group='DSA+SEA TTA', name='CSPNet + DSA+SEA+AdaBN', run_id='20260616_sfreq100_cspnet_datasetea_subjectea_adabn', model='cspnet', metric='adabn', args=['--both', '--model', 'cspnet', '--dataset_ea', '--ea', '--adabn']),
    dict(priority=2, group='DSA+SEA TTA', name='CSPNet + DSA+SEA+TENT', run_id='20260616_sfreq100_cspnet_datasetea_subjectea_tent', model='cspnet', metric='tent', args=['--both', '--model', 'cspnet', '--dataset_ea', '--ea', '--tent']),
    dict(priority=2, group='DSA+SEA TTA', name='EEGNet + DSA+SEA+AdaBN', run_id='20260616_sfreq100_eegnet_datasetea_subjectea_adabn', model='eegnet', metric='adabn', args=['--both', '--model', 'eegnet', '--dataset_ea', '--ea', '--adabn']),
    dict(priority=2, group='DSA+SEA TTA', name='EEGNet + DSA+SEA+TENT', run_id='20260616_sfreq100_eegnet_datasetea_subjectea_tent', model='eegnet', metric='tent', args=['--both', '--model', 'eegnet', '--dataset_ea', '--ea', '--tent']),
    dict(priority=2, group='DSA+SEA TTA', name='Conformer + DSA+SEA+AdaBN', run_id='20260616_sfreq100_conformer_datasetea_subjectea_adabn', model='conformer', metric='adabn', args=['--both', '--model', 'conformer', '--dataset_ea', '--ea', '--adabn']),
    dict(priority=2, group='DSA+SEA TTA', name='Conformer + DSA+SEA+TENT', run_id='20260616_sfreq100_conformer_datasetea_subjectea_tent', model='conformer', metric='tent', args=['--both', '--model', 'conformer', '--dataset_ea', '--ea', '--tent']),

    # Priority 3: extend current best method to other backbones.
    dict(priority=3, group='Best method backbone extension', name='EEGNet + DSA+SEA+SessionEA+SourceWeight', run_id='20260616_sfreq100_eegnet_datasetea_subjectea_sessionea_sourceweight_tau5', model='eegnet', metric='base', args=['--both', '--model', 'eegnet', '--dataset_ea', '--ea', '--session_ea', '--source_weighting', '--source_weight_tau', '5.0']),
    dict(priority=3, group='Best method backbone extension', name='Conformer + DSA+SEA+SessionEA+SourceWeight', run_id='20260616_sfreq100_conformer_datasetea_subjectea_sessionea_sourceweight_tau5', model='conformer', metric='base', args=['--both', '--model', 'conformer', '--dataset_ea', '--ea', '--session_ea', '--source_weighting', '--source_weight_tau', '5.0']),
]

METRIC_COLUMNS = {
    'base': ('acc', 'kappa'),
    'adabn': ('adabn_acc', 'adabn_kappa'),
    'tent': ('tent_acc', 'tent_kappa'),
    'snap': ('snap_acc', 'snap_kappa'),
    'snap_adabn': ('snap_adabn_acc', 'snap_adabn_kappa'),
}

REFERENCE = [
    dict(priority=0, group='Reference', name='CSPNet + DSA+SEA+SessionEA+SourceWeight', model='cspnet', metric='base',
         cho=RESULTS/'loso_results_20260615_sfreq100_datasetea_subjectea_sessionea_sourceweight_tau5_cross_cho2017_to_lee2019_cspnet.csv',
         lee=RESULTS/'loso_results_20260615_sfreq100_datasetea_subjectea_sessionea_sourceweight_tau5_cross_lee2019_to_cho2017_cspnet.csv'),
    dict(priority=0, group='Reference', name='CSPNet + DSA+SEA+SourceWeight', model='cspnet', metric='base',
         cho=RESULTS/'loso_results_20260612_sfreq100_datasetea_ea_sourceweight_tau5_cross_cho2017_to_lee2019_cspnet.csv',
         lee=RESULTS/'loso_results_20260612_sfreq100_datasetea_ea_sourceweight_tau5_cross_lee2019_to_cho2017_cspnet.csv'),
    dict(priority=0, group='Reference', name='EEGNet + DSA+SEA', model='eegnet', metric='base',
         cho=RESULTS/'loso_results_20260615_sfreq100_eegnet_datasetea_subjectea_cross_cho2017_to_lee2019_eegnet.csv',
         lee=RESULTS/'loso_results_20260615_sfreq100_eegnet_datasetea_subjectea_cross_lee2019_to_cho2017_eegnet.csv'),
    dict(priority=0, group='Reference', name='Conformer + DSA+SEA', model='conformer', metric='base',
         cho=RESULTS/'loso_results_20260615_sfreq100_conformer_datasetea_subjectea_cross_cho2017_to_lee2019_conformer.csv',
         lee=RESULTS/'loso_results_20260615_sfreq100_conformer_datasetea_subjectea_cross_lee2019_to_cho2017_conformer.csv'),
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
    return len(accs), (sum(accs) / len(accs) if accs else None), (sum(kaps) / len(kaps) if kaps else None)

def fmt(v):
    n, acc, kap = v
    if acc is None:
        return f'n={n}'
    return f'{acc:.2f}% / k={kap:.3f} (n={n})'

def avg(v1, v2):
    if v1[1] is None or v2[1] is None:
        return ''
    return f'{(v1[1] + v2[1]) / 2:.2f}%'

def write_summary(note=''):
    lines = [
        '# Cross-Dataset Final Matrix Queue (2026-06-16)', '',
        f'Last updated: `{datetime.now():%Y-%m-%d %H:%M}` KST', '',
        '- Input: `/home/hkim/MI_test/preprocessed_sfreq100`, `MI_N_TIMES=201`',
        '- SubjEA = `--ea`; DSA+SEA = `--dataset_ea --ea`; SessionEA = `--session_ea`.',
        '- TTA metric columns: AdaBN=`adabn_acc`, TENT=`tent_acc`, Snapshot=`snap_acc`.', '',
        '| Priority | Group | Method | Status | Cho->Lee | Lee->Cho | Avg |',
        '|---:|---|---|---|---:|---:|---:|',
    ]
    for ref in REFERENCE:
        v1 = read_metric(ref['cho'], ref['metric'])
        v2 = read_metric(ref['lee'], ref['metric'])
        status = 'completed' if v1[0] and v2[0] else 'missing'
        lines.append(f"| {ref['priority']} | {ref['group']} | {ref['name']} | {status} | {fmt(v1)} | {fmt(v2)} | {avg(v1, v2)} |")
    for exp in EXPERIMENTS:
        v1 = read_metric(csv_path(exp, 'cho2017', 'lee2019'), exp['metric'])
        v2 = read_metric(csv_path(exp, 'lee2019', 'cho2017'), exp['metric'])
        status = 'completed' if v1[0] and v2[0] else ('partial' if v1[0] or v2[0] else 'pending')
        lines.append(f"| {exp['priority']} | {exp['group']} | {exp['name']} | {status} | {fmt(v1)} | {fmt(v2)} | {avg(v1, v2)} |")
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
    env = os.environ.copy()
    env['MI_PREPROCESSED_DIR'] = str(PREP)
    env['MI_N_TIMES'] = '201'
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    log = RUNS / f"cross_dataset_final_matrix_{exp['run_id']}.log"
    cmd = [sys.executable, '-u', str(CROSS), *exp['args'], '--run_id', exp['run_id']]
    print(f"[run] P{exp['priority']} {exp['name']} -> {log}", flush=True)
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
