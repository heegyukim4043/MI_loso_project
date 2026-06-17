from __future__ import annotations

import csv
import shutil
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path('/home/hkim/MI_test')
PROJECT = ROOT / 'MI_loso_project'
RESULTS = ROOT / 'results'
OUT = ROOT / 'forgit' / 'loso'
RESULT_OUT = OUT / 'results'
MODEL_OUT = OUT / 'models'
RESULT_OUT.mkdir(parents=True, exist_ok=True)
MODEL_OUT.mkdir(parents=True, exist_ok=True)

MODEL_FILES = [
    'train_loso.py', 'loso_csp_lda.py', 'mrfbcsp_loso.py', 'cspnet.py', 'cspnet_contrastive.py',
    'cspnet_dann.py', 'eegnet.py', 'conformer.py', 'eeg_ea.py', 'eeg_augment.py', 'eeg_style_aug.py',
    'adabn.py', 'tent.py', 'sam.py', 'statistical_tests.py', 'refresh_forgit_loso.py',
]
for name in MODEL_FILES:
    src = PROJECT / name
    if src.exists():
        shutil.copy2(src, MODEL_OUT / name)

# New/long-running LOSO jobs that produce rich train_loso CSVs in /results.
RICH_RESULT_MAPPINGS = [
    {
        'source': 'loso_results_conformer_ea_snapshot_conformer.csv',
        'dest': 'ea_snapshot_conformer.csv',
        'metric': 'snap_acc',
        'method': 'ea_snapshot_conformer',
        'description': 'EA + Conformer + Snapshot(x6) ensemble',
    },
]

for spec in RICH_RESULT_MAPPINGS:
    src = RESULTS / spec['source']
    if not src.exists():
        continue
    with src.open(newline='') as f:
        rows = list(csv.DictReader(f))
    out_rows = []
    for r in rows:
        val = r.get(spec['metric']) or r.get('acc')
        if val in (None, ''):
            continue
        try:
            acc = float(val)
        except ValueError:
            continue
        if acc <= 1.0:
            acc *= 100.0
        out_rows.append({'dataset': r.get('dataset', ''), 'subject': r.get('subject', ''), 'acc': round(acc, 4)})
    if out_rows:
        with (RESULT_OUT / spec['dest']).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['dataset', 'subject', 'acc'])
            w.writeheader(); w.writerows(out_rows)

old_desc: dict[str, str] = {}
summary_path = RESULT_OUT / 'summary_all_methods.csv'
if summary_path.exists():
    with summary_path.open(newline='') as f:
        for r in csv.DictReader(f):
            old_desc.setdefault(r.get('method', ''), r.get('description', ''))
for spec in RICH_RESULT_MAPPINGS:
    old_desc[spec['method']] = spec['description']

DEFAULT_DESCRIPTIONS = {
    'csp_lda': 'CSP-LDA baseline, no EA',
    'ea_csp_lda': 'EA + CSP-LDA',
    'cspnet_noea': 'CSPNet baseline, no EA',
    'ea_cspnet': 'EA + CSPNet',
    'ea_adabn_cspnet': 'EA + CSPNet + AdaBN',
    'ea_tent_cspnet': 'EA + CSPNet + TENT',
    'ea_snapshot_adabn_cspnet': 'EA + CSPNet + Snapshot(x6) + AdaBN',
    'eegnet_noea': 'EEGNet baseline, no EA',
    'ea_eegnet': 'EA + EEGNet',
    'ea_adabn_eegnet': 'EA + EEGNet + AdaBN',
    'ea_tent_eegnet': 'EA + EEGNet + TENT',
    'ea_snapshot_eegnet': 'EA + EEGNet + Snapshot(x6) ensemble',
    'conformer_noea': 'Conformer baseline, no EA',
    'ea_conformer': 'EA + Conformer',
    'ea_adabn_conformer': 'EA + Conformer + AdaBN',
    'ea_tent_conformer': 'EA + Conformer + TENT',
    'ea_snapshot_conformer': 'EA + Conformer + Snapshot(x6) ensemble',
    'dann_cspnet': 'DANN + CSPNet',
}

def method_from_file(path: Path) -> str:
    return path.stem

def clean_acc(v: str) -> float | None:
    if v in ('', None):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x * 100.0 if x <= 1.0 else x

summary_rows = []
for csv_path in sorted(RESULT_OUT.glob('*.csv')):
    if csv_path.name == 'summary_all_methods.csv':
        continue
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'dataset' not in rows[0]:
        continue
    method = method_from_file(csv_path)
    desc = old_desc.get(method) or DEFAULT_DESCRIPTIONS.get(method) or method
    by_ds: dict[str, list[float]] = {}
    for r in rows:
        acc = clean_acc(r.get('acc', ''))
        ds = r.get('dataset', '')
        if acc is None or not ds:
            continue
        by_ds.setdefault(ds, []).append(acc)
    for ds, vals in sorted(by_ds.items()):
        vals_sorted = sorted(vals)
        summary_rows.append({
            'method': method,
            'description': desc,
            'dataset': ds,
            'n': len(vals),
            'mean': round(mean(vals), 2),
            'std': round(pstdev(vals), 2) if len(vals) > 1 else 0.0,
            'median': round(median(vals), 2),
            'min': round(min(vals), 2),
            'max': round(max(vals), 2),
        })

with summary_path.open('w', newline='') as f:
    fields = ['method', 'description', 'dataset', 'n', 'mean', 'std', 'median', 'min', 'max']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(summary_rows)
print(f'forgit loso refreshed: {OUT}')
result_file_count = len([x for x in RESULT_OUT.glob('*.csv') if x.name != 'summary_all_methods.csv'])
print(f'summary_rows={len(summary_rows)} result_files={result_file_count}')
