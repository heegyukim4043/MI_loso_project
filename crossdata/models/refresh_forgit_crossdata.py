from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/hkim/MI_test')
OUT = ROOT / 'forgit' / 'crossdata'
RESULTS = ROOT / 'results'
PROJECT = ROOT / 'MI_loso_project'

for d in [OUT/'models', OUT/'models'/'support', OUT/'results'/'raw_csv', OUT/'results'/'summaries', OUT/'results'/'aggregated', OUT/'reports', OUT/'progress']:
    d.mkdir(parents=True, exist_ok=True)

model_files = [
    'cross_dataset.py', 'cross_dataset_csp_lda.py', 'cross_dataset_csp_lda_alignment_variants.py',
    'cross_dataset_csp_lda_extra_variants.py', 'cross_dataset_csp_lda_session_coral_mmd.py',
    'cross_dataset_kmm_tradaboost.py', 'generate_cross_dataset_analysis_artifacts.py',
    'manage_cross_dataset_arch_methods_20260615.py', 'manage_cross_dataset_final_matrix_20260616.py',
    'manage_cross_dataset_sfreq100_queue.py', 'manage_cross_dataset_sourceweight_backbones_20260616.py',
    'manage_cross_dataset_priority_20260617.py',
    'refresh_forgit_crossdata.py', 'cspnet.py', 'cspnet_contrastive.py', 'eegnet.py', 'conformer.py',
    'eeg_ea.py', 'adabn.py', 'tent.py', 'dsbn.py', 'mrfbcsp_loso.py',
]
for name in model_files:
    src = PROJECT / name
    if src.exists():
        shutil.copy2(src, OUT/'models'/name)
for name in ['run_cross_dataset_eval.py', 'generate_subject_performance_csv.py', 'generate_subject_performance_tables.py', 'update_summary.py']:
    src = PROJECT / name
    if src.exists():
        shutil.copy2(src, OUT/'models'/'support'/name)

raw_csv_files = []
for p in RESULTS.glob('*.csv'):
    lname = p.name.lower()
    if 'cross' in lname or 'session_coral_mmd_summary' in lname:
        raw_csv_files.append(p)
raw_csv_files = sorted(set(raw_csv_files), key=lambda x: x.name)
for p in raw_csv_files:
    shutil.copy2(p, OUT/'results'/'raw_csv'/p.name)
    if p.name.startswith('cross_dataset_'):
        shutil.copy2(p, OUT/'results'/'summaries'/p.name)

report_names = [
    'progress.md', 'subject_performance_cross_dataset.md',
    'cross_dataset_classification_performance_table.csv', 'cross_dataset_classification_performance_table.md',
    'cross_dataset_classification_performance_wide.csv', 'cross_dataset_classification_summary.md',
    'cross_dataset_sfreq100_results_summary.md', 'cross_dataset_arch_methods_20260615.md',
    'cross_dataset_sourceweight_backbones_20260616.md', 'cross_dataset_final_matrix_20260616.md',
    'cross_dataset_priority_20260617.md',
    'loso_methods_applied_cross_dataset_performance.md', 'preprocessing_methods_and_model_parameters.md',
    'environments.md',
]
for name in report_names:
    src = ROOT / name
    if src.exists():
        target_dir = OUT/'progress' if name == 'progress.md' else OUT/'reports'
        shutil.copy2(src, target_dir/name)

METRIC_SPECS = [
    ('base', 'acc', 'bac', 'kappa', 'precision_macro', 'f1_macro'),
    ('adabn', 'adabn_acc', 'adabn_bac', 'adabn_kappa', '', ''),
    ('tent', 'tent_acc', 'tent_bac', 'tent_kappa', '', ''),
    ('snapshot', 'snap_acc', 'snap_bac', 'snap_kappa', '', ''),
    ('snapshot_adabn', 'snap_adabn_acc', 'snap_adabn_bac', 'snap_adabn_kappa', '', ''),
]

def clean_float(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except Exception:
        return None

def infer_from_filename(name):
    stem = name.removesuffix('.csv')
    if '_cross_' in stem:
        method = stem.split('_cross_', 1)[0].removeprefix('loso_results_')
        tail = stem.split('_cross_', 1)[1]
        if '_to_' in tail:
            train, rest = tail.split('_to_', 1)
            pieces = rest.split('_')
            test = pieces[0]
            model = '_'.join(pieces[1:])
            return method, train, test, model
    return stem.removeprefix('loso_results_'), '', '', ''

subject_rows = []
summary_rows = []
manifest = []
for p in raw_csv_files:
    method_id, file_train, file_test, file_model = infer_from_filename(p.name)
    try:
        rows = list(csv.DictReader(p.open(newline='')))
    except Exception:
        rows = []
    has_subject = bool(rows) and 'subject' in rows[0] and ('acc' in rows[0] or 'kappa' in rows[0])
    manifest.append({'file': str((OUT/'results'/'raw_csv'/p.name).relative_to(OUT)), 'rows': len(rows), 'subject_level': has_subject})
    if not has_subject:
        continue
    normalized = []
    for row in rows:
        base = {
            'source_file': p.name,
            'method_id': method_id,
            'train_dataset': row.get('train_ds') or file_train,
            'test_dataset': row.get('test_ds') or file_test,
            'model': row.get('model') or row.get('variant') or file_model,
            'variant': row.get('variant', ''),
            'subject': row.get('subject', ''),
            'n_test': row.get('n_test', ''),
        }
        for col in ['acc','precision_macro','f1_macro','bac','kappa','adabn_acc','adabn_bac','adabn_kappa','tent_acc','tent_bac','tent_kappa','snap_acc','snap_bac','snap_kappa','snap_adabn_acc','snap_adabn_bac','snap_adabn_kappa']:
            base[col] = row.get(col, '')
        subject_rows.append(base)
        normalized.append(base)
    for eval_type, acc_col, bac_col, kap_col, prec_col, f1_col in METRIC_SPECS:
        vals = [r for r in normalized if clean_float(r.get(acc_col)) is not None]
        if not vals:
            continue
        def mean_col(col):
            xs = [clean_float(r.get(col)) for r in vals]
            xs = [x for x in xs if x is not None]
            return round(sum(xs)/len(xs), 4) if xs else ''
        summary_rows.append({
            'source_file': p.name, 'method_id': method_id, 'eval_type': eval_type,
            'train_dataset': vals[0].get('train_dataset', ''), 'test_dataset': vals[0].get('test_dataset', ''),
            'model': vals[0].get('model', ''), 'n_subjects': len(vals),
            'mean_acc': mean_col(acc_col), 'mean_precision_macro': mean_col(prec_col) if prec_col else '',
            'mean_f1_macro': mean_col(f1_col) if f1_col else '', 'mean_bac': mean_col(bac_col),
            'mean_kappa': mean_col(kap_col),
        })

subject_fields = ['source_file','method_id','train_dataset','test_dataset','model','variant','subject','n_test','acc','precision_macro','f1_macro','bac','kappa','adabn_acc','adabn_bac','adabn_kappa','tent_acc','tent_bac','tent_kappa','snap_acc','snap_bac','snap_kappa','snap_adabn_acc','snap_adabn_bac','snap_adabn_kappa']
with (OUT/'results'/'aggregated'/'crossdataset_subject_level_all.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=subject_fields)
    w.writeheader(); w.writerows(subject_rows)
summary_fields = ['source_file','method_id','eval_type','train_dataset','test_dataset','model','n_subjects','mean_acc','mean_precision_macro','mean_f1_macro','mean_bac','mean_kappa']
summary_rows.sort(key=lambda r: (r['method_id'], r['train_dataset'], r['test_dataset'], r['eval_type']))
with (OUT/'results'/'aggregated'/'crossdataset_method_summary.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=summary_fields)
    w.writeheader(); w.writerows(summary_rows)

with (OUT/'manifest.json').open('w') as f:
    json.dump({'created_at': datetime.now().isoformat(timespec='seconds'), 'raw_cross_csv_count': len(raw_csv_files), 'subject_level_rows': len(subject_rows), 'method_summary_rows': len(summary_rows), 'files': manifest}, f, indent=2, ensure_ascii=False)

readme_lines = [
    '# Cross-Dataset Package', '',
    'Prepared for upload to `https://github.com/heegyukim4043/MI_loso_project`.', '',
    f'Refreshed: `{datetime.now():%Y-%m-%d %H:%M}` KST', '',
    '## Directory Layout', '',
    '- `models/`: cross-dataset training/evaluation scripts and model definitions.',
    '- `results/raw_csv/`: all cross-dataset CSV outputs copied from `/home/hkim/MI_test/results`.',
    '- `results/summaries/`: original cross-dataset summary CSV files.',
    '- `results/aggregated/crossdataset_subject_level_all.csv`: unified subject-level table.',
    '- `results/aggregated/crossdataset_method_summary.csv`: per-method/per-direction mean metrics.',
    '- `reports/`: curated Markdown/CSV summaries and environment/method documents.',
    '- `progress/progress.md`: copied project progress log.',
    '- `manifest.json`: file inventory and aggregate row counts.', '',
    '## Latest Reports', '',
    '- `reports/cross_dataset_final_matrix_20260616.md`: final matrix queue results.',
    '- `reports/cross_dataset_sourceweight_backbones_20260616.md`: backbone + SourceWeight comparison.',
    '- `reports/cross_dataset_arch_methods_20260615.md`: architecture method cells.',
    '- `reports/subject_performance_cross_dataset.md`: subject-level cross-dataset report.', '',
]
(OUT/'README.md').write_text('\n'.join(readme_lines))
print(f'forgit crossdata refreshed: {OUT}')
print(f'raw_csv_count={len(raw_csv_files)} subject_rows={len(subject_rows)} summary_rows={len(summary_rows)}')
