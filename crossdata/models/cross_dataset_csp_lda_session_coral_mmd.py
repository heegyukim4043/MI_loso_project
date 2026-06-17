"""SessionEA, feature-CORAL, and MMD-style resampling for cross-dataset CSP-LDA.

Baseline preprocessing inside this script uses the current main alignment:
DatasetEA -> SubjectEA -> train-stat z-score on raw-unified Cho2017/Lee2019.

Variants:
- session_ea: add Lee2019 session-level EA after SubjectEA. Lee sessions are
  approximated by the saved raw-unified order: 100 trials/session per subject.
- feature_coral: fit CSP on aligned source, then align source CSP feature
  covariance to each unlabeled target subject before LDA.
- mmd_resample: fit CSP, then resample source CSP features according to RBF
  similarity to each target subject's unlabeled feature distribution.
- session_ea_feature_coral: combination of SessionEA and feature-CORAL.
- session_ea_mmd_resample: SessionEA followed by MMD-style resampling.
- session_ea_feature_coral_mmd_resample: SessionEA + CORAL + MMD resampling.
- source_select_k<N>: target-subject-specific source subject selection by covariance distance.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
)
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cross_dataset import STANDARD_MI_CHANNELS, RESULTS_DIR
from eeg_ea import euclidean_align, apply_ea_loso
from mrfbcsp_loso import ManualCSP

RAW_UNIFIED = Path(os.environ.get('MI_CSP_LDA_PREPROCESSED_DIR', str((ROOT / '..' / 'preprocessed_raw_unified').resolve()))).resolve()
SEED = 2026


def load_dataset(name):
    path = RAW_UNIFIED / f'{name}.npz'
    print(f'  Loading {path}')
    d = np.load(path, allow_pickle=True)
    return d['X'].astype(np.float32), d['y'].astype(np.int64), d['subjects'].astype(np.int64), [str(c) for c in d['ch_names']]


def load_pair(src, tgt):
    Xs, ys, ss, chs = load_dataset(src)
    Xt, yt, st, cht = load_dataset(tgt)
    common = [c for c in STANDARD_MI_CHANNELS if c in chs and c in cht]
    Xs = Xs[:, [chs.index(c) for c in common], :]
    Xt = Xt[:, [cht.index(c) for c in common], :]
    return Xs, ys, ss, Xt, yt, st, common


def zscore_train_test(Xs, Xt):
    mu = Xs.mean(axis=(0, 2), keepdims=True)
    sd = Xs.std(axis=(0, 2), keepdims=True) + 1e-8
    return ((Xs - mu) / sd).astype(np.float32), ((Xt - mu) / sd).astype(np.float32)


def base_align(Xs, ss, Xt, st):
    Xs = euclidean_align(Xs)
    Xt = euclidean_align(Xt)
    Xs = apply_ea_loso(Xs, ss)
    Xt = apply_ea_loso(Xt, st)
    return Xs, Xt


def apply_lee_session_ea(X, subjects, dataset_name):
    if dataset_name != 'lee2019':
        return X
    out = X.copy()
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        # Lee2019 raw-unified was built session-wise: two 100-trial sessions.
        # Use order-preserving split to avoid needing a separate session label file.
        splits = np.array_split(idx, 2)
        for part in splits:
            if len(part) >= 4:
                out[part] = euclidean_align(X[part])
    return out



def mean_cov(X, eps=1e-6):
    C = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for trial in X.astype(np.float64):
        z = trial - trial.mean(axis=1, keepdims=True)
        c = z @ z.T / max(z.shape[1] - 1, 1)
        tr = np.trace(c)
        if tr > 1e-12:
            c = c / tr
        C += c
    C /= max(len(X), 1)
    C = (C + C.T) / 2.0 + eps * np.eye(C.shape[0])
    return C


def log_spd(C, eps=1e-8):
    vals, vecs = np.linalg.eigh((C + C.T) / 2.0)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(np.log(vals)) @ vecs.T


def select_source_subjects(Xs, ys, ss, Xt_subject, k):
    target_log = log_spd(mean_cov(Xt_subject))
    distances = []
    for subj in np.unique(ss):
        subj_log = log_spd(mean_cov(Xs[ss == subj]))
        distances.append((float(np.linalg.norm(subj_log - target_log, ord='fro')), int(subj)))
    distances.sort()
    selected = {subj for _, subj in distances[:min(k, len(distances))]}
    mask = np.array([subj in selected for subj in ss])
    return Xs[mask], ys[mask], selected


def source_select_k_from_variant(variant):
    if not variant.startswith('source_select_k'):
        return None
    return int(variant.rsplit('k', 1)[1])

def fit_csp_features(Xs, ys, Xt, n_csp=8):
    csp = ManualCSP(n_components=n_csp)
    csp.fit(Xs, ys)
    Fs = csp.transform(Xs)
    Ft = csp.transform(Xt)
    scaler = StandardScaler()
    Fs = scaler.fit_transform(Fs)
    Ft = scaler.transform(Ft)
    return Fs, Ft


def cov_inv_sqrt(C, eps=1e-6):
    C = (C + C.T) / 2 + eps * np.eye(C.shape[0])
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def cov_sqrt(C, eps=1e-6):
    C = (C + C.T) / 2 + eps * np.eye(C.shape[0])
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals ** 0.5) @ vecs.T


def coral_source_to_target(Fs, Ft):
    ms = Fs.mean(axis=0, keepdims=True)
    mt = Ft.mean(axis=0, keepdims=True)
    Cs = np.cov((Fs - ms).T)
    Ct = np.cov((Ft - mt).T)
    A = cov_inv_sqrt(Cs) @ cov_sqrt(Ct)
    return (Fs - ms) @ A + mt


def mmd_resample_source(Fs, Ft, ys, rng):
    # RBF similarity to target feature cloud approximates target-density emphasis.
    d2 = pairwise_distances(Fs, Ft, metric='sqeuclidean')
    # Median heuristic over a bounded subset to keep scale stable.
    med = np.median(d2[:, :min(200, d2.shape[1])])
    gamma = 1.0 / max(med, 1e-6)
    sim = np.exp(-gamma * d2).mean(axis=1)
    sim = sim + 1e-8
    # Preserve class balance by normalizing weights within each class.
    weights = np.zeros_like(sim)
    for cls in np.unique(ys):
        m = ys == cls
        weights[m] = sim[m] / sim[m].sum() / len(np.unique(ys))
    weights = weights / weights.sum()
    idx = rng.choice(len(Fs), size=len(Fs), replace=True, p=weights)
    return Fs[idx], ys[idx]


def evaluate_subjectwise(src, tgt, variant, run_id, n_csp=8):
    Xs, ys, ss, Xt, yt, st, common = load_pair(src, tgt)
    print('=' * 70)
    print(f'{variant}: {src}->{tgt} Xs={Xs.shape} Xt={Xt.shape} ch={len(common)}')
    Xs, Xt = base_align(Xs, ss, Xt, st)
    if 'session_ea' in variant:
        Xs = apply_lee_session_ea(Xs, ss, src)
        Xt = apply_lee_session_ea(Xt, st, tgt)
    Xs, Xt = zscore_train_test(Xs, Xt)

    out_csv = Path(RESULTS_DIR) / f'loso_results_{run_id}_{variant}_cross_{src}_to_{tgt}_csp_lda.csv'
    fields = [
        'train_ds',
        'test_ds',
        'variant',
        'subject',
        'n_test',
        'acc',
        'precision_macro',
        'f1_macro',
        'bac',
        'kappa',
    ]
    with out_csv.open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    rng = np.random.default_rng(SEED)
    rows = []
    for subject in np.unique(st):
        mask = st == subject
        X_train_subject = Xs
        y_train_subject = ys
        k_select = source_select_k_from_variant(variant)
        if k_select is not None:
            X_train_subject, y_train_subject, _selected = select_source_subjects(
                Xs, ys, ss, Xt[mask], k_select
            )
        Fs, Ft = fit_csp_features(X_train_subject, y_train_subject, Xt[mask], n_csp=n_csp)
        y_train = y_train_subject
        if 'feature_coral' in variant:
            Fs = coral_source_to_target(Fs, Ft)
        if 'mmd_resample' in variant:
            Fs, y_train = mmd_resample_source(Fs, Ft, y_train, rng)
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(Fs, y_train)
        pred = clf.predict(Ft)
        y_true = yt[mask]
        row = {
            'train_ds': src,
            'test_ds': tgt,
            'variant': variant,
            'subject': int(subject),
            'n_test': int(mask.sum()),
            'acc': round(accuracy_score(y_true, pred) * 100, 2),
            'precision_macro': round(precision_score(y_true, pred, average='macro', zero_division=0) * 100, 2),
            'f1_macro': round(f1_score(y_true, pred, average='macro', zero_division=0) * 100, 2),
            'bac': round(balanced_accuracy_score(y_true, pred) * 100, 2),
            'kappa': round(cohen_kappa_score(y_true, pred), 3),
        }
        rows.append(row)
        with out_csv.open('a', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
    acc = float(np.mean([r['acc'] for r in rows]))
    kap = float(np.mean([r['kappa'] for r in rows]))
    print(f'{src}->{tgt}: acc={acc:.2f}% k={kap:.3f} saved={out_csv}')
    return acc, kap, out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=['session_ea', 'feature_coral', 'mmd_resample', 'session_ea_feature_coral', 'session_ea_mmd_resample', 'session_ea_feature_coral_mmd_resample', 'source_select_k10', 'source_select_k20', 'source_select_k30', 'source_select_k40'])
    parser.add_argument('--run_id', default=None)
    args = parser.parse_args()
    print(f'Input dir: {RAW_UNIFIED}')
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_session_coral_mmd')
    summary = []
    for variant in args.variants:
        r1 = evaluate_subjectwise('cho2017', 'lee2019', variant, run_id)
        r2 = evaluate_subjectwise('lee2019', 'cho2017', variant, run_id)
        summary.append((variant, r1, r2))

    out = Path(RESULTS_DIR) / f'cross_dataset_session_coral_mmd_summary_{run_id}.csv'
    with out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'cho_to_lee_acc', 'cho_to_lee_kappa', 'lee_to_cho_acc', 'lee_to_cho_kappa', 'cho_to_lee_csv', 'lee_to_cho_csv'])
        for variant, r1, r2 in summary:
            w.writerow([variant, f'{r1[0]:.2f}', f'{r1[1]:.3f}', f'{r2[0]:.2f}', f'{r2[1]:.3f}', r1[2], r2[2]])
    print(f'summary={out}')

    progress = ROOT.parent / 'progress.md'
    lines = ['', f'## Cross-Dataset SessionEA/CORAL/MMD Sweep ({datetime.now():%Y-%m-%d %H:%M})', '', '- Status: `completed`', f'- Summary CSV: `{out}`', '', '| Variant | Cho2017 -> Lee2019 | Lee2019 -> Cho2017 |', '|---|---:|---:|']
    for variant, r1, r2 in summary:
        lines.append(f'| {variant} | {r1[0]:.2f}% / k={r1[1]:.3f} | {r2[0]:.2f}% / k={r2[1]:.3f} |')
    progress.write_text(progress.read_text() + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
