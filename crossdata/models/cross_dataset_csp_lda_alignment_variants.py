"""Cross-dataset CSP-LDA alignment variants on raw-unified data.

Variants:
- zscore_before_ea: channel z-score independently per dataset before DatasetEA -> SubjectEA
- zscore_between_ea: DatasetEA -> train-stat z-score -> SubjectEA
- weighted_dataset_ea: source DatasetEA reference weighted by target-similar source subjects
- riemannian_dataset_subject: Riemannian mean whitening for dataset and subject alignment
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cross_dataset import RESULTS_DIR, STANDARD_MI_CHANNELS
from mrfbcsp_loso import ManualCSP
from eeg_ea import euclidean_align, apply_ea_loso

try:
    from pyriemann.utils.mean import mean_riemann
except Exception:  # pragma: no cover
    mean_riemann = None

DEFAULT_PREP = (ROOT / ".." / "preprocessed_raw_unified").resolve()


def load_npz(name, preprocessed_dir=DEFAULT_PREP):
    d = np.load(Path(preprocessed_dir) / f"{name}.npz", allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int64), d["subjects"].astype(np.int64), list(d["ch_names"])


def channel_indices(ch_names):
    missing = [ch for ch in STANDARD_MI_CHANNELS if ch not in ch_names]
    common = [ch for ch in STANDARD_MI_CHANNELS if ch in ch_names]
    idx = [ch_names.index(ch) for ch in common]
    return common, idx, missing


def load_pair(train_name, test_name, preprocessed_dir=DEFAULT_PREP):
    Xtr, ytr, strn, chtr = load_npz(train_name, preprocessed_dir)
    Xte, yte, ste, chte = load_npz(test_name, preprocessed_dir)
    common = [ch for ch in STANDARD_MI_CHANNELS if ch in chtr and ch in chte]
    train_idx = [chtr.index(ch) for ch in common]
    test_idx = [chte.index(ch) for ch in common]
    return Xtr[:, train_idx, :], ytr, strn, Xte[:, test_idx, :], yte, ste, common


def covariances(X, eps=1e-6):
    covs = []
    for x in X.astype(np.float64):
        z = x - x.mean(axis=1, keepdims=True)
        c = z @ z.T / max(z.shape[1] - 1, 1)
        c = (c + c.T) / 2.0
        c += eps * np.eye(c.shape[0])
        covs.append(c)
    return np.asarray(covs)


def mean_cov(X, eps=1e-6):
    c = np.mean(covariances(X, eps=eps), axis=0)
    c = (c + c.T) / 2.0
    c += eps * np.eye(c.shape[0])
    return c



def logeuclid_mean(covs, eps=1e-8):
    logs = []
    for C in covs:
        vals, vecs = np.linalg.eigh((C + C.T) / 2.0)
        vals = np.maximum(vals, eps)
        logs.append(vecs @ np.diag(np.log(vals)) @ vecs.T)
    L = np.mean(logs, axis=0)
    vals, vecs = np.linalg.eigh((L + L.T) / 2.0)
    return vecs @ np.diag(np.exp(vals)) @ vecs.T

def invsqrt(C, eps=1e-8):
    vals, vecs = np.linalg.eigh((C + C.T) / 2.0)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def apply_transform(X, A):
    return np.einsum("cd,ndt->nct", A, X).astype(np.float32)


def zscore_dataset(X):
    mu = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return ((X - mu) / std).astype(np.float32)


def zscore_train_test(X_train, X_test):
    mu = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return ((X_train - mu) / std).astype(np.float32), ((X_test - mu) / std).astype(np.float32)


def dataset_ea(X):
    return euclidean_align(X)


def riemannian_align(X):
    # Fast log-Euclidean Riemannian mean. This avoids the expensive iterative
    # affine-invariant mean over every trial while keeping the SPD geometry.
    R = logeuclid_mean(covariances(X))
    return apply_transform(X, invsqrt(R))


def riemannian_dataset_align(X, subjects):
    """Dataset-level Riemannian whitening from per-subject mean covariances.

    Using all trial covariances is unnecessarily slow for 10k+ trials. The
    subject-level means preserve the intended dataset reference while keeping
    the run cheap and avoiding subject-count imbalance.
    """
    subj_covs = [mean_cov(X[subjects == s]) for s in np.unique(subjects)]
    R = logeuclid_mean(np.asarray(subj_covs))
    return apply_transform(X, invsqrt(R))


def apply_riemannian_loso(X, subjects):
    out = X.copy()
    for s in np.unique(subjects):
        mask = subjects == s
        # 200 trial covariances per subject is small enough and keeps the
        # subject-level alignment faithful to the Riemannian objective.
        out[mask] = riemannian_align(X[mask])
    return out


def log_spd(C):
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-8)
    return vecs @ np.diag(np.log(vals)) @ vecs.T


def weighted_reference_cov(X_train, subj_train, X_target, tau=0.1):
    target_log = log_spd(mean_cov(X_target))
    subj_ids = np.unique(subj_train)
    covs = []
    dists = []
    counts = []
    for s in subj_ids:
        mask = subj_train == s
        c = mean_cov(X_train[mask])
        covs.append(c)
        counts.append(int(mask.sum()))
        dists.append(np.linalg.norm(log_spd(c) - target_log, ord="fro"))
    dists = np.asarray(dists)
    logits = -dists / max(tau, 1e-8)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    R = np.tensordot(weights, np.asarray(covs), axes=(0, 0))
    R = (R + R.T) / 2.0 + 1e-6 * np.eye(R.shape[0])
    return R, dict(min_dist=float(dists.min()), mean_dist=float(dists.mean()), max_dist=float(dists.max()), max_weight=float(weights.max()))


def features_csp_lda(X_train, y_train, X_test, n_csp=8):
    csp = ManualCSP(n_components=n_csp)
    csp.fit(X_train, y_train)
    F_train = csp.transform(X_train)
    F_test = csp.transform(X_test)
    scaler = StandardScaler()
    F_train = scaler.fit_transform(F_train)
    F_test = scaler.transform(F_test)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(F_train, y_train)
    return lda, F_test


def apply_variant(variant, X_train, subj_train, X_test, subj_test, tau):
    meta = {}
    if variant == "zscore_before_ea":
        X_train = zscore_dataset(X_train)
        X_test = zscore_dataset(X_test)
        X_train = dataset_ea(X_train)
        X_test = dataset_ea(X_test)
        X_train = apply_ea_loso(X_train, subj_train)
        X_test = apply_ea_loso(X_test, subj_test)
    elif variant == "zscore_between_ea":
        X_train = dataset_ea(X_train)
        X_test = dataset_ea(X_test)
        X_train, X_test = zscore_train_test(X_train, X_test)
        X_train = apply_ea_loso(X_train, subj_train)
        X_test = apply_ea_loso(X_test, subj_test)
    elif variant == "weighted_dataset_ea":
        R, meta = weighted_reference_cov(X_train, subj_train, X_test, tau=tau)
        X_train = apply_transform(X_train, invsqrt(R))
        # Keep target dataset aligned with its own label-free dataset covariance.
        X_test = dataset_ea(X_test)
        X_train = apply_ea_loso(X_train, subj_train)
        X_test = apply_ea_loso(X_test, subj_test)
    elif variant == "riemannian_dataset_subject":
        X_train = riemannian_dataset_align(X_train, subj_train)
        X_test = riemannian_dataset_align(X_test, subj_test)
        X_train = apply_riemannian_loso(X_train, subj_train)
        X_test = apply_riemannian_loso(X_test, subj_test)
    else:
        raise ValueError(variant)
    # Final train-stat normalization follows the current CSP-LDA baseline.
    X_train, X_test = zscore_train_test(X_train, X_test)
    return X_train, X_test, meta


def run_one(train_name, test_name, variant, run_id, tau=0.1, preprocessed_dir=DEFAULT_PREP, n_csp=8):
    X_train, y_train, subj_train, X_test, y_test, subj_test, common = load_pair(train_name, test_name, preprocessed_dir)
    print("=" * 70)
    print(f"{variant}: {train_name} -> {test_name}  Xtr={X_train.shape} Xte={X_test.shape} channels={len(common)}")
    X_train, X_test, meta = apply_variant(variant, X_train, subj_train, X_test, subj_test, tau)
    print(f"meta={meta}")

    out_csv = Path(RESULTS_DIR) / f"loso_results_{run_id}_{variant}_cross_{train_name}_to_{test_name}_csp_lda.csv"
    fields = ["train_ds", "test_ds", "variant", "subject", "n_test", "acc", "bac", "kappa"]
    rows = []
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    for s in np.unique(subj_test):
        mask = subj_test == s
        lda, F_test = features_csp_lda(X_train, y_train, X_test[mask], n_csp=n_csp)
        pred = lda.predict(F_test)
        yt = y_test[mask]
        row = {
            "train_ds": train_name,
            "test_ds": test_name,
            "variant": variant,
            "subject": int(s),
            "n_test": int(mask.sum()),
            "acc": round(accuracy_score(yt, pred) * 100, 2),
            "bac": round(balanced_accuracy_score(yt, pred) * 100, 2),
            "kappa": round(cohen_kappa_score(yt, pred), 3),
        }
        rows.append(row)
        with out_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
    acc = float(np.mean([r["acc"] for r in rows]))
    kap = float(np.mean([r["kappa"] for r in rows]))
    print(f"{train_name}->{test_name}: acc={acc:.2f}% k={kap:.3f} saved={out_csv}")
    return acc, kap, out_csv


def append_progress(lines):
    progress = ROOT.parent / "progress.md"
    if progress.exists():
        progress.write_text(progress.read_text() + "\n" + "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["zscore_before_ea", "zscore_between_ea", "weighted_dataset_ea", "riemannian_dataset_subject"])
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_alignment_variants")
    summary_rows = []
    for variant in args.variants:
        r1 = run_one("cho2017", "lee2019", variant, run_id, tau=args.tau)
        r2 = run_one("lee2019", "cho2017", variant, run_id, tau=args.tau)
        summary_rows.append((variant, r1, r2))

    summary_path = Path(RESULTS_DIR) / f"cross_dataset_alignment_variants_summary_{run_id}.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "cho_to_lee_acc", "cho_to_lee_kappa", "lee_to_cho_acc", "lee_to_cho_kappa", "cho_to_lee_csv", "lee_to_cho_csv"])
        for variant, r1, r2 in summary_rows:
            w.writerow([variant, f"{r1[0]:.2f}", f"{r1[1]:.3f}", f"{r2[0]:.2f}", f"{r2[1]:.3f}", r1[2], r2[2]])
    print(f"summary={summary_path}")

    lines = ["", f"## Cross-Dataset Alignment Variant Sweep ({datetime.now():%Y-%m-%d %H:%M})", "", "- Status: `completed`", f"- Summary CSV: `{summary_path}`", "", "| Variant | Cho2017 -> Lee2019 | Lee2019 -> Cho2017 |", "|---|---:|---:|"]
    for variant, r1, r2 in summary_rows:
        lines.append(f"| {variant} | {r1[0]:.2f}% / k={r1[1]:.3f} | {r2[0]:.2f}% / k={r2[1]:.3f} |")
    append_progress(lines)


if __name__ == "__main__":
    main()
