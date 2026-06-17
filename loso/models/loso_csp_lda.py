"""
LOSO CSP-LDA baseline.

Implements two classical baselines for MI-EEG:
  - CSP-LDA      : no alignment
  - EA-CSP-LDA   : per-subject Euclidean Alignment before CSP

Usage:
    python loso_csp_lda.py --dataset cho2017
    python loso_csp_lda.py --dataset cho2017 --ea
    python loso_csp_lda.py --dataset both --ea
    python loso_csp_lda.py --dataset both        # runs both CSP-LDA and EA-CSP-LDA
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR    = ROOT.parent / "preprocessed"
RESULTS_DIR = ROOT.parent / "results"

from eeg_ea import euclidean_align, apply_ea_loso   # noqa: E402
from mrfbcsp_loso import ManualCSP                  # noqa: E402


def load_data(dataset_name: str):
    path = DATA_DIR / f"{dataset_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32),
            d["y"].astype(np.int64),
            d["subjects"].astype(np.int64),
            float(d["sfreq"]))


def run_loso_csp_lda(dataset_name: str, use_ea: bool = False,
                     n_csp: int = 8, resume: bool = False):
    tag    = "ea_csp_lda" if use_ea else "csp_lda"
    csv_path = RESULTS_DIR / f"loso_results_{tag}_{dataset_name}.csv"
    fields = ["dataset", "subject", "n_train", "n_test",
              "acc", "bac", "kappa", "time_min"]

    done_subjects: set[int] = set()
    if resume and csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("dataset") == dataset_name:
                    try:
                        done_subjects.add(int(row["subject"]))
                    except (KeyError, ValueError):
                        pass
        print(f"[{tag}/{dataset_name}] resume: {len(done_subjects)} already done")

    X, y, subjects, sfreq = load_data(dataset_name)

    if use_ea:
        X = apply_ea_loso(X, subjects)

    uniq_subj = sorted(np.unique(subjects))
    print(f"[{tag}/{dataset_name}] {len(uniq_subj)} subjects | "
          f"EA={'yes' if use_ea else 'no'} | n_csp={n_csp}")

    write_mode = "a" if (resume and csv_path.exists()) else "w"
    with open(csv_path, write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_mode == "w":
            writer.writeheader()

        for subj in uniq_subj:
            if subj in done_subjects:
                continue

            t0 = time.time()
            test_mask  = subjects == subj
            train_mask = ~test_mask

            X_train, y_train = X[train_mask], y[train_mask]
            X_test,  y_test  = X[test_mask],  y[test_mask]

            # CSP features
            csp = ManualCSP(n_components=n_csp)
            try:
                csp.fit(X_train, y_train)
            except Exception as e:
                print(f"  [SKIP] subject {subj}: CSP fit failed — {e}")
                continue

            F_train = csp.transform(X_train)
            F_test  = csp.transform(X_test)

            scaler  = StandardScaler()
            F_train = scaler.fit_transform(F_train)
            F_test  = scaler.transform(F_test)

            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(F_train, y_train)
            pred = lda.predict(F_test)

            acc   = float((pred == y_test).mean())
            bac   = float(balanced_accuracy_score(y_test, pred))
            kappa = float(cohen_kappa_score(y_test, pred))
            elapsed = (time.time() - t0) / 60.0

            writer.writerow({
                "dataset": dataset_name, "subject": subj,
                "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
                "acc": f"{acc:.4f}", "bac": f"{bac:.4f}", "kappa": f"{kappa:.4f}",
                "time_min": f"{elapsed:.3f}",
            })
            f.flush()
            print(f"  [{subj:03d}] acc={acc*100:.1f}%  bac={bac*100:.1f}%  κ={kappa:.3f}  "
                  f"({elapsed*60:.1f}s)")

    # summary
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("dataset") == dataset_name:
                rows.append(row)
    accs = [float(r["acc"]) for r in rows]
    kaps = [float(r["kappa"]) for r in rows]
    print(f"\n[{tag}/{dataset_name}] {len(accs)} subjects | "
          f"Acc={np.mean(accs)*100:.2f}% ±{np.std(accs)*100:.2f} | "
          f"κ={np.mean(kaps):.3f}")
    print(f"Saved → {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cho2017", "lee2019", "both", "physionet"],
                        default="both")
    parser.add_argument("--ea",     action="store_true")
    parser.add_argument("--both_ea", action="store_true",
                        help="Run both CSP-LDA (no EA) and EA-CSP-LDA in sequence")
    parser.add_argument("--n_csp",  type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    datasets = (["cho2017", "lee2019"] if args.dataset == "both"
                else [args.dataset])

    ea_flags = ([False, True] if args.both_ea else [args.ea])

    for ds in datasets:
        for ea in ea_flags:
            run_loso_csp_lda(ds, use_ea=ea, n_csp=args.n_csp, resume=args.resume)


if __name__ == "__main__":
    main()
