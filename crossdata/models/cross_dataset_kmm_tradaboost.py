"""Cross-dataset KMM-TrAdaBoost baseline.

Train on all subjects of source dataset, test per-subject on target dataset.
Uses common channel set (standard_mi) shared by Cho2017 and Lee2019.

Usage
-----
    python cross_dataset_kmm_tradaboost.py --both
    python cross_dataset_kmm_tradaboost.py --both --ea
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cross_dataset import (
    RESULTS_DIR,
    SEED,
    N_TIMES,
    find_common_channels,
    load_dataset,
    normalize,
    apply_dataset_ea,
)
from eeg_ea import apply_ea_loso
from mrfbcsp_loso import ManualCSP


def median_gamma(Xs: np.ndarray, Xt: np.ndarray, max_samples: int = 1200) -> float:
    rng = np.random.default_rng(SEED)
    X = np.vstack([Xs, Xt])
    if len(X) > max_samples:
        X = X[rng.choice(len(X), size=max_samples, replace=False)]
    X = X[: min(400, len(X))]
    diffs = X[:, None, :] - X[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    med = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
    return 1.0 / max(med, 1e-6)


def kmm_proxy_weights(Xs: np.ndarray, Xt: np.ndarray, B: float = 20.0) -> np.ndarray:
    gamma = median_gamma(Xs, Xt)
    Kst = rbf_kernel(Xs, Xt, gamma=gamma)
    w = Kst.mean(axis=1)
    w = w / (np.mean(w) + 1e-12)
    w = np.clip(w, 0.0, B)
    w = w / (np.mean(w) + 1e-12)
    return w.astype(np.float64)


def make_adaboost(n_estimators: int, learning_rate: float):
    stump = DecisionTreeClassifier(max_depth=1, random_state=SEED)
    attempts = [
        dict(estimator=stump, n_estimators=n_estimators, learning_rate=learning_rate, random_state=SEED),
        dict(estimator=stump, n_estimators=n_estimators, learning_rate=learning_rate, algorithm="SAMME", random_state=SEED),
        dict(base_estimator=stump, n_estimators=n_estimators, learning_rate=learning_rate, random_state=SEED),
        dict(base_estimator=stump, n_estimators=n_estimators, learning_rate=learning_rate, algorithm="SAMME", random_state=SEED),
        dict(n_estimators=n_estimators, learning_rate=learning_rate, random_state=SEED),
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return AdaBoostClassifier(**kwargs)
        except TypeError as e:
            last_error = e
    raise RuntimeError(f"AdaBoostClassifier init failed: {last_error}")


def run_cross(
    train_name: str,
    test_name: str,
    run_id: str,
    use_ea: bool = False,
    use_dataset_ea: bool = False,
    channel_set: str = "standard_mi",
    n_csp: int = 8,
    n_estimators: int = 100,
    learning_rate: float = 0.5,
):
    common_ch, cho_idx, lee_idx = find_common_channels(channel_set=channel_set)
    train_idx = cho_idx if train_name == "cho2017" else lee_idx
    test_idx  = cho_idx if test_name  == "cho2017" else lee_idx

    method = "KMM-TrAdaBoost-Cross"
    if use_dataset_ea and use_ea:
        method = "DatasetEA+SubjectEA+KMM-TrAdaBoost-Cross"
    elif use_dataset_ea:
        method = "DatasetEA+KMM-TrAdaBoost-Cross"
    elif use_ea:
        method = "EA+KMM-TrAdaBoost-Cross"

    print(f"\n{'='*60}")
    print(f"  {method}")
    print(f"  TRAIN={train_name.upper()} -> TEST={test_name.upper()}")
    print(f"  channels={len(common_ch)}  n_csp={n_csp}  estimators={n_estimators}")
    print(f"{'='*60}")

    X_train, y_train, subj_train = load_dataset(train_name, train_idx, n_times=N_TIMES)
    X_test,  y_test,  subj_test  = load_dataset(test_name,  test_idx,  n_times=N_TIMES)

    if use_dataset_ea:
        print("  Applying dataset-level EA...")
        X_train = apply_dataset_ea(X_train)
        X_test  = apply_dataset_ea(X_test)

    if use_ea:
        print("  Applying subject-level EA...")
        X_train = apply_ea_loso(X_train, subj_train)
        X_test  = apply_ea_loso(X_test,  subj_test)

    X_train, X_test = normalize(X_train, X_test)

    tag = f"cross_{train_name}_to_{test_name}_kmm_tradaboost"
    out_csv = Path(RESULTS_DIR) / f"loso_results_{run_id}_{tag}.csv"
    fields = ["train_ds","test_ds","model","subject","n_test",
              "acc","precision","f1","bac","kappa","time_min"]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    # Fit CSP on full source data
    csp = ManualCSP(n_components=n_csp)
    csp.fit(X_train, y_train)
    F_train_all = csp.transform(X_train)
    scaler = StandardScaler()
    F_train_all = scaler.fit_transform(F_train_all)

    rows = []
    n_test_subjs = len(np.unique(subj_test))
    for i, subj in enumerate(sorted(np.unique(subj_test)), 1):
        t0 = time.time()
        mask = subj_test == subj
        F_test = scaler.transform(csp.transform(X_test[mask]))

        weights = kmm_proxy_weights(F_train_all, F_test)
        clf = make_adaboost(n_estimators=n_estimators, learning_rate=learning_rate)
        clf.fit(F_train_all, y_train, sample_weight=weights)
        pred = clf.predict(F_test)
        y_true = y_test[mask]

        acc  = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, average="macro", zero_division=0)
        f1   = f1_score(y_true, pred, average="macro", zero_division=0)
        bac  = balanced_accuracy_score(y_true, pred)
        kap  = cohen_kappa_score(y_true, pred)
        elapsed = (time.time() - t0) / 60.0

        row = {"train_ds": train_name, "test_ds": test_name, "model": "kmm_tradaboost",
               "subject": int(subj), "n_test": int(mask.sum()),
               "acc": round(acc*100, 2), "precision": round(prec*100, 2),
               "f1": round(f1*100, 2), "bac": round(bac*100, 2),
               "kappa": round(kap, 3), "time_min": round(elapsed, 2)}
        rows.append(row)
        with out_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        eta = elapsed * (n_test_subjs - i)
        print(f"  [{i:02d}/{n_test_subjs}] S{subj:02d}: acc={acc*100:.1f}%  k={kap:.3f}  [{elapsed:.1f}min | ETA {eta:.0f}min]", flush=True)

    accs = [r["acc"] for r in rows]
    kaps = [r["kappa"] for r in rows]
    print(f"\n  {train_name.upper()} -> {test_name.upper()} | {method}")
    print(f"  Accuracy : {np.mean(accs):.2f} ± {np.std(accs):.2f} %")
    print(f"  Kappa    : {np.mean(kaps):.3f} ± {np.std(kaps):.3f}")
    print(f"  Saved    : {out_csv}")
    return float(np.mean(accs)), float(np.mean(kaps)), out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--train", choices=["cho2017","lee2019"], default="cho2017")
    parser.add_argument("--test",  choices=["cho2017","lee2019"], default="lee2019")
    parser.add_argument("--ea", action="store_true", help="Subject-level EA")
    parser.add_argument("--dataset_ea", action="store_true", help="Dataset-level EA")
    parser.add_argument("--channel_set", choices=["common","standard_mi"], default="standard_mi")
    parser.add_argument("--n_csp", type=int, default=8)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.5)
    parser.add_argument("--run_id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_kmm_cross"

    pairs = [("cho2017","lee2019"),("lee2019","cho2017")] if args.both else [(args.train, args.test)]
    for train_ds, test_ds in pairs:
        run_cross(
            train_name=train_ds, test_name=test_ds, run_id=run_id,
            use_ea=args.ea, use_dataset_ea=args.dataset_ea,
            channel_set=args.channel_set, n_csp=args.n_csp,
            n_estimators=args.n_estimators, learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
