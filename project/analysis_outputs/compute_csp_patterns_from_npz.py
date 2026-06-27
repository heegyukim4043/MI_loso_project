"""Compute CSP topomaps from streamed MOABB NPZ files.

Inputs are files created by stream_moabb_preprocess.py.
For each file, this creates NoEA and EA CSP pattern images and metrics.
It also creates dataset-level average absolute-pattern figures.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def euclidean_align(X: np.ndarray) -> np.ndarray:
    cov = np.mean([x @ x.T / max(x.shape[1] - 1, 1) for x in X], axis=0)
    cov += 1e-10 * np.eye(cov.shape[0])
    inv_sqrt = linalg.inv(linalg.sqrtm(cov)).real
    return np.asarray([inv_sqrt @ x for x in X], dtype="float32")


def make_info(ch_names: list[str], sfreq: float) -> mne.Info:
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
    return info


def fit_csp(X: np.ndarray, y: np.ndarray, n_components: int, cv_splits: int):
    X = np.asarray(X, dtype="float64")
    csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)
    pipe = Pipeline(
        [
            ("csp", csp),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )
    n_per_class = min(np.bincount(y))
    splits = max(2, min(cv_splits, int(n_per_class)))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2026)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy")
    csp.fit(X, y)
    return csp, scores


def lateralization(pattern: np.ndarray, ch_names: list[str]) -> dict[str, float]:
    left_roi = [ch for ch in ["C3", "FC3", "CP3", "C5", "C1"] if ch in ch_names]
    right_roi = [ch for ch in ["C4", "FC4", "CP4", "C6", "C2"] if ch in ch_names]
    idx = {ch: i for i, ch in enumerate(ch_names)}
    left = float(np.mean(np.abs([pattern[idx[ch]] for ch in left_roi]))) if left_roi else np.nan
    right = float(np.mean(np.abs([pattern[idx[ch]] for ch in right_roi]))) if right_roi else np.nan
    return {
        "left_sensorimotor_abs": left,
        "right_sensorimotor_abs": right,
        "lr_asymmetry": float((left - right) / (left + right + 1e-12)),
    }


def sensorimotor_ratio(pattern: np.ndarray, ch_names: list[str]) -> float:
    roi = [ch for ch in ["C3", "C4", "FC3", "FC4", "CP3", "CP4", "C1", "C2", "C5", "C6"] if ch in ch_names]
    idx = {ch: i for i, ch in enumerate(ch_names)}
    if not roi:
        return np.nan
    return float(np.mean(np.abs([pattern[idx[ch]] for ch in roi])) / (np.mean(np.abs(pattern)) + 1e-12))


def plot_patterns(patterns: np.ndarray, info: mne.Info, title: str, out_file: Path, n_components: int):
    n_cols = 3
    n_rows = int(np.ceil(n_components / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for i in range(n_components):
        mne.viz.plot_topomap(patterns[i], info, axes=axes[i], show=False, contours=6, cmap="RdBu_r")
        axes[i].set_title(f"CSP {i + 1}")
    for j in range(n_components, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def process_file(path: Path, out_root: Path, n_components: int, cv_splits: int, skip_existing: bool):
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype("float32")
    y = data["y"].astype("int64")
    ch_names = [str(x) for x in data["ch_names"]]
    sfreq = float(data["sfreq"][0])
    dataset = str(data["dataset"][0])
    subject = int(data["subject"][0])
    session = str(data["session"][0])
    run = str(data["run"][0])

    info = make_info(ch_names, sfreq)
    stem = path.stem
    out_dir = out_root / dataset / f"sub-{subject:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_figs = [
        out_dir / f"{stem}_noea_csp_patterns.png",
        out_dir / f"{stem}_ea_csp_patterns.png",
    ]
    if skip_existing and all(p.exists() for p in expected_figs):
        del data, X, y
        gc.collect()
        return [], []

    rows = []
    pattern_records = []
    for condition, X_cond in [("NoEA", X), ("EA", euclidean_align(X))]:
        fig_file = out_dir / f"{stem}_{condition.lower()}_csp_patterns.png"
        csp, scores = fit_csp(X_cond, y, n_components, cv_splits)
        if not (skip_existing and fig_file.exists()):
            plot_patterns(csp.patterns_[:n_components], info, f"{stem} - {condition}", fig_file, n_components)
        for comp in range(n_components):
            pattern = csp.patterns_[comp]
            row = {
                "file": str(path),
                "dataset": dataset,
                "subject": subject,
                "session": session,
                "run": run,
                "condition": condition,
                "component": comp + 1,
                "n_trials": int(X.shape[0]),
                "n_channels": int(X.shape[1]),
                "n_times": int(X.shape[2]),
                "cv_balanced_accuracy_mean": float(np.mean(scores)),
                "cv_balanced_accuracy_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "sensorimotor_ratio": sensorimotor_ratio(pattern, ch_names),
            }
            row.update(lateralization(pattern, ch_names))
            rows.append(row)
            pattern_records.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "component": comp + 1,
                    "ch_names": tuple(ch_names),
                    "pattern_abs": np.abs(pattern).astype("float64"),
                    "sfreq": sfreq,
                }
            )
    del data, X, y
    gc.collect()
    return rows, pattern_records


def plot_group_average(pattern_records: list[dict], out_root: Path):
    grouped = {}
    for rec in pattern_records:
        key = (rec["dataset"], rec["condition"], rec["component"], rec["ch_names"])
        grouped.setdefault(key, []).append(rec["pattern_abs"])

    for (dataset, condition, component, ch_names), vals in grouped.items():
        if len(vals) < 2:
            continue
        avg = np.mean(vals, axis=0)
        info = make_info(list(ch_names), 100.0)
        out_dir = out_root / "group_average" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        mne.viz.plot_topomap(avg, info, axes=ax, show=False, contours=6, cmap="Reds")
        ax.set_title(f"{dataset} {condition} abs CSP {component} n={len(vals)}")
        fig.savefig(out_dir / f"{dataset}_{condition.lower()}_abs_csp{component}_avg.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="moabb_streamed_npz")
    parser.add_argument("--output-dir", default="analysis_outputs/csp_patterns_streamed")
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_patterns = []
    metrics_path = out_root / "csp_pattern_metrics.csv"
    existing_metrics = pd.read_csv(metrics_path) if args.skip_existing and metrics_path.exists() else pd.DataFrame()
    files = sorted(input_dir.rglob("*_mi.npz"))
    if not files:
        raise SystemExit(f"No *_mi.npz files found under {input_dir}")

    for path in files:
        print(f"Processing {path}")
        rows, patterns = process_file(path, out_root, args.n_components, args.cv_splits, args.skip_existing)
        all_rows.extend(rows)
        all_patterns.extend(patterns)

    metrics = pd.DataFrame(all_rows)
    if not existing_metrics.empty:
        metrics = pd.concat([existing_metrics, metrics], ignore_index=True)
        metrics = metrics.drop_duplicates(["file", "condition", "component"], keep="last")
    metrics.to_csv(metrics_path, index=False)

    summary = (
        metrics.groupby(["dataset", "condition"], as_index=False)
        .agg(
            n_files=("file", "nunique"),
            mean_cv_balanced_accuracy=("cv_balanced_accuracy_mean", "mean"),
            mean_sensorimotor_ratio=("sensorimotor_ratio", "mean"),
            mean_abs_lr_asymmetry=("lr_asymmetry", lambda x: float(np.mean(np.abs(x)))),
        )
    )
    summary.to_csv(out_root / "csp_pattern_summary.csv", index=False)
    plot_group_average(all_patterns, out_root)

    readme = out_root / "README.md"
    readme.write_text(
        "# Streamed CSP Pattern Results\n\n"
        f"Input: `{input_dir}`\n\n"
        "Per-file topomaps are under dataset/sub-* folders.\n\n"
        "```csv\n"
        + summary.to_csv(index=False)
        + "```\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {out_root}")


if __name__ == "__main__":
    main()
