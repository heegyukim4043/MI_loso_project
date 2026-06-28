from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS = PROJECT / "results"
RUNS = RESULTS / "runs"
CROSS = ROOT / "cross_dataset.py"
SUMMARY = PROJECT / "colab_dsa_sea_snapshot_20260628.md"

DEFAULT_RUN_PREFIX = "20260628_colab_dsa_sea_snapshot"
DEFAULT_PREP = PROJECT / "preprocessed_sfreq100"
DEFAULT_BACKUP = os.environ.get("MI_BACKUP_DIR")
MODELS = ("cspnet", "eegnet", "conformer")
SNAPSHOT_T0 = 50
EXPECTED_PREPROCESSED = {
    "cho2017": {
        "shape": (10520, 64, 201),
        "subjects": 52,
        "trials_per_subject": None,
        "sfreq": 100.0,
    },
    "lee2019": {
        "shape": (10800, 62, 201),
        "subjects": 54,
        "trials_per_subject": 200,
        "sfreq": 100.0,
    },
}


def result_csv(run_id: str, train: str, test: str, model: str) -> Path:
    return RESULTS / f"loso_results_{run_id}_cross_{train}_to_{test}_{model}.csv"


def read_metric(path: Path, metric_col: str = "snap_acc") -> tuple[int, Optional[float], Optional[float]]:
    if not path.exists():
        return 0, None, None
    accs: list[float] = []
    snap_adabn: list[float] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get(metric_col) not in ("", None):
                accs.append(float(row[metric_col]))
            if row.get("snap_adabn_acc") not in ("", None):
                snap_adabn.append(float(row["snap_adabn_acc"]))
    return (
        len(accs),
        sum(accs) / len(accs) if accs else None,
        sum(snap_adabn) / len(snap_adabn) if snap_adabn else None,
    )


def fmt_metric(v: tuple[int, Optional[float], Optional[float]]) -> str:
    n, snap, snap_adabn = v
    if snap is None:
        return f"n={n}"
    if snap_adabn is None:
        return f"Snap={snap:.2f}% (n={n})"
    return f"Snap={snap:.2f}% / Snap+AdaBN={snap_adabn:.2f}% (n={n})"


def direction_pairs(direction: str) -> list[tuple[str, str]]:
    if direction == "cho2lee":
        return [("cho2017", "lee2019")]
    if direction == "lee2cho":
        return [("lee2019", "cho2017")]
    if direction == "both":
        return [("cho2017", "lee2019"), ("lee2019", "cho2017")]
    raise ValueError(direction)


def validate_preprocessed_dir(preprocessed_dir: Path) -> None:
    """Require the same sfreq100 files used by the existing cross-dataset matrix."""
    for name, expected in EXPECTED_PREPROCESSED.items():
        path = preprocessed_dir / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {path}. Pass --preprocessed_dir or set MI_PREPROCESSED_DIR."
            )
        data = np.load(path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        subjects = data["subjects"]
        sfreq = float(data["sfreq"])
        unique_subjects, counts = np.unique(subjects, return_counts=True)
        if tuple(X.shape) != expected["shape"]:
            raise ValueError(
                f"{name}.npz has X={X.shape}, expected {expected['shape']} for the "
                "legacy sfreq100 cross-dataset pipeline. Do not use the one-session "
                "MOABB Colab export for these runs."
            )
        if len(y) != X.shape[0] or len(subjects) != X.shape[0]:
            raise ValueError(f"{name}.npz length mismatch: X={len(X)} y={len(y)} subjects={len(subjects)}")
        if len(unique_subjects) != expected["subjects"]:
            raise ValueError(f"{name}.npz has {len(unique_subjects)} subjects, expected {expected['subjects']}")
        if abs(sfreq - expected["sfreq"]) > 1e-6:
            raise ValueError(f"{name}.npz has sfreq={sfreq}, expected {expected['sfreq']}")
        if expected["trials_per_subject"] is not None and not np.all(counts == expected["trials_per_subject"]):
            raise ValueError(
                f"{name}.npz trial counts per subject are {sorted(set(counts.tolist()))}, "
                f"expected all {expected['trials_per_subject']}."
            )
        labels, label_counts = np.unique(y, return_counts=True)
        print(
            f"[preproc ok] {name}: X={X.shape}, subjects={len(unique_subjects)}, "
            f"sfreq={sfreq}, labels={dict(zip(labels.tolist(), label_counts.tolist()))}",
            flush=True,
        )


def run_id_for(prefix: str, model: str) -> str:
    return f"{prefix}_{model}"


def backup_outputs(prefix: str, backup_dir: Optional[Path]) -> None:
    if backup_dir is None:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    if SUMMARY.exists():
        shutil.copy2(SUMMARY, backup_dir / SUMMARY.name)
        copied += 1
    for path in sorted(RESULTS.glob(f"loso_results_{prefix}_*_cross_*.csv")):
        shutil.copy2(path, backup_dir / path.name)
        copied += 1
    print(f"[backup] {copied} files -> {backup_dir}", flush=True)


def write_summary(prefix: str, note: str = "", backup_dir: Optional[Path] = None) -> None:
    lines = [
        "# Colab DSA+SEA+Snapshot Cross-Dataset Queue",
        "",
        f"Last updated: `{datetime.now():%Y-%m-%d %H:%M}`",
        "",
        "- Method: DatasetEA + SubjectEA + Snapshot ensemble",
        f"- Snapshot: `T0={SNAPSHOT_T0}`; `cross_dataset.py` stores both `snap_acc` and `snap_adabn_acc`.",
        "- DSA+SEA args: `--dataset_ea --ea`",
        "",
        "| Model | Cho->Lee | Lee->Cho | Avg Snap | Avg Snap+AdaBN |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        run_id = run_id_for(prefix, model)
        cho = read_metric(result_csv(run_id, "cho2017", "lee2019", model))
        lee = read_metric(result_csv(run_id, "lee2019", "cho2017", model))
        snap_vals = [v[1] for v in (cho, lee) if v[1] is not None]
        snap_adabn_vals = [v[2] for v in (cho, lee) if v[2] is not None]
        avg_snap = f"{sum(snap_vals) / len(snap_vals):.2f}%" if snap_vals else ""
        avg_snap_adabn = f"{sum(snap_adabn_vals) / len(snap_adabn_vals):.2f}%" if snap_adabn_vals else ""
        lines.append(f"| {model} | {fmt_metric(cho)} | {fmt_metric(lee)} | {avg_snap} | {avg_snap_adabn} |")
    if note:
        lines += ["", "## Queue Status", "", note]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] {SUMMARY}", flush=True)
    backup_outputs(prefix, backup_dir)


def should_skip(path: Path, expected_subjects: int) -> bool:
    n, snap, _ = read_metric(path)
    return path.exists() and n >= expected_subjects and snap is not None


def run_one(
    model: str,
    train: str,
    test: str,
    prefix: str,
    preprocessed_dir: Path,
    force: bool,
) -> int:
    run_id = run_id_for(prefix, model)
    expected_subjects = 54 if test == "lee2019" else 52
    out_csv = result_csv(run_id, train, test, model)
    if not force and should_skip(out_csv, expected_subjects):
        print(f"[skip] {model} {train}->{test}: complete CSV exists: {out_csv}", flush=True)
        return 0

    env = os.environ.copy()
    env["MI_PREPROCESSED_DIR"] = str(preprocessed_dir)
    env["MI_N_TIMES"] = "201"
    env.setdefault("PYTHONUNBUFFERED", "1")

    RUNS.mkdir(parents=True, exist_ok=True)
    log = RUNS / f"colab_dsa_sea_snapshot_{run_id}_{train}_to_{test}.log"
    cmd = [
        sys.executable,
        "-u",
        str(CROSS),
        "--train",
        train,
        "--test",
        test,
        "--model",
        model,
        "--dataset_ea",
        "--ea",
        "--snapshot_ensemble",
        "--snapshot_T0",
        str(SNAPSHOT_T0),
        "--run_id",
        run_id,
    ]

    print(f"[run] {model} {train}->{test}", flush=True)
    print(f"[log] {log}", flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"[done] {model} {train}->{test} exit={rc}", flush=True)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["cspnet"])
    parser.add_argument("--direction", choices=["cho2lee", "lee2cho", "both"], default="both")
    parser.add_argument("--preprocessed_dir", type=Path, default=Path(os.environ.get("MI_PREPROCESSED_DIR", DEFAULT_PREP)))
    parser.add_argument("--backup_dir", type=Path, default=Path(DEFAULT_BACKUP) if DEFAULT_BACKUP else None)
    parser.add_argument("--run_prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    validate_preprocessed_dir(args.preprocessed_dir)

    write_summary(args.run_prefix, "- Queue started.", args.backup_dir)
    for model in args.models:
        for train, test in direction_pairs(args.direction):
            rc = run_one(model, train, test, args.run_prefix, args.preprocessed_dir, args.force)
            write_summary(args.run_prefix, f"- Last run: `{model}` `{train}->{test}` exit={rc}.", args.backup_dir)
            if rc != 0:
                write_summary(args.run_prefix, f"- Queue stopped at `{model}` `{train}->{test}` exit={rc}.", args.backup_dir)
                return rc
    write_summary(args.run_prefix, "- Queue completed.", args.backup_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
