from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS = PROJECT / "results"
RUNS = RESULTS / "runs"
CROSS = ROOT / "cross_dataset.py"
SUMMARY = PROJECT / "colab_dsa_sea_snapshot_20260628.md"

DEFAULT_RUN_PREFIX = "20260628_colab_dsa_sea_snapshot"
DEFAULT_PREP = PROJECT / "preprocessed_sfreq100"
MODELS = ("cspnet", "eegnet", "conformer")
SNAPSHOT_T0 = 50


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


def run_id_for(prefix: str, model: str) -> str:
    return f"{prefix}_{model}"


def write_summary(prefix: str, models: list[str], note: str = "") -> None:
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
    for model in models:
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
    parser.add_argument("--run_prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    if not (args.preprocessed_dir / "cho2017.npz").exists() or not (args.preprocessed_dir / "lee2019.npz").exists():
        raise FileNotFoundError(
            f"Expected cho2017.npz and lee2019.npz under {args.preprocessed_dir}. "
            "Pass --preprocessed_dir or set MI_PREPROCESSED_DIR."
        )

    write_summary(args.run_prefix, args.models, "- Queue started.")
    for model in args.models:
        for train, test in direction_pairs(args.direction):
            rc = run_one(model, train, test, args.run_prefix, args.preprocessed_dir, args.force)
            write_summary(args.run_prefix, args.models, f"- Last run: `{model}` `{train}->{test}` exit={rc}.")
            if rc != 0:
                write_summary(args.run_prefix, args.models, f"- Queue stopped at `{model}` `{train}->{test}` exit={rc}.")
                return rc
    write_summary(args.run_prefix, args.models, "- Queue completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
