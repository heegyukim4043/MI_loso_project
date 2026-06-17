"""
Pipeline harness for MI EEG LOSO experiments.

Features:
  - Preprocess datasets (skip if preprocessed files exist)
  - Run LOSO training for selected models/datasets
  - Run MRFBCSP+LDA baseline (optional)
  - Resume support via --run_id + --resume
  - Run manifest logging

Example:
  python pipeline.py --datasets cho2017,lee2019 --models spdnet,min2net --augment_models spdnet \
    --include_mrfbcsp --run_id 20260410_2100 --resume
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_DIR = os.path.join(REPO_DIR, "preprocessed")
DEFAULT_RESULTS_DIR = os.path.join(REPO_DIR, "results")


def _split_list(value: str):
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_manifest(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _run_cmd(cmd, step, manifest_path):
    print(">> " + " ".join(cmd))
    step["start_time"] = datetime.now().isoformat(timespec="seconds")
    _write_manifest(manifest_path, step["_manifest_root"])
    result = subprocess.run(cmd)
    step["end_time"] = datetime.now().isoformat(timespec="seconds")
    step["returncode"] = result.returncode
    step["status"] = "ok" if result.returncode == 0 else "failed"
    _write_manifest(manifest_path, step["_manifest_root"])
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="cho2017,lee2019",
                        help="Comma list of datasets (default: cho2017,lee2019)")
    parser.add_argument("--models", default="spdnet,min2net",
                        help="Comma list of models (default: spdnet,min2net)")
    parser.add_argument("--augment_models", default="",
                        help="Comma list of models to run with --augment")
    parser.add_argument("--include_mrfbcsp", action="store_true",
                        help="Run MRFBCSP+LDA baseline")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip preprocessing step")
    parser.add_argument("--force_preprocess", action="store_true",
                        help="Force preprocessing even if .npz exists")
    parser.add_argument("--run_id", default=None,
                        help="Run id for output filenames (default: timestamp)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: results/runs/<run_id>)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume by passing --resume to training scripts")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter to use (default: current)")
    args = parser.parse_args()

    datasets = _split_list(args.datasets)
    models = _split_list(args.models)
    augment_models = set(_split_list(args.augment_models))

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(DEFAULT_RESULTS_DIR, "runs", run_id)
    _ensure_dir(out_dir)

    manifest = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "models": models,
        "augment_models": sorted(augment_models),
        "include_mrfbcsp": args.include_mrfbcsp,
        "skip_preprocess": args.skip_preprocess,
        "force_preprocess": args.force_preprocess,
        "out_dir": out_dir,
        "steps": [],
    }
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    _write_manifest(manifest_path, manifest)

    # Preprocess
    if not args.skip_preprocess:
        for ds in datasets:
            npz = os.path.join(PREPROCESS_DIR, f"{ds}.npz")
            if (not args.force_preprocess) and os.path.exists(npz):
                step = {"name": f"preprocess:{ds}", "status": "skipped", "_manifest_root": manifest}
                manifest["steps"].append(step)
                _write_manifest(manifest_path, manifest)
                continue
            cmd = [args.python, os.path.join(REPO_DIR, "preprocess_data.py"), "--dataset", ds]
            step = {"name": f"preprocess:{ds}", "cmd": cmd, "status": "pending", "_manifest_root": manifest}
            manifest["steps"].append(step)
            _run_cmd(cmd, step, manifest_path)

    # Train LOSO
    for ds in datasets:
        for model in models:
            cmd = [
                args.python, os.path.join(REPO_DIR, "train_loso.py"),
                "--dataset", ds,
                "--model", model,
                "--out_dir", out_dir,
                "--run_id", run_id,
            ]
            if args.resume:
                cmd.append("--resume")
            if model in augment_models:
                cmd.append("--augment")
            step = {"name": f"train:{ds}:{model}", "cmd": cmd, "status": "pending", "_manifest_root": manifest}
            manifest["steps"].append(step)
            _run_cmd(cmd, step, manifest_path)

    # MRFBCSP baseline
    if args.include_mrfbcsp:
        for ds in datasets:
            cmd = [
                args.python, os.path.join(REPO_DIR, "mrfbcsp_loso.py"),
                "--dataset", ds,
                "--out_dir", out_dir,
                "--run_id", run_id,
            ]
            if args.resume:
                cmd.append("--resume")
            step = {"name": f"mrfbcsp:{ds}", "cmd": cmd, "status": "pending", "_manifest_root": manifest}
            manifest["steps"].append(step)
            _run_cmd(cmd, step, manifest_path)

    manifest["end_time"] = datetime.now().isoformat(timespec="seconds")
    _write_manifest(manifest_path, manifest)
    print(f"Pipeline complete. Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
