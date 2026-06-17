from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS = PROJECT / "results"
RUNS = RESULTS / "runs"
CROSS = ROOT / "cross_dataset.py"
SUMMARY = PROJECT / "cross_dataset_priority_20260617.md"
PREP = PROJECT / "preprocessed_sfreq100"

EXPERIMENTS = [
    dict(priority=5, group="DSA+SEA AdaBN+Con", name="EEGNet + DSA+SEA+AdaBN+Con", run_id="20260617_sfreq100_eegnet_datasetea_subjectea_adabn_con", model="eegnetcontrastive", metric="adabn", args=["--both", "--model", "eegnetcontrastive", "--dataset_ea", "--ea", "--adabn"]),
    dict(priority=5, group="DSA+SEA AdaBN+Con", name="Conformer + DSA+SEA+AdaBN+Con", run_id="20260617_sfreq100_conformer_datasetea_subjectea_adabn_con", model="conformercontrastive", metric="adabn", args=["--both", "--model", "conformercontrastive", "--dataset_ea", "--ea", "--adabn"]),
    dict(priority=4, group="DANN", name="CSPNet + DSA+SEA+DANN", run_id="20260617_sfreq100_cspnet_datasetea_subjectea_dann", model="cspnetdann", metric="base", args=["--both", "--model", "cspnetdann", "--dataset_ea", "--ea"]),
]

METRIC_COLUMNS = {
    "base": ("acc", "kappa"),
    "adabn": ("adabn_acc", "adabn_kappa"),
    "tent": ("tent_acc", "tent_kappa"),
}


def csv_path(exp, src, tgt):
    name = "loso_results_%s_cross_%s_to_%s_%s.csv" % (exp["run_id"], src, tgt, exp["model"])
    return RESULTS / name


def read_metric(path: Path, metric: str):
    if not path.exists():
        return 0, None, None
    acc_col, kap_col = METRIC_COLUMNS[metric]
    accs, kaps = [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get(acc_col) not in ("", None):
                accs.append(float(row[acc_col]))
            if row.get(kap_col) not in ("", None):
                kaps.append(float(row[kap_col]))
    return len(accs), (sum(accs) / len(accs) if accs else None), (sum(kaps) / len(kaps) if kaps else None)


def fmt(v):
    n, acc, kap = v
    if acc is None:
        return "n=%d" % n
    return "%.2f%% / k=%.3f (n=%d)" % (acc, kap, n)


def avg(v1, v2):
    if v1[1] is None or v2[1] is None:
        return ""
    return "%.2f%%" % ((v1[1] + v2[1]) / 2)


def write_summary(note=""):
    lines = [
        "# Cross-Dataset Priority Queue (2026-06-17)", "",
        "Last updated: %s KST" % datetime.now().strftime("%Y-%m-%d %H:%M"), "",
        "- Input: /home/hkim/MI_test/preprocessed_sfreq100, MI_N_TIMES=201",
        "- DSA+SEA = --dataset_ea --ea.",
        "- AdaBN+Con uses SupCon contrastive wrappers for EEGNet/Conformer.",
        "- DANN uses binary source-vs-target domain adversarial training with CSPNet.", "",
        "| Priority | Group | Method | Status | Cho->Lee | Lee->Cho | Avg |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for exp in EXPERIMENTS:
        v1 = read_metric(csv_path(exp, "cho2017", "lee2019"), exp["metric"])
        v2 = read_metric(csv_path(exp, "lee2019", "cho2017"), exp["metric"])
        status = "completed" if v1[0] and v2[0] else ("partial" if v1[0] or v2[0] else "pending")
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (exp["priority"], exp["group"], exp["name"], status, fmt(v1), fmt(v2), avg(v1, v2)))
    if note:
        lines += ["", "## Queue Status", "", note]
    SUMMARY.write_text("\n".join(lines) + "\n")
    print(SUMMARY, flush=True)


def run_exp(exp):
    out1 = csv_path(exp, "cho2017", "lee2019")
    out2 = csv_path(exp, "lee2019", "cho2017")
    if out1.exists() and out2.exists():
        write_summary("- Skipped %s because both CSV files already exist." % exp["name"])
        return 0
    env = os.environ.copy()
    env["MI_PREPROCESSED_DIR"] = str(PREP)
    env["MI_N_TIMES"] = "201"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    log = RUNS / ("cross_dataset_priority_20260617_%s.log" % exp["run_id"])
    cmd = [sys.executable, "-u", str(CROSS), *exp["args"], "--run_id", exp["run_id"]]
    print("[run] P%s %s -> %s" % (exp["priority"], exp["name"], log), flush=True)
    with log.open("a") as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    write_summary("- Last completed %s exit=%s" % (exp["name"], rc))
    return rc


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    write_summary("- Queue started.")
    for exp in EXPERIMENTS:
        rc = run_exp(exp)
        if rc != 0:
            write_summary("- Queue stopped at %s exit=%s." % (exp["name"], rc))
            return rc
    write_summary("- Queue completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
