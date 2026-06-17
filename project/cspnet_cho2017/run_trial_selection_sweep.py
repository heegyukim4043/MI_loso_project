"""
Run a keep-ratio sweep for trial suitability selection and summarise the results.

Example:
  python run_trial_selection_sweep.py --dataset cho2017 --model spdnet \
      --augment --score_method combined --selection_mode hard
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
from datetime import datetime


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(REPO_DIR, "train_loso.py")
DEFAULT_OUT_DIR = os.path.join(REPO_DIR, "..", "results", "runs")


def parse_ratios(raw: str):
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if not (0.0 < val <= 1.0):
            raise ValueError(f"Invalid keep ratio: {val}")
        vals.append(val)
    if not vals:
        raise ValueError("At least one keep ratio is required.")
    return vals


def result_csv_path(out_dir: str, run_id: str, model: str, augment: bool,
                    keep_ratio: float, score_method: str, selection_mode: str,
                    selection_source: str):
    aug_tag = "_aug" if augment else ""
    sel_tag = ""
    if keep_ratio < 1.0:
        sel_tag = (
            f"_sel{selection_source}{selection_mode}"
            f"{int(keep_ratio * 100)}{score_method}"
        )
    return os.path.join(out_dir, f"loso_results_{run_id}_{model}{aug_tag}{sel_tag}.csv")


def summarise_result_csv(path: str):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    accs = [float(r["acc"]) for r in rows]
    bacs = [float(r["bac"]) for r in rows]
    kappas = [float(r["kappa"]) for r in rows]
    times = [float(r["time_min"]) for r in rows]
    train_sizes = [int(r["n_train"]) for r in rows]

    return {
        "subjects": len(rows),
        "acc_mean": statistics.mean(accs),
        "acc_std": statistics.pstdev(accs),
        "bac_mean": statistics.mean(bacs),
        "bac_std": statistics.pstdev(bacs),
        "kappa_mean": statistics.mean(kappas),
        "kappa_std": statistics.pstdev(kappas),
        "time_total_min": sum(times),
        "time_mean_min": statistics.mean(times),
        "n_train_mean": statistics.mean(train_sizes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cho2017", "lee2019"], required=True)
    parser.add_argument("--model", choices=["spdnet", "riemgat", "min2net", "cspnet"],
                        default="spdnet")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--score_method",
                        choices=["band_power", "laterality", "cov_quality", "combined",
                                 "confidence", "trueprob", "margin",
                                 "quality", "domain", "final", "rho_cosine"],
                        default="combined")
    parser.add_argument("--selection_source", choices=["heuristic", "classifier", "discriminator", "rho_loss"],
                        default="heuristic")
    parser.add_argument("--selection_mode", choices=["hard", "weighted"],
                        default="hard")
    parser.add_argument("--min_weight", type=float, default=0.25)
    parser.add_argument("--selector_epochs", type=int, default=100)
    parser.add_argument("--save_selection_plots", action="store_true")
    parser.add_argument("--selection_plot_max_points", type=int, default=4000)
    parser.add_argument("--ratios", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    ratios = parse_ratios(args.ratios)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(DEFAULT_OUT_DIR, f"sweep_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(
        out_dir,
        f"trial_selection_sweep_{run_id}_{args.dataset}_{args.model}_"
        f"{args.selection_mode}_{args.score_method}.csv",
    )
    summary_fields = [
        "dataset", "model", "augment", "selection_mode", "score_method",
        "keep_ratio", "subjects", "n_train_mean",
        "acc_mean", "acc_std", "bac_mean", "bac_std",
        "kappa_mean", "kappa_std", "time_total_min", "time_mean_min",
        "result_csv",
    ]

    with open(summary_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    for keep_ratio in ratios:
        cmd = [
            args.python, TRAIN,
            "--dataset", args.dataset,
            "--model", args.model,
            "--out_dir", out_dir,
            "--run_id", run_id,
            "--score_method", args.score_method,
            "--keep_ratio", str(keep_ratio),
            "--selection_source", args.selection_source,
            "--selection_mode", args.selection_mode,
            "--min_weight", str(args.min_weight),
            "--selector_epochs", str(args.selector_epochs),
            "--selection_plot_max_points", str(args.selection_plot_max_points),
        ]
        if args.augment:
            cmd.append("--augment")
        if args.resume:
            cmd.append("--resume")
        if args.save_selection_plots:
            cmd.append("--save_selection_plots")

        print(">> " + " ".join(cmd), flush=True)
        ret = subprocess.run(cmd, cwd=REPO_DIR)
        if ret.returncode != 0:
            raise SystemExit(ret.returncode)

        result_csv = result_csv_path(
            out_dir, run_id, args.model, args.augment,
            keep_ratio, args.score_method, args.selection_mode, args.selection_source,
        )
        stats = summarise_result_csv(result_csv)
        row = {
            "dataset": args.dataset,
            "model": args.model,
            "augment": int(args.augment),
            "selection_mode": args.selection_mode,
            "score_method": args.score_method,
            "keep_ratio": keep_ratio,
            "subjects": stats["subjects"],
            "n_train_mean": round(stats["n_train_mean"], 2),
            "acc_mean": round(stats["acc_mean"], 6),
            "acc_std": round(stats["acc_std"], 6),
            "bac_mean": round(stats["bac_mean"], 6),
            "bac_std": round(stats["bac_std"], 6),
            "kappa_mean": round(stats["kappa_mean"], 6),
            "kappa_std": round(stats["kappa_std"], 6),
            "time_total_min": round(stats["time_total_min"], 2),
            "time_mean_min": round(stats["time_mean_min"], 2),
            "result_csv": os.path.basename(result_csv),
        }
        with open(summary_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=summary_fields).writerow(row)

        print(
            f"  keep={int(keep_ratio * 100)}% | "
            f"acc={stats['acc_mean']*100:.2f}±{stats['acc_std']*100:.2f}% | "
            f"kappa={stats['kappa_mean']:.3f}",
            flush=True,
        )

    print(f"Sweep summary -> {summary_path}")


if __name__ == "__main__":
    main()
