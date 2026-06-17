import csv
from pathlib import Path


RUNS_DIR = Path("/home/hkim/MI_test/results/runs")
OUT_MD = RUNS_DIR / "subject_performance_tables_20260515.md"


METHODS = [
    {
        "title": "Weighted-only Best",
        "setting": "heuristic / combined / weighted / min_weight=0.3 / keep_ratio=0.8",
        "csv": RUNS_DIR / "sweep_20260423_A_weighted_mw3_cho2017" /
               "loso_results_20260423_A_weighted_mw3_cho2017_cspnet_selheuristicweighted80combined.csv",
    },
    {
        "title": "Quality x Diversity Best",
        "setting": "heuristic / quality_diversity / weighted / alpha=0.4 / min_weight=0.5 / keep_ratio=0.4",
        "csv": RUNS_DIR / "sweep_20260423_B_qdiv_a4_cho2017" /
               "loso_results_20260423_B_qdiv_a4_cho2017_cspnet_selheuristicweighted40quality_diversity.csv",
    },
    {
        "title": "Uncertainty Best (Completed Family)",
        "setting": "uncertainty / quality_entropy / weighted / lambda=0.2 / min_weight=0.5 / keep_ratio=0.8",
        "csv": RUNS_DIR / "sweep_20260423_C_unc_l2_cho2017" /
               "loso_results_20260423_C_unc_l2_cho2017_cspnet_seluncertaintyweighted80quality_entropy.csv",
    },
    {
        "title": "Phase-2 Uncertainty mw=0.2",
        "setting": "uncertainty / quality_entropy / weighted / min_weight=0.2 / keep_ratio=0.8",
        "csv": RUNS_DIR / "sweep_20260506_U_qent_mw2_cho2017" /
               "loso_results_20260506_U_qent_mw2_cho2017_cspnet_seluncertaintyweighted80quality_entropy.csv",
    },
    {
        "title": "Phase-2 Uncertainty mw=0.3",
        "setting": "uncertainty / quality_entropy / weighted / min_weight=0.3 / keep_ratio=0.8",
        "csv": RUNS_DIR / "sweep_20260506_U_qent_mw3_cho2017" /
               "loso_results_20260506_U_qent_mw3_cho2017_cspnet_seluncertaintyweighted80quality_entropy.csv",
    },
    {
        "title": "Phase-2 Uncertainty mw=0.4",
        "setting": "uncertainty / quality_entropy / weighted / min_weight=0.4 / keep_ratio=0.8",
        "csv": RUNS_DIR / "sweep_20260506_U_qent_mw4_cho2017" /
               "loso_results_20260506_U_qent_mw4_cho2017_cspnet_seluncertaintyweighted80quality_entropy.csv",
    },
]


def read_rows(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["subject"]))
    return rows


def acc_pct(row):
    return f"{float(row['acc']) * 100:.2f}"


def kappa_fmt(row):
    return f"{float(row['kappa']):.3f}"


def val_acc_pct(row):
    return f"{float(row['best_val_acc']) * 100:.2f}"


def mean_acc(rows):
    vals = [float(r["acc"]) for r in rows]
    return sum(vals) / len(vals) * 100.0


def mean_kappa(rows):
    vals = [float(r["kappa"]) for r in rows]
    return sum(vals) / len(vals)


def main():
    lines = []
    lines.append("# Subject-wise Performance Tables (2026-05-15)")
    lines.append("")
    lines.append("Dataset: `cho2017`  ")
    lines.append("Model: `cspnet`")
    lines.append("")
    lines.append("Each section below stores subject-level LOSO results for one representative method/configuration.")
    lines.append("")

    for method in METHODS:
        rows = read_rows(method["csv"])
        lines.append(f"## {method['title']}")
        lines.append("")
        lines.append(f"Setting: `{method['setting']}`")
        lines.append("")
        lines.append(f"Source CSV: `{method['csv']}`")
        lines.append("")
        lines.append(f"Mean accuracy: `{mean_acc(rows):.2f}%`  ")
        lines.append(f"Mean kappa: `{mean_kappa(rows):.3f}`")
        lines.append("")
        lines.append("| subject | acc (%) | kappa | best_epoch | best_val_acc (%) | time_min |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {int(row['subject']):02d} | {acc_pct(row)} | {kappa_fmt(row)} | "
                f"{int(row['best_epoch'])} | {val_acc_pct(row)} | {float(row['time_min']):.2f} |"
            )
        lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(OUT_MD))


if __name__ == "__main__":
    main()
