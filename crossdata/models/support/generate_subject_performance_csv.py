import csv
import sys
from pathlib import Path


RUNS_DIR = Path("/home/hkim/MI_test/results/runs")
TABLES_BY_DATASET = {
    "cho2017": [
        ("weighted_mw3_r40", RUNS_DIR / "sweep_20260423_A_weighted_mw3_cho2017"
         / "loso_results_20260423_A_weighted_mw3_cho2017_cspnet_selheuristicweighted40combined.csv"),
        ("weighted_mw3_r60", RUNS_DIR / "sweep_20260423_A_weighted_mw3_cho2017"
         / "loso_results_20260423_A_weighted_mw3_cho2017_cspnet_selheuristicweighted60combined.csv"),
        ("weighted_mw3_r80", RUNS_DIR / "sweep_20260423_A_weighted_mw3_cho2017"
         / "loso_results_20260423_A_weighted_mw3_cho2017_cspnet_selheuristicweighted80combined.csv"),
        ("weighted_mw3_r100", RUNS_DIR / "sweep_20260423_A_weighted_mw3_cho2017"
         / "loso_results_20260423_A_weighted_mw3_cho2017_cspnet.csv"),
        ("weighted_mw5_r40", RUNS_DIR / "sweep_20260423_A_weighted_mw5_cho2017"
         / "loso_results_20260423_A_weighted_mw5_cho2017_cspnet_selheuristicweighted40combined.csv"),
        ("weighted_mw5_r60", RUNS_DIR / "sweep_20260423_A_weighted_mw5_cho2017"
         / "loso_results_20260423_A_weighted_mw5_cho2017_cspnet_selheuristicweighted60combined.csv"),
        ("weighted_mw5_r80", RUNS_DIR / "sweep_20260423_A_weighted_mw5_cho2017"
         / "loso_results_20260423_A_weighted_mw5_cho2017_cspnet_selheuristicweighted80combined.csv"),
        ("weighted_mw5_r100", RUNS_DIR / "sweep_20260423_A_weighted_mw5_cho2017"
         / "loso_results_20260423_A_weighted_mw5_cho2017_cspnet.csv"),
        ("weighted_mw7_r40", RUNS_DIR / "sweep_20260423_A_weighted_mw7_cho2017"
         / "loso_results_20260423_A_weighted_mw7_cho2017_cspnet_selheuristicweighted40combined.csv"),
        ("weighted_mw7_r60", RUNS_DIR / "sweep_20260423_A_weighted_mw7_cho2017"
         / "loso_results_20260423_A_weighted_mw7_cho2017_cspnet_selheuristicweighted60combined.csv"),
        ("weighted_mw7_r80", RUNS_DIR / "sweep_20260423_A_weighted_mw7_cho2017"
         / "loso_results_20260423_A_weighted_mw7_cho2017_cspnet_selheuristicweighted80combined.csv"),
        ("weighted_mw7_r100", RUNS_DIR / "sweep_20260423_A_weighted_mw7_cho2017"
         / "loso_results_20260423_A_weighted_mw7_cho2017_cspnet.csv"),
        ("qdiv_a4_r40", RUNS_DIR / "sweep_20260423_B_qdiv_a4_cho2017"
         / "loso_results_20260423_B_qdiv_a4_cho2017_cspnet_selheuristicweighted40quality_diversity.csv"),
        ("qdiv_a4_r60", RUNS_DIR / "sweep_20260423_B_qdiv_a4_cho2017"
         / "loso_results_20260423_B_qdiv_a4_cho2017_cspnet_selheuristicweighted60quality_diversity.csv"),
        ("qdiv_a4_r80", RUNS_DIR / "sweep_20260423_B_qdiv_a4_cho2017"
         / "loso_results_20260423_B_qdiv_a4_cho2017_cspnet_selheuristicweighted80quality_diversity.csv"),
        ("qdiv_a4_r100", RUNS_DIR / "sweep_20260423_B_qdiv_a4_cho2017"
         / "loso_results_20260423_B_qdiv_a4_cho2017_cspnet.csv"),
        ("unc_l2_r40", RUNS_DIR / "sweep_20260423_C_unc_l2_cho2017"
         / "loso_results_20260423_C_unc_l2_cho2017_cspnet_seluncertaintyweighted40quality_entropy.csv"),
        ("unc_l2_r60", RUNS_DIR / "sweep_20260423_C_unc_l2_cho2017"
         / "loso_results_20260423_C_unc_l2_cho2017_cspnet_seluncertaintyweighted60quality_entropy.csv"),
        ("unc_l2_r80", RUNS_DIR / "sweep_20260423_C_unc_l2_cho2017"
         / "loso_results_20260423_C_unc_l2_cho2017_cspnet_seluncertaintyweighted80quality_entropy.csv"),
        ("unc_l2_r100", RUNS_DIR / "sweep_20260423_C_unc_l2_cho2017"
         / "loso_results_20260423_C_unc_l2_cho2017_cspnet.csv"),
        ("phase2_unc_mw2_r80", RUNS_DIR / "sweep_20260506_U_qent_mw2_cho2017"
         / "loso_results_20260506_U_qent_mw2_cho2017_cspnet_seluncertaintyweighted80quality_entropy.csv"),
        ("phase2_unc_mw2_r100", RUNS_DIR / "sweep_20260506_U_qent_mw2_cho2017"
         / "loso_results_20260506_U_qent_mw2_cho2017_cspnet.csv"),
        ("phase2_unc_mw3_r80", RUNS_DIR / "sweep_20260506_U_qent_mw3_cho2017"
         / "loso_results_20260506_U_qent_mw3_cho2017_cspnet_seluncertaintyweighted80quality_entropy.csv"),
        ("phase2_unc_mw3_r100", RUNS_DIR / "sweep_20260506_U_qent_mw3_cho2017"
         / "loso_results_20260506_U_qent_mw3_cho2017_cspnet.csv"),
        ("phase2_unc_mw4_r80", RUNS_DIR / "sweep_20260506_U_qent_mw4_cho2017"
         / "loso_results_20260506_U_qent_mw4_cho2017_cspnet_seluncertaintyweighted80quality_entropy.csv"),
        ("phase2_unc_mw4_r100", RUNS_DIR / "sweep_20260506_U_qent_mw4_cho2017"
         / "loso_results_20260506_U_qent_mw4_cho2017_cspnet.csv"),
    ],
    "lee2019": [
        ("weighted_mw3_r40", RUNS_DIR / "sweep_20260423_A_weighted_mw3_lee2019"
         / "loso_results_20260423_A_weighted_mw3_lee2019_cspnet_selheuristicweighted40combined.csv"),
        ("weighted_mw3_r60", RUNS_DIR / "sweep_20260423_A_weighted_mw3_lee2019"
         / "loso_results_20260423_A_weighted_mw3_lee2019_cspnet_selheuristicweighted60combined.csv"),
        ("weighted_mw3_r80", RUNS_DIR / "sweep_20260423_A_weighted_mw3_lee2019"
         / "loso_results_20260423_A_weighted_mw3_lee2019_cspnet_selheuristicweighted80combined.csv"),
        ("weighted_mw3_r100", RUNS_DIR / "sweep_20260423_A_weighted_mw3_lee2019"
         / "loso_results_20260423_A_weighted_mw3_lee2019_cspnet.csv"),
        ("weighted_mw5_r40", RUNS_DIR / "sweep_20260423_A_weighted_mw5_lee2019"
         / "loso_results_20260423_A_weighted_mw5_lee2019_cspnet_selheuristicweighted40combined.csv"),
        ("weighted_mw5_r60", RUNS_DIR / "sweep_20260423_A_weighted_mw5_lee2019"
         / "loso_results_20260423_A_weighted_mw5_lee2019_cspnet_selheuristicweighted60combined.csv"),
        ("weighted_mw5_r80", RUNS_DIR / "sweep_20260423_A_weighted_mw5_lee2019"
         / "loso_results_20260423_A_weighted_mw5_lee2019_cspnet_selheuristicweighted80combined.csv"),
        ("weighted_mw5_r100", RUNS_DIR / "sweep_20260423_A_weighted_mw5_lee2019"
         / "loso_results_20260423_A_weighted_mw5_lee2019_cspnet.csv"),
        ("weighted_mw7_r40", RUNS_DIR / "sweep_20260423_A_weighted_mw7_lee2019"
         / "loso_results_20260423_A_weighted_mw7_lee2019_cspnet_selheuristicweighted40combined.csv"),
        ("weighted_mw7_r60", RUNS_DIR / "sweep_20260423_A_weighted_mw7_lee2019"
         / "loso_results_20260423_A_weighted_mw7_lee2019_cspnet_selheuristicweighted60combined.csv"),
        ("weighted_mw7_r80", RUNS_DIR / "sweep_20260423_A_weighted_mw7_lee2019"
         / "loso_results_20260423_A_weighted_mw7_lee2019_cspnet_selheuristicweighted80combined.csv"),
        ("weighted_mw7_r100", RUNS_DIR / "sweep_20260423_A_weighted_mw7_lee2019"
         / "loso_results_20260423_A_weighted_mw7_lee2019_cspnet.csv"),
        ("qdiv_a4_r40", RUNS_DIR / "sweep_20260423_B_qdiv_a4_lee2019"
         / "loso_results_20260423_B_qdiv_a4_lee2019_cspnet_selheuristicweighted40quality_diversity.csv"),
        ("qdiv_a4_r60", RUNS_DIR / "sweep_20260423_B_qdiv_a4_lee2019"
         / "loso_results_20260423_B_qdiv_a4_lee2019_cspnet_selheuristicweighted60quality_diversity.csv"),
        ("qdiv_a4_r80", RUNS_DIR / "sweep_20260423_B_qdiv_a4_lee2019"
         / "loso_results_20260423_B_qdiv_a4_lee2019_cspnet_selheuristicweighted80quality_diversity.csv"),
        ("qdiv_a4_r100", RUNS_DIR / "sweep_20260423_B_qdiv_a4_lee2019"
         / "loso_results_20260423_B_qdiv_a4_lee2019_cspnet.csv"),
        ("unc_l2_r40", RUNS_DIR / "sweep_20260423_C_unc_l2_lee2019"
         / "loso_results_20260423_C_unc_l2_lee2019_cspnet_seluncertaintyweighted40quality_entropy.csv"),
        ("unc_l2_r60", RUNS_DIR / "sweep_20260423_C_unc_l2_lee2019"
         / "loso_results_20260423_C_unc_l2_lee2019_cspnet_seluncertaintyweighted60quality_entropy.csv"),
        ("unc_l2_r80", RUNS_DIR / "sweep_20260423_C_unc_l2_lee2019"
         / "loso_results_20260423_C_unc_l2_lee2019_cspnet_seluncertaintyweighted80quality_entropy.csv"),
        ("unc_l2_r100", RUNS_DIR / "sweep_20260423_C_unc_l2_lee2019"
         / "loso_results_20260423_C_unc_l2_lee2019_cspnet.csv"),
    ],
}


def read_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cho2017"
    if dataset not in TABLES_BY_DATASET:
        raise SystemExit(f"Unknown dataset: {dataset}")

    out_csv = RUNS_DIR / f"subject_performance_tables_{dataset}_20260515.csv"
    tables = TABLES_BY_DATASET[dataset]
    rows_by_subject = {}
    for method_name, path in tables:
        for row in read_rows(path):
            subject = int(row["subject"])
            out = rows_by_subject.setdefault(subject, {"subject": subject})
            out[f"{method_name}_acc"] = float(row["acc"])
            out[f"{method_name}_bac"] = float(row["bac"])
            out[f"{method_name}_kappa"] = float(row["kappa"])
            out[f"{method_name}_best_epoch"] = int(row["best_epoch"])
            out[f"{method_name}_best_val_acc"] = float(row["best_val_acc"])
            out[f"{method_name}_best_val_loss"] = float(row["best_val_loss"])
            out[f"{method_name}_time_min"] = float(row["time_min"])

    fieldnames = ["subject"]
    for method_name, _ in tables:
        fieldnames.extend([
            f"{method_name}_acc",
            f"{method_name}_bac",
            f"{method_name}_kappa",
            f"{method_name}_best_epoch",
            f"{method_name}_best_val_acc",
            f"{method_name}_best_val_loss",
            f"{method_name}_time_min",
        ])

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for subject in sorted(rows_by_subject):
            writer.writerow(rows_by_subject[subject])

    print(str(out_csv))


if __name__ == "__main__":
    main()
