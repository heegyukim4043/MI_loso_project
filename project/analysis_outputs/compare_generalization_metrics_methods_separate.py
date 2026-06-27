from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("analysis_outputs/generalization_methods_separate")
ROBUSTNESS = Path("analysis_outputs/loso_subject_robustness_metrics.csv")
DELTAS = Path("analysis_outputs/loso_paired_subject_deltas.csv")
TRANSFER_CORR = Path("analysis_outputs/transfer_benefit/transfer_benefit_loso_spearman.csv")

METHOD_ORDER = ["Original", "EA", "TENT", "AdaBN", "Snapshot"]
BACKBONE_ORDER = ["EEGNet", "CSPNet", "Conformer"]


def backbone_from_method(method: str) -> str:
    if "eegnet" in method:
        return "EEGNet"
    if "cspnet" in method:
        return "CSPNet"
    if "conformer" in method:
        return "Conformer"
    return "Unknown"


def final_method_from_id(method: str) -> str | None:
    if method in {"eegnet_noea", "cspnet_noea", "conformer_noea"}:
        return "Original"
    if method in {"ea_eegnet", "ea_cspnet", "ea_conformer"}:
        return "EA"
    if method in {"ea_tent_eegnet", "ea_tent_cspnet", "ea_tent_conformer"}:
        return "TENT"
    if method in {"ea_adabn_eegnet", "ea_adabn_cspnet", "ea_adabn_conformer"}:
        return "AdaBN"
    if method in {"ea_snapshot_eegnet", "ea_snapshot_adabn_cspnet", "ea_snapshot_conformer"}:
        return "Snapshot"
    return None


def load_robustness() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(ROBUSTNESS)
    df = df[df["backbone"].isin(BACKBONE_ORDER)].copy()
    df["method_final"] = df["method"].map(final_method_from_id)
    df = df[df["method_final"].isin(METHOD_ORDER)].copy()
    df["method_final"] = pd.Categorical(df["method_final"], METHOD_ORDER, ordered=True)
    df["backbone"] = pd.Categorical(df["backbone"], BACKBONE_ORDER, ordered=True)

    detail = df[
        [
            "method_final",
            "backbone",
            "dataset",
            "n",
            "mean_acc",
            "sd_acc",
            "median_acc",
            "p10_acc",
            "q1_acc",
            "min_acc",
            "coverage_ge_60_pct",
            "coverage_ge_70_pct",
        ]
    ].sort_values(["method_final", "backbone", "dataset"])

    rows = []
    for method, g in detail[detail["dataset"].eq("all")].groupby("method_final", observed=False):
        rows.append(
            {
                "method": str(method),
                "n_backbones": g["backbone"].nunique(),
                "mean_acc": round(float(g["mean_acc"].mean()), 2),
                "sd_acc": round(float(g["sd_acc"].mean()), 2),
                "median_acc": round(float(g["median_acc"].mean()), 2),
                "p10_acc": round(float(g["p10_acc"].mean()), 2),
                "q1_acc": round(float(g["q1_acc"].mean()), 2),
                "min_acc": round(float(g["min_acc"].mean()), 2),
                "coverage_ge_60_pct": round(float(g["coverage_ge_60_pct"].mean()), 2),
                "coverage_ge_70_pct": round(float(g["coverage_ge_70_pct"].mean()), 2),
            }
        )
    summary = pd.DataFrame(rows)
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    return summary.sort_values("method"), detail


def load_subject_accuracy_table() -> pd.DataFrame:
    deltas = pd.read_csv(DELTAS)
    deltas["subject"] = deltas["subject"].astype(int)
    rows = []
    for _, r in deltas.iterrows():
        base_method = final_method_from_id(str(r["baseline"]))
        cand_method = final_method_from_id(str(r["candidate"]))
        for method_id, method_final, acc_col in [
            (r["baseline"], base_method, "baseline_acc"),
            (r["candidate"], cand_method, "candidate_acc"),
        ]:
            if method_final is None:
                continue
            rows.append(
                {
                    "dataset": r["dataset"],
                    "subject": int(r["subject"]),
                    "backbone": backbone_from_method(str(method_id)),
                    "method": method_final,
                    "acc": float(r[acc_col]),
                }
            )
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(["dataset", "subject", "backbone", "method", "acc"])
    out = (
        out.groupby(["dataset", "subject", "backbone", "method"], as_index=False)
        .agg(acc=("acc", "mean"))
    )
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    out["backbone"] = pd.Categorical(out["backbone"], BACKBONE_ORDER, ordered=True)
    return out.sort_values(["method", "backbone", "dataset", "subject"])


def transfer_vs_reference(acc: pd.DataFrame, reference: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = acc[acc["method"].astype(str).eq(reference)].rename(columns={"acc": "reference_acc"})
    ref = ref[["dataset", "subject", "backbone", "reference_acc"]]
    cand = acc[~acc["method"].astype(str).eq(reference)].copy()
    merged = cand.merge(ref, on=["dataset", "subject", "backbone"], how="inner")
    merged["reference"] = reference
    merged["delta"] = merged["acc"] - merged["reference_acc"]

    rows = []
    for method, g in merged.groupby("method", observed=True):
        if len(g) == 0:
            continue
        rows.append(
            {
                "reference": reference,
                "method": str(method),
                "n_subject_backbone": len(g),
                "mean_delta": round(float(g["delta"].mean()), 2),
                "median_delta": round(float(g["delta"].median()), 2),
                "p10_delta": round(float(g["delta"].quantile(0.10)), 2),
                "min_delta": round(float(g["delta"].min()), 2),
                "max_delta": round(float(g["delta"].max()), 2),
                "responder_rate_pct": round(float((g["delta"] > 0).mean() * 100), 1),
                "large_benefit_ge5pp_pct": round(float((g["delta"] >= 5).mean() * 100), 1),
                "harm_rate_pct": round(float((g["delta"] < 0).mean() * 100), 1),
            }
        )
    summary = pd.DataFrame(rows)
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    return summary.sort_values("method"), merged.sort_values(["method", "backbone", "dataset", "subject"])


def riemannian_summary() -> pd.DataFrame:
    corr = pd.read_csv(TRANSFER_CORR)
    corr["method"] = corr["candidate"].map(final_method_from_id)
    corr = corr[corr["method"].isin(["EA", "TENT", "AdaBN", "Snapshot"])].copy()
    corr = corr[
        corr["feature"].isin(
            [
                "baseline_acc",
                "cov_condition_num",
                "source_pool_mean_dist",
                "source_pool_min_dist",
                "source_pool_knn3_dist",
                "source_pool_knn5_dist",
                "source_pool_knn10_dist",
                "source_pool_sim_weight",
            ]
        )
    ].dropna(subset=["rho"])
    corr["abs_rho"] = corr["rho"].abs()
    idx = corr.groupby(["method", "dataset"])["abs_rho"].idxmax()
    out = corr.loc[idx, ["method", "dataset", "baseline", "candidate", "feature", "n", "rho", "p"]].copy()
    out["rho"] = out["rho"].round(3)
    out["p"] = out["p"].round(4)
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    return out.sort_values(["method", "dataset"])


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    clean = df.replace({np.nan: ""})
    cols = list(clean.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(
    robustness_summary: pd.DataFrame,
    robustness_detail: pd.DataFrame,
    vs_original: pd.DataFrame,
    vs_ea: pd.DataFrame,
    riem: pd.DataFrame,
) -> None:
    lines = [
        "# Generalization Metrics by Separate Method",
        "",
        "Methods are treated as final methods: `Original`, `EA`, `TENT`, `AdaBN`, `Snapshot`.",
        "`TENT`, `AdaBN`, and `Snapshot` are not merged into EA in the tables below.",
        "",
        "## Subject Robustness / Coverage",
        "",
        "Backbone-averaged LOSO summary across EEGNet, CSPNet, and Conformer.",
        "",
        md_table(robustness_summary),
        "",
        "## Transfer Benefit vs Original",
        "",
        "Every method is compared directly against its own Original/NoEA backbone baseline for the same dataset and subject.",
        "",
        md_table(vs_original),
        "",
        "## Incremental Benefit vs EA",
        "",
        "Only TENT, AdaBN, and Snapshot are shown here. This asks whether the method improves beyond EA.",
        "",
        md_table(vs_ea[vs_ea["method"].astype(str).isin(["TENT", "AdaBN", "Snapshot"])]),
        "",
        "## Riemannian Benefit Predictor",
        "",
        "Strongest subject-level Spearman predictor per method and dataset.",
        "",
        md_table(riem),
        "",
        "## Notes",
        "",
        "- `Original` has no transfer-benefit delta because it is the reference baseline.",
        "- Calibration and risk-coverage still require trial-level probability/logit outputs; current aggregate files cannot compute them.",
        "- Generalization gap against cross-dataset remains a separate table because the available cross-dataset methods do not map one-to-one to all five LOSO methods.",
        "",
    ]
    (OUT / "generalization_metrics_by_separate_method.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    robustness_summary, robustness_detail = load_robustness()
    acc = load_subject_accuracy_table()
    vs_original, vs_original_detail = transfer_vs_reference(acc, "Original")
    vs_ea, vs_ea_detail = transfer_vs_reference(acc[acc["method"].astype(str).ne("Original")], "EA")
    riem = riemannian_summary()

    robustness_summary.to_csv(OUT / "separate_method_robustness_summary.csv", index=False)
    robustness_detail.to_csv(OUT / "separate_method_robustness_by_backbone_dataset.csv", index=False)
    acc.to_csv(OUT / "separate_method_subject_accuracy_long.csv", index=False)
    vs_original.to_csv(OUT / "separate_method_transfer_vs_original_summary.csv", index=False)
    vs_original_detail.to_csv(OUT / "separate_method_transfer_vs_original_subjects.csv", index=False)
    vs_ea.to_csv(OUT / "separate_method_incremental_vs_ea_summary.csv", index=False)
    vs_ea_detail.to_csv(OUT / "separate_method_incremental_vs_ea_subjects.csv", index=False)
    riem.to_csv(OUT / "separate_method_riemannian_predictor_summary.csv", index=False)
    write_report(robustness_summary, robustness_detail, vs_original, vs_ea, riem)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
