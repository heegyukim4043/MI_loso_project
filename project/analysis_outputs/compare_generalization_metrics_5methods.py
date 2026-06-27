from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("analysis_outputs/generalization_5method_comparison")
ROBUSTNESS = Path("analysis_outputs/loso_subject_robustness_metrics.csv")
DELTAS = Path("analysis_outputs/loso_paired_subject_deltas.csv")
TRANSFER_CORR = Path("analysis_outputs/transfer_benefit/transfer_benefit_loso_spearman.csv")
CROSS_COVERAGE = Path("analysis_outputs/coverage_threshold_metrics.csv")
CROSS_DELTA = Path("analysis_outputs/transfer_benefit/transfer_benefit_cross_features.csv")


METHOD_ORDER = ["No EA", "EA", "EA+TENT", "EA+AdaBN", "EA+Snapshot"]
METHOD_ALIASES = {
    "No EA": "NoEA",
    "EA": "EA",
    "EA+TENT": "TENT",
    "EA+AdaBN": "AdaBN",
    "EA+Snapshot": "Snapshot",
}


def _load_robustness() -> pd.DataFrame:
    df = pd.read_csv(ROBUSTNESS)
    df = df[df["backbone"].isin(["EEGNet", "CSPNet", "Conformer"])]
    df = df[df["gen_method"].isin(METHOD_ORDER)]
    df["method_family"] = pd.Categorical(df["gen_method"], METHOD_ORDER, ordered=True)
    return df


def _robustness_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "mean_acc",
        "sd_acc",
        "p10_acc",
        "q1_acc",
        "min_acc",
        "coverage_ge_60_pct",
        "coverage_ge_70_pct",
    ]
    for (dataset, method), g in df[df["dataset"].eq("all")].groupby(["dataset", "method_family"], observed=False):
        row = {"scope": "LOSO", "dataset": dataset, "method_family": str(method), "n_backbones": g["backbone"].nunique()}
        for m in metrics:
            row[m] = round(float(g[m].mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("method_family")


def _backbone_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "backbone",
        "gen_method",
        "dataset",
        "n",
        "mean_acc",
        "sd_acc",
        "p10_acc",
        "min_acc",
        "coverage_ge_60_pct",
        "coverage_ge_70_pct",
    ]
    out = df[df["dataset"].eq("all")][keep].copy()
    out["gen_method"] = pd.Categorical(out["gen_method"], METHOD_ORDER, ordered=True)
    return out.sort_values(["gen_method", "backbone"])


def _load_delta() -> pd.DataFrame:
    df = pd.read_csv(DELTAS)
    df["subject"] = df["subject"].astype(int)
    return (
        df.groupby(["group", "baseline", "candidate", "dataset", "subject"], as_index=False)
        .agg(baseline_acc=("baseline_acc", "mean"), candidate_acc=("candidate_acc", "mean"), delta=("delta", "mean"))
    )


def _candidate_family(candidate: str) -> str | None:
    if candidate.startswith("ea_tent_"):
        return "EA+TENT"
    if candidate.startswith("ea_adabn_"):
        return "EA+AdaBN"
    if candidate.startswith("ea_snapshot_"):
        return "EA+Snapshot"
    if candidate in {"ea_eegnet", "ea_cspnet", "ea_conformer"}:
        return "EA"
    return None


def _backbone_from_method(method: str) -> str:
    if "eegnet" in method:
        return "EEGNet"
    if "cspnet" in method:
        return "CSPNet"
    if "conformer" in method:
        return "Conformer"
    return "Unknown"


def _transfer_metrics(delta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (group, baseline, candidate, dataset), g in delta.groupby(["group", "baseline", "candidate", "dataset"]):
        if group not in {"NoEA_to_EA", "EA_to_TTA", "EA_to_Snapshot"}:
            continue
        family = _candidate_family(candidate)
        if family not in METHOD_ORDER:
            continue
        if dataset != "all":
            pass
        rows.append(
            {
                "comparison_group": group,
                "baseline": baseline,
                "candidate": candidate,
                "backbone": _backbone_from_method(candidate),
                "candidate_family": family,
                "dataset": dataset,
                "n": len(g),
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
    detail = pd.DataFrame(rows)

    all_rows = []
    detail_for_all = detail.copy()
    detail_for_all["dataset"] = "all"
    for (family, dataset), g in detail_for_all.groupby(["candidate_family", "dataset"]):
        all_rows.append(
            {
                "scope": "LOSO",
                "delta_reference": "NoEA for EA; EA for TENT/AdaBN/Snapshot",
                "method_family": family,
                "dataset": dataset,
                "n_backbones": g["backbone"].nunique(),
                "mean_delta": round(float(g["mean_delta"].mean()), 2),
                "median_delta": round(float(g["median_delta"].mean()), 2),
                "p10_delta": round(float(g["p10_delta"].mean()), 2),
                "responder_rate_pct": round(float(g["responder_rate_pct"].mean()), 1),
                "large_benefit_ge5pp_pct": round(float(g["large_benefit_ge5pp_pct"].mean()), 1),
                "harm_rate_pct": round(float(g["harm_rate_pct"].mean()), 1),
            }
        )
    summary = pd.DataFrame(all_rows)
    summary["method_family"] = pd.Categorical(summary["method_family"], METHOD_ORDER, ordered=True)
    return summary.sort_values("method_family"), detail.sort_values(["candidate_family", "backbone", "dataset"])


def _riemannian_predictor_summary() -> pd.DataFrame:
    corr = pd.read_csv(TRANSFER_CORR)
    features = [
        "cov_condition_num",
        "source_pool_mean_dist",
        "source_pool_min_dist",
        "source_pool_knn3_dist",
        "source_pool_knn5_dist",
        "source_pool_knn10_dist",
        "source_pool_sim_weight",
        "baseline_acc",
    ]
    corr = corr[corr["feature"].isin(features)].dropna(subset=["rho"])
    corr["candidate_family"] = corr["candidate"].map(_candidate_family)
    corr = corr[corr["candidate_family"].isin(METHOD_ORDER)]
    corr["abs_rho"] = corr["rho"].abs()
    idx = corr.groupby(["candidate_family", "dataset"])["abs_rho"].idxmax()
    out = corr.loc[idx, ["candidate_family", "dataset", "baseline", "candidate", "feature", "n", "rho", "p"]].copy()
    out["rho"] = out["rho"].round(3)
    out["p"] = out["p"].round(4)
    out["interpretation"] = np.where(
        out["feature"].str.contains("dist") & (out["rho"] > 0),
        "larger covariance distance predicts larger benefit",
        np.where(
            out["feature"].str.contains("sim_weight") & (out["rho"] < 0),
            "lower source similarity predicts larger benefit",
            np.where(
                out["feature"].eq("baseline_acc") & (out["rho"] < 0),
                "weaker baseline subjects benefit more",
                "feature predicts subject-level benefit",
            ),
        ),
    )
    out["candidate_family"] = pd.Categorical(out["candidate_family"], METHOD_ORDER, ordered=True)
    return out.sort_values(["candidate_family", "dataset"])


def _crossdataset_gap_summary() -> pd.DataFrame:
    rows = []
    if CROSS_COVERAGE.exists():
        cov = pd.read_csv(CROSS_COVERAGE)
        cov = cov[cov["scope"].eq("CrossDataset")].copy()
        for _, r in cov.iterrows():
            rows.append(
                {
                    "scope": "CrossDataset coverage",
                    "method": r["method"],
                    "direction": r["dataset"],
                    "n": int(r["n"]),
                    "mean_acc": round(float(r["mean_acc"]), 2),
                    "coverage_ge70_pct": round(float(r["ge70_pct"]), 1),
                }
            )
    out = pd.DataFrame(rows)

    if CROSS_DELTA.exists():
        d = pd.read_csv(CROSS_DELTA)
        delta_summary = (
            d.groupby(["comparison", "train_dataset", "test_dataset"])
            .agg(
                n=("delta", "size"),
                mean_delta=("delta", "mean"),
                median_delta=("delta", "median"),
                responder_rate_pct=("delta", lambda x: (x > 0).mean() * 100),
                harm_rate_pct=("delta", lambda x: (x < 0).mean() * 100),
            )
            .reset_index()
        )
        delta_summary["mean_delta"] = delta_summary["mean_delta"].round(2)
        delta_summary["median_delta"] = delta_summary["median_delta"].round(2)
        delta_summary["responder_rate_pct"] = delta_summary["responder_rate_pct"].round(1)
        delta_summary["harm_rate_pct"] = delta_summary["harm_rate_pct"].round(1)
        delta_summary.to_csv(OUT / "crossdataset_transfer_delta_summary.csv", index=False)

    return out


def _availability_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric_family": "Subject robustness / coverage",
                "availability": "Directly available",
                "files": "loso_subject_robustness_metrics.csv, coverage_threshold_metrics.csv",
                "use_in_paper": "Main table: mean, SD, P10, worst, >=70% coverage",
            },
            {
                "metric_family": "Transfer benefit metrics",
                "availability": "Directly available",
                "files": "loso_paired_subject_deltas.csv",
                "use_in_paper": "Responder, >=5pp benefit, harm rate, delta distribution",
            },
            {
                "metric_family": "Riemannian domain-gap / predictor",
                "availability": "Available as predictor, not as full pre/post gap for every method",
                "files": "transfer_benefit_loso_spearman.csv",
                "use_in_paper": "Covariance geometry predicts which subjects benefit",
            },
            {
                "metric_family": "Generalization gap",
                "availability": "Partial",
                "files": "coverage_threshold_metrics.csv, crossdataset transfer outputs",
                "use_in_paper": "LOSO vs cross-dataset gap is available; within-subject/session gap requires matching results",
            },
            {
                "metric_family": "Calibration / risk-coverage",
                "availability": "Not available from current aggregate CSV",
                "files": "Requires trial-level probability/logit/confidence outputs",
                "use_in_paper": "Future/optional analysis unless inference probabilities are regenerated",
            },
        ]
    )


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    clean = df.copy()
    clean = clean.replace({np.nan: ""})
    cols = list(clean.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in clean.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    robustness_summary: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    riem: pd.DataFrame,
    cross: pd.DataFrame,
    availability: pd.DataFrame,
) -> None:
    lines = [
        "# Five-Method Generalization Metric Comparison",
        "",
        "Compared LOSO method families: `No EA`, `EA`, `EA+TENT`, `EA+AdaBN`, `EA+Snapshot`.",
        "",
        "## 1. Metric Availability",
        "",
        _md_table(availability),
        "",
        "## 2. Subject Robustness / Coverage",
        "",
        "Backbone-averaged LOSO summary across EEGNet, CSPNet, and Conformer.",
        "",
        _md_table(robustness_summary),
        "",
        "Interpretation: the strongest deployment-oriented readout is `coverage_ge_70_pct`, because it approximates how many subjects pass a practical MI-BCI threshold.",
        "",
        "## 3. Transfer Benefit Metrics",
        "",
        "`EA` is compared against `NoEA`. `EA+TENT`, `EA+AdaBN`, and `EA+Snapshot` are compared against their corresponding `EA` baseline.",
        "",
        _md_table(transfer_summary),
        "",
        "Interpretation: EA gives the dominant transfer benefit. TENT/AdaBN/Snapshot are smaller incremental effects on top of EA and should be reported with harm rate, not only mean delta.",
        "",
        "## 4. Riemannian Domain-Gap / Benefit Predictor",
        "",
        "This table lists the strongest subject-level Spearman predictor per method family and dataset.",
        "",
        _md_table(riem),
        "",
        "Interpretation: covariance geometry is useful as a vulnerability/benefit predictor. For EA, larger source-pool distance often predicts larger benefit, so the safer claim is not 'similar sources always help' but 'source-target covariance geometry predicts alignment benefit'.",
        "",
        "## 5. Generalization Gap",
        "",
        "Current direct evidence is strongest for LOSO vs cross-dataset degradation/recovery. Within-subject or within-session gaps require matched within-subject/session result files.",
        "",
        _md_table(cross),
        "",
        "## 6. Calibration / Risk-Coverage",
        "",
        "Not comparable from the current aggregate result files because trial-level confidence, probability, or logits are not stored. To compute ECE, Brier score, reliability diagrams, and risk-coverage curves, rerun inference while saving per-trial predicted probability and correctness.",
        "",
    ]
    (OUT / "five_method_generalization_metric_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    robustness = _load_robustness()
    robustness_summary = _robustness_summary(robustness)
    backbone_table = _backbone_table(robustness)

    delta = _load_delta()
    transfer_summary, transfer_detail = _transfer_metrics(delta)
    riem = _riemannian_predictor_summary()
    cross = _crossdataset_gap_summary()
    availability = _availability_table()

    robustness_summary.to_csv(OUT / "loso_5method_robustness_summary.csv", index=False)
    backbone_table.to_csv(OUT / "loso_5method_robustness_by_backbone.csv", index=False)
    transfer_summary.to_csv(OUT / "loso_5method_transfer_benefit_summary.csv", index=False)
    transfer_detail.to_csv(OUT / "loso_5method_transfer_benefit_by_backbone_dataset.csv", index=False)
    riem.to_csv(OUT / "loso_5method_riemannian_predictor_summary.csv", index=False)
    cross.to_csv(OUT / "crossdataset_generalization_gap_coverage_summary.csv", index=False)
    availability.to_csv(OUT / "generalization_metric_availability.csv", index=False)

    write_report(robustness_summary, transfer_summary, riem, cross, availability)

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
