from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


IN = Path("analysis_outputs/generalization_methods_separate/separate_method_subject_accuracy_long.csv")
OUT = Path("analysis_outputs/illiteracy_generalization")
METHODS = ["Original", "EA", "TENT", "AdaBN", "Snapshot"]
ADAPTIVE = ["EA", "TENT", "AdaBN", "Snapshot"]
THRESHOLD = 70.0


def summarize_binary(x: pd.Series) -> str:
    return f"{int(x.sum())}/{len(x)} ({x.mean() * 100:.1f}%)"


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.replace({np.nan: ""}).iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN)

    wide = df.pivot_table(index=["dataset", "subject", "backbone"], columns="method", values="acc").reset_index()
    for m in METHODS:
        if m not in wide:
            raise ValueError(f"Missing method column: {m}")

    wide["original_illiterate"] = wide["Original"] < THRESHOLD
    for m in ADAPTIVE:
        wide[f"{m}_pass70"] = wide[m] >= THRESHOLD
        wide[f"{m}_delta"] = wide[m] - wide["Original"]
    wide["any_method_pass70"] = wide[[f"{m}_pass70" for m in ADAPTIVE]].any(axis=1)
    wide["all_methods_fail70"] = ~wide["any_method_pass70"]
    wide["n_methods_pass70"] = wide[[f"{m}_pass70" for m in ADAPTIVE]].sum(axis=1)
    wide["best_adaptive_acc"] = wide[ADAPTIVE].max(axis=1)
    wide["best_adaptive_method"] = wide[ADAPTIVE].idxmax(axis=1)
    wide["best_delta"] = wide["best_adaptive_acc"] - wide["Original"]

    rows = []
    for group_name, g in [
        ("Original<70", wide[wide["original_illiterate"]]),
        ("Original>=70", wide[~wide["original_illiterate"]]),
    ]:
        rows.append(
            {
                "group": group_name,
                "n_subject_backbone": len(g),
                "mean_original": round(g["Original"].mean(), 2),
                "mean_best_adaptive": round(g["best_adaptive_acc"].mean(), 2),
                "mean_best_delta": round(g["best_delta"].mean(), 2),
                "any_method_reaches_70": summarize_binary(g["any_method_pass70"]),
                "persistent_fail_all_methods": summarize_binary(g["all_methods_fail70"]),
                "mean_n_methods_pass70": round(g["n_methods_pass70"].mean(), 2),
            }
        )
    group_summary = pd.DataFrame(rows)

    method_rows = []
    for m in ADAPTIVE:
        for ill, label in [(True, "Original<70"), (False, "Original>=70")]:
            g = wide[wide["original_illiterate"].eq(ill)]
            method_rows.append(
                {
                    "baseline_group": label,
                    "method": m,
                    "n": len(g),
                    "mean_acc": round(g[m].mean(), 2),
                    "mean_delta": round(g[f"{m}_delta"].mean(), 2),
                    "pass70_rate_pct": round((g[m] >= THRESHOLD).mean() * 100, 1),
                    "responder_rate_pct": round((g[f"{m}_delta"] > 0).mean() * 100, 1),
                    "harm_rate_pct": round((g[f"{m}_delta"] < 0).mean() * 100, 1),
                }
            )
    method_summary = pd.DataFrame(method_rows)

    subject_mean = df.groupby(["dataset", "subject", "method"], as_index=False).agg(acc=("acc", "mean"))
    subj_wide = subject_mean.pivot_table(index=["dataset", "subject"], columns="method", values="acc").reset_index()
    subj_wide["original_illiterate"] = subj_wide["Original"] < THRESHOLD
    for m in ADAPTIVE:
        subj_wide[f"{m}_pass70"] = subj_wide[m] >= THRESHOLD
        subj_wide[f"{m}_delta"] = subj_wide[m] - subj_wide["Original"]
    subj_wide["any_method_pass70"] = subj_wide[[f"{m}_pass70" for m in ADAPTIVE]].any(axis=1)
    subj_wide["all_methods_fail70"] = ~subj_wide["any_method_pass70"]
    subj_wide["n_methods_pass70"] = subj_wide[[f"{m}_pass70" for m in ADAPTIVE]].sum(axis=1)
    subj_wide["best_adaptive_acc"] = subj_wide[ADAPTIVE].max(axis=1)
    subj_wide["best_adaptive_method"] = subj_wide[ADAPTIVE].idxmax(axis=1)
    subj_wide["best_delta"] = subj_wide["best_adaptive_acc"] - subj_wide["Original"]

    subj_rows = []
    for group_name, g in [
        ("Original<70 subject-mean", subj_wide[subj_wide["original_illiterate"]]),
        ("Original>=70 subject-mean", subj_wide[~subj_wide["original_illiterate"]]),
    ]:
        subj_rows.append(
            {
                "group": group_name,
                "n_subjects": len(g),
                "mean_original": round(g["Original"].mean(), 2),
                "mean_best_adaptive": round(g["best_adaptive_acc"].mean(), 2),
                "mean_best_delta": round(g["best_delta"].mean(), 2),
                "any_method_reaches_70": summarize_binary(g["any_method_pass70"]),
                "persistent_fail_all_methods": summarize_binary(g["all_methods_fail70"]),
                "mean_n_methods_pass70": round(g["n_methods_pass70"].mean(), 2),
            }
        )
    subject_summary = pd.DataFrame(subj_rows)

    persistent = subj_wide[subj_wide["original_illiterate"] & subj_wide["all_methods_fail70"]].copy()
    persistent = persistent.sort_values(["Original", "best_adaptive_acc"])
    persistent_out = persistent[
        ["dataset", "subject", "Original", "EA", "TENT", "AdaBN", "Snapshot", "best_adaptive_acc", "best_adaptive_method", "best_delta"]
    ].round(2)

    recovered = subj_wide[subj_wide["original_illiterate"] & subj_wide["any_method_pass70"]].copy()
    recovered = recovered.sort_values(["best_delta"], ascending=False)
    recovered_out = recovered[
        ["dataset", "subject", "Original", "EA", "TENT", "AdaBN", "Snapshot", "best_adaptive_acc", "best_adaptive_method", "best_delta", "n_methods_pass70"]
    ].round(2)

    wide.to_csv(OUT / "illiteracy_by_subject_backbone.csv", index=False)
    subj_wide.to_csv(OUT / "illiteracy_by_subject_mean.csv", index=False)
    group_summary.to_csv(OUT / "illiteracy_subject_backbone_summary.csv", index=False)
    method_summary.to_csv(OUT / "illiteracy_method_summary.csv", index=False)
    subject_summary.to_csv(OUT / "illiteracy_subject_mean_summary.csv", index=False)
    persistent_out.to_csv(OUT / "persistent_illiteracy_subjects.csv", index=False)
    recovered_out.to_csv(OUT / "recovered_illiteracy_subjects.csv", index=False)

    lines = [
        "# BCI Illiteracy vs Generalization Consistency",
        "",
        f"Operational threshold: accuracy < {THRESHOLD:.0f}% is treated as BCI-illiteracy / below practical criterion.",
        "",
        "## Subject-Backbone Level",
        "",
        "Each row is one dataset-subject-backbone case. This is the strictest view because a subject can be illiterate for one backbone but not another.",
        "",
        md_table(group_summary),
        "",
        "## Method Effects Within Original<70 Group",
        "",
        md_table(method_summary[method_summary["baseline_group"].eq("Original<70")]),
        "",
        "## Subject-Mean Level",
        "",
        "Each subject is averaged across EEGNet, CSPNet, and Conformer before thresholding.",
        "",
        md_table(subject_summary),
        "",
        "## Persistent Illiteracy Subjects",
        "",
        "Subject-mean Original<70 and all EA/TENT/AdaBN/Snapshot means still <70.",
        "",
        md_table(persistent_out.head(30)),
        "",
        "## Strongly Recovered Illiteracy Subjects",
        "",
        "Subject-mean Original<70 but at least one generalization method reaches >=70.",
        "",
        md_table(recovered_out.head(30)),
        "",
        "## Interpretation",
        "",
        "- There are persistent low-performing subjects, so BCI-illiteracy is not fully solved by current generalization methods.",
        "- But a substantial fraction of Original<70 cases crosses 70% after at least one method, meaning many apparent illiteracy cases are method/alignment dependent.",
        "- Report both persistent-fail rate and recovered rate; using only Original accuracy overstates fixed subject inability.",
        "",
    ]
    (OUT / "illiteracy_generalization_consistency_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
