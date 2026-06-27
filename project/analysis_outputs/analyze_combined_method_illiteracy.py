from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DELTAS = Path("analysis_outputs/loso_paired_subject_deltas.csv")
OUT = Path("analysis_outputs/combined_method_illiteracy")
THRESHOLD = 70.0

METHOD_MAP = {
    "cspnet_noea": "Original",
    "ea_cspnet": "EA",
    "ea_adabn_cspnet": "EA+AdaBN",
    "ea_tent_cspnet": "EA+TENT",
    "ea_snapshot_adabn_cspnet": "EA+Snapshot+AdaBN",
    "ea_supcon_cspnet": "EA+SupCon",
    "ea_supcon_adabn_cspnet": "EA+SupCon+AdaBN",
    "ea_supcon_coral_cspnet": "EA+SupCon+CORAL",
}

METHOD_ORDER = [
    "Original",
    "EA",
    "EA+TENT",
    "EA+AdaBN",
    "EA+Snapshot+AdaBN",
    "EA+SupCon",
    "EA+SupCon+AdaBN",
    "EA+SupCon+CORAL",
]

COMBINED_METHODS = [
    "EA+Snapshot+AdaBN",
    "EA+SupCon",
    "EA+SupCon+AdaBN",
    "EA+SupCon+CORAL",
]


def summarize_binary(x: pd.Series) -> str:
    return f"{int(x.sum())}/{len(x)} ({x.mean() * 100:.1f}%)"


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    clean = df.replace({np.nan: ""})
    cols = list(clean.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_accuracy_table() -> pd.DataFrame:
    df = pd.read_csv(DELTAS)
    df["subject"] = df["subject"].astype(int)
    rows = []
    for _, r in df.iterrows():
        for col_method, col_acc in [("baseline", "baseline_acc"), ("candidate", "candidate_acc")]:
            method_id = str(r[col_method])
            method = METHOD_MAP.get(method_id)
            if method is None:
                continue
            rows.append(
                {
                    "dataset": r["dataset"],
                    "subject": int(r["subject"]),
                    "method_id": method_id,
                    "method": method,
                    "acc": float(r[col_acc]),
                }
            )
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(["dataset", "subject", "method", "acc"])
    out = out.groupby(["dataset", "subject", "method"], as_index=False).agg(acc=("acc", "mean"))
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    return out.sort_values(["dataset", "subject", "method"])


def method_effect_summary(wide: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for m in methods:
        for ill, label in [(True, "Original<70"), (False, "Original>=70")]:
            g = wide[wide["original_illiterate"].eq(ill)]
            rows.append(
                {
                    "baseline_group": label,
                    "method": m,
                    "n": len(g),
                    "mean_acc": round(float(g[m].mean()), 2),
                    "mean_delta_vs_original": round(float((g[m] - g["Original"]).mean()), 2),
                    "mean_delta_vs_ea": round(float((g[m] - g["EA"]).mean()), 2),
                    "pass70_rate_pct": round(float((g[m] >= THRESHOLD).mean() * 100), 1),
                    "responder_vs_original_pct": round(float((g[m] > g["Original"]).mean() * 100), 1),
                    "harm_vs_original_pct": round(float((g[m] < g["Original"]).mean() * 100), 1),
                    "responder_vs_ea_pct": round(float((g[m] > g["EA"]).mean() * 100), 1),
                    "harm_vs_ea_pct": round(float((g[m] < g["EA"]).mean() * 100), 1),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc = build_accuracy_table()
    wide = acc.pivot_table(index=["dataset", "subject"], columns="method", values="acc").reset_index()
    missing = [m for m in METHOD_ORDER if m not in wide.columns]
    if missing:
        raise ValueError(f"Missing method columns: {missing}")

    wide["original_illiterate"] = wide["Original"] < THRESHOLD
    for m in METHOD_ORDER:
        wide[f"{m}_pass70"] = wide[m] >= THRESHOLD
        wide[f"{m}_delta_original"] = wide[m] - wide["Original"]
        wide[f"{m}_delta_ea"] = wide[m] - wide["EA"]

    wide["any_combined_pass70"] = wide[[f"{m}_pass70" for m in COMBINED_METHODS]].any(axis=1)
    wide["all_combined_fail70"] = ~wide["any_combined_pass70"]
    wide["n_combined_pass70"] = wide[[f"{m}_pass70" for m in COMBINED_METHODS]].sum(axis=1)
    wide["best_combined_acc"] = wide[COMBINED_METHODS].max(axis=1)
    wide["best_combined_method"] = wide[COMBINED_METHODS].idxmax(axis=1)
    wide["best_combined_delta_original"] = wide["best_combined_acc"] - wide["Original"]
    wide["best_combined_delta_ea"] = wide["best_combined_acc"] - wide["EA"]

    group_rows = []
    for label, g in [
        ("Original<70", wide[wide["original_illiterate"]]),
        ("Original>=70", wide[~wide["original_illiterate"]]),
    ]:
        group_rows.append(
            {
                "group": label,
                "n_subjects": len(g),
                "mean_original": round(float(g["Original"].mean()), 2),
                "mean_ea": round(float(g["EA"].mean()), 2),
                "mean_best_combined": round(float(g["best_combined_acc"].mean()), 2),
                "mean_best_combined_delta_original": round(float(g["best_combined_delta_original"].mean()), 2),
                "mean_best_combined_delta_ea": round(float(g["best_combined_delta_ea"].mean()), 2),
                "any_combined_reaches_70": summarize_binary(g["any_combined_pass70"]),
                "persistent_fail_all_combined": summarize_binary(g["all_combined_fail70"]),
                "mean_n_combined_pass70": round(float(g["n_combined_pass70"].mean()), 2),
            }
        )
    group_summary = pd.DataFrame(group_rows)
    method_summary = method_effect_summary(wide, COMBINED_METHODS)

    persistent = wide[wide["original_illiterate"] & wide["all_combined_fail70"]].copy()
    persistent = persistent.sort_values(["Original", "best_combined_acc"])
    persistent_out = persistent[
        ["dataset", "subject", "Original", "EA", *COMBINED_METHODS, "best_combined_acc", "best_combined_method", "best_combined_delta_original", "best_combined_delta_ea"]
    ].round(2)

    recovered = wide[wide["original_illiterate"] & wide["any_combined_pass70"]].copy()
    recovered = recovered.sort_values("best_combined_delta_original", ascending=False)
    recovered_out = recovered[
        ["dataset", "subject", "Original", "EA", *COMBINED_METHODS, "best_combined_acc", "best_combined_method", "best_combined_delta_original", "best_combined_delta_ea", "n_combined_pass70"]
    ].round(2)

    acc.to_csv(OUT / "combined_method_subject_accuracy_long.csv", index=False)
    wide.to_csv(OUT / "combined_method_subject_accuracy_wide.csv", index=False)
    group_summary.to_csv(OUT / "combined_method_illiteracy_summary.csv", index=False)
    method_summary.to_csv(OUT / "combined_method_effect_summary.csv", index=False)
    persistent_out.to_csv(OUT / "combined_method_persistent_illiteracy_subjects.csv", index=False)
    recovered_out.to_csv(OUT / "combined_method_recovered_illiteracy_subjects.csv", index=False)

    lines = [
        "# Combined Methods: BCI Illiteracy vs Generalization Consistency",
        "",
        f"Operational threshold: accuracy < {THRESHOLD:.0f}%.",
        "Subject-level analysis is CSPNet-only because the available combination methods are CSPNet variants.",
        "",
        "## Combined Method Set",
        "",
        "- EA+Snapshot+AdaBN",
        "- EA+SupCon",
        "- EA+SupCon+AdaBN",
        "- EA+SupCon+CORAL",
        "",
        "## Summary",
        "",
        md_table(group_summary),
        "",
        "## Method Effects Within Original<70 Group",
        "",
        md_table(method_summary[method_summary["baseline_group"].eq("Original<70")]),
        "",
        "## Persistent Illiteracy Under All Combined Methods",
        "",
        md_table(persistent_out.head(40)),
        "",
        "## Recovered Illiteracy Under Combined Methods",
        "",
        md_table(recovered_out.head(40)),
        "",
        "## Interpretation",
        "",
        "- Combined methods can recover some Original<70 subjects, but a persistent low-performing subgroup remains.",
        "- Compare `best_combined_delta_ea` to judge whether combinations add benefit beyond plain EA.",
        "- If `mean_best_combined_delta_ea` is small, the main effect is still EA/alignment rather than the extra combined component.",
        "",
    ]
    (OUT / "combined_method_illiteracy_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
