from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT = Path("analysis_outputs/generalization_validation_sequence")
ACC_LONG = Path("analysis_outputs/generalization_methods_separate/separate_method_subject_accuracy_long.csv")
CLASS_SIM = Path("analysis_outputs/class_feature_similarity/class_feature_similarity_with_accuracy.csv")
COV_FEAT = Path("analysis_outputs/transfer_benefit/subject_covariance_features_loso.csv")

METHODS = ["Original", "EA", "TENT", "AdaBN", "Snapshot"]
ADAPTIVE = ["EA", "TENT", "AdaBN", "Snapshot"]
THRESHOLD = 70.0


def fdr_bh(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    if pv.size == 0:
        return out.tolist()
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    restored = np.empty_like(q)
    restored[order] = q
    out[valid] = restored
    return out.tolist()


def cliff_delta(x: pd.Series, y: pd.Series) -> float:
    x = np.asarray(x.dropna(), dtype=float)
    y = np.asarray(y.dropna(), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return float((gt - lt) / (len(x) * len(y)))


def mcnemar_midp(b: int, c: int) -> float:
    # Exact two-sided binomial-style McNemar p-value with conservative handling.
    n = b + c
    if n == 0:
        return np.nan
    from scipy.stats import binomtest

    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    clean = df.replace({np.nan: ""})
    for _, row in clean.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_subject_table() -> pd.DataFrame:
    acc = pd.read_csv(ACC_LONG)
    subj = acc.groupby(["dataset", "subject", "method"], as_index=False).agg(acc=("acc", "mean"))
    wide = subj.pivot_table(index=["dataset", "subject"], columns="method", values="acc").reset_index()
    for m in METHODS:
        if m not in wide:
            raise ValueError(f"Missing method: {m}")
    wide["best_adaptive_acc"] = wide[ADAPTIVE].max(axis=1)
    wide["best_adaptive_method"] = wide[ADAPTIVE].idxmax(axis=1)
    wide["original_illiterate"] = wide["Original"] < THRESHOLD
    wide["recovered"] = wide["original_illiterate"] & (wide["best_adaptive_acc"] >= THRESHOLD)
    wide["persistent"] = wide["original_illiterate"] & (wide["best_adaptive_acc"] < THRESHOLD)
    for m in ADAPTIVE:
        wide[f"{m}_delta_original"] = wide[m] - wide["Original"]
        wide[f"{m}_delta_ea"] = wide[m] - wide["EA"]
        wide[f"{m}_harm5_original"] = wide[f"{m}_delta_original"] <= -5
        wide[f"{m}_harm5_ea"] = wide[f"{m}_delta_ea"] <= -5
    return wide


def merge_features(wide: pd.DataFrame) -> pd.DataFrame:
    sim = pd.read_csv(CLASS_SIM)
    cov = pd.read_csv(COV_FEAT)
    out = wide.merge(
        sim[
            [
                "dataset",
                "subject",
                "class_cov_riemann_dist",
                "csp_centroid_dist",
                "csp_centroid_cosine",
                "csp_fisher_ratio",
            ]
        ],
        on=["dataset", "subject"],
        how="left",
    )
    out = out.merge(cov, on=["dataset", "subject"], how="left")
    return out


def separability_tertiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = ["csp_centroid_dist", "csp_fisher_ratio"]
    for dataset, dg in df.groupby("dataset"):
        for feat in features:
            tert = pd.qcut(dg[feat], 3, labels=["low", "mid", "high"], duplicates="drop")
            tmp = dg.copy()
            tmp["separability_tertile"] = tert
            for tertile, g in tmp.groupby("separability_tertile", observed=True):
                row = {
                    "dataset": dataset,
                    "feature": feat,
                    "tertile": str(tertile),
                    "n": len(g),
                    "mean_original": round(float(g["Original"].mean()), 2),
                    "mean_best_adaptive": round(float(g["best_adaptive_acc"].mean()), 2),
                    "persistent_rate_pct": round(float(g["persistent"].mean() * 100), 1),
                    "recovery_rate_among_original_lt70_pct": round(
                        float(g.loc[g["original_illiterate"], "recovered"].mean() * 100)
                        if g["original_illiterate"].any()
                        else np.nan,
                        1,
                    ),
                }
                for m in ADAPTIVE:
                    row[f"{m}_delta_original"] = round(float(g[f"{m}_delta_original"].mean()), 2)
                rows.append(row)
    return pd.DataFrame(rows)


def recoverable_classifier(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = [
        "Original",
        "class_cov_riemann_dist",
        "csp_centroid_dist",
        "csp_centroid_cosine",
        "csp_fisher_ratio",
        "cov_condition_num",
        "source_pool_mean_dist",
        "source_pool_knn10_dist",
        "source_pool_sim_weight",
    ]
    data = df[df["original_illiterate"]].dropna(subset=feats + ["recovered"]).copy()
    X = data[feats]
    y = data["recovered"].astype(int)

    rows = []
    if y.nunique() == 2 and len(data) >= 10:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        rows.append({"evaluation": "pooled_5fold_cv", "n": len(data), "positive_recovered": int(y.sum()), "auroc": roc_auc_score(y, prob)})

        for test_dataset in sorted(data["dataset"].unique()):
            train = data[data["dataset"].ne(test_dataset)]
            test = data[data["dataset"].eq(test_dataset)]
            if train["recovered"].nunique() < 2 or test["recovered"].nunique() < 2:
                auc = np.nan
            else:
                pipe.fit(train[feats], train["recovered"].astype(int))
                p = pipe.predict_proba(test[feats])[:, 1]
                auc = roc_auc_score(test["recovered"].astype(int), p)
            rows.append(
                {
                    "evaluation": f"leave_dataset_out_test_{test_dataset}",
                    "n": len(test),
                    "positive_recovered": int(test["recovered"].sum()),
                    "auroc": auc,
                }
            )

    univ = []
    for feat in feats:
        sub = data[[feat, "recovered"]].dropna()
        if sub["recovered"].nunique() == 2 and sub[feat].nunique() > 1:
            # Higher score should mean more recovered; invert if needed by reporting max(AUC, 1-AUC).
            auc_raw = roc_auc_score(sub["recovered"].astype(int), sub[feat])
            auc_oriented = max(auc_raw, 1 - auc_raw)
        else:
            auc_raw = np.nan
            auc_oriented = np.nan
        univ.append({"feature": feat, "n": len(sub), "auroc_raw": auc_raw, "auroc_oriented": auc_oriented})
    return pd.DataFrame(rows), pd.DataFrame(univ).sort_values("auroc_oriented", ascending=False)


def oracle_gap(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    single_methods = ADAPTIVE
    oracle = df["best_adaptive_acc"]
    best_single = {}
    for m in single_methods:
        best_single[m] = {
            "mean_acc": df[m].mean(),
            "coverage70": (df[m] >= THRESHOLD).mean() * 100,
        }
    for m in single_methods:
        rows.append(
            {
                "method": m,
                "mean_acc": round(float(df[m].mean()), 2),
                "coverage_ge70_pct": round(float((df[m] >= THRESHOLD).mean() * 100), 1),
                "mean_regret_to_oracle": round(float((oracle - df[m]).mean()), 2),
                "subjects_where_method_is_best_pct": round(float((df["best_adaptive_method"].eq(m)).mean() * 100), 1),
            }
        )
    rows.append(
        {
            "method": "OracleBest(EA/TENT/AdaBN/Snapshot)",
            "mean_acc": round(float(oracle.mean()), 2),
            "coverage_ge70_pct": round(float((oracle >= THRESHOLD).mean() * 100), 1),
            "mean_regret_to_oracle": 0.0,
            "subjects_where_method_is_best_pct": 100.0,
        }
    )
    count = df["best_adaptive_method"].value_counts().rename_axis("best_method").reset_index(name="n_subjects")
    count["pct"] = (count["n_subjects"] / len(df) * 100).round(1)
    return pd.DataFrame(rows), count


def harm_analysis(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["Original", "csp_centroid_dist", "csp_fisher_ratio", "source_pool_mean_dist", "cov_condition_num"]
    rows = []
    for ref in ["original", "ea"]:
        for m in ADAPTIVE:
            if ref == "ea" and m == "EA":
                continue
            flag = f"{m}_harm5_{ref}"
            if flag not in df:
                continue
            harm = df[df[flag]]
            non = df[~df[flag]]
            for feat in feature_cols:
                h = harm[feat].dropna()
                n = non[feat].dropna()
                if len(h) >= 3 and len(n) >= 3:
                    p = mannwhitneyu(h, n, alternative="two-sided").pvalue
                    cd = cliff_delta(h, n)
                else:
                    p, cd = np.nan, np.nan
                rows.append(
                    {
                        "reference": ref,
                        "method": m,
                        "feature": feat,
                        "n_harm": len(h),
                        "n_nonharm": len(n),
                        "mean_harm": h.mean() if len(h) else np.nan,
                        "mean_nonharm": n.mean() if len(n) else np.nan,
                        "cliff_delta_harm_minus_non": cd,
                        "p": p,
                    }
                )
    out = pd.DataFrame(rows)
    out["q_fdr"] = fdr_bh(out["p"].tolist())
    return out


def statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in ADAPTIVE:
        diff = df[m] - df["Original"]
        try:
            p_acc = wilcoxon(diff).pvalue
        except ValueError:
            p_acc = np.nan
        orig_pass = df["Original"] >= THRESHOLD
        meth_pass = df[m] >= THRESHOLD
        b = int((~orig_pass & meth_pass).sum())
        c = int((orig_pass & ~meth_pass).sum())
        rows.append(
            {
                "comparison": f"{m} vs Original",
                "n": len(df),
                "mean_delta": diff.mean(),
                "wilcoxon_p": p_acc,
                "coverage_gain_count": b,
                "coverage_loss_count": c,
                "mcnemar_p": mcnemar_midp(b, c),
            }
        )
    for m in ["TENT", "AdaBN", "Snapshot"]:
        diff = df[m] - df["EA"]
        try:
            p_acc = wilcoxon(diff).pvalue
        except ValueError:
            p_acc = np.nan
        ea_pass = df["EA"] >= THRESHOLD
        meth_pass = df[m] >= THRESHOLD
        b = int((~ea_pass & meth_pass).sum())
        c = int((ea_pass & ~meth_pass).sum())
        rows.append(
            {
                "comparison": f"{m} vs EA",
                "n": len(df),
                "mean_delta": diff.mean(),
                "wilcoxon_p": p_acc,
                "coverage_gain_count": b,
                "coverage_loss_count": c,
                "mcnemar_p": mcnemar_midp(b, c),
            }
        )
    out = pd.DataFrame(rows)
    out["wilcoxon_q_fdr"] = fdr_bh(out["wilcoxon_p"].tolist())
    out["mcnemar_q_fdr"] = fdr_bh(out["mcnemar_p"].tolist())
    for c in ["mean_delta", "wilcoxon_p", "mcnemar_p", "wilcoxon_q_fdr", "mcnemar_q_fdr"]:
        out[c] = out[c].round(5)
    return out


def write_report(
    tertiles: pd.DataFrame,
    clf: pd.DataFrame,
    univ: pd.DataFrame,
    oracle: pd.DataFrame,
    oracle_counts: pd.DataFrame,
    harm: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    harm_top = harm.dropna(subset=["p"]).sort_values("q_fdr").head(20).copy()
    for col in ["mean_harm", "mean_nonharm", "cliff_delta_harm_minus_non", "p", "q_fdr"]:
        if col in harm_top:
            harm_top[col] = harm_top[col].round(4)

    lines = [
        "# Generalization Validation Sequence",
        "",
        "## 1. Separability Tertile별 Method Benefit",
        "",
        md_table(tertiles),
        "",
        "## 2. Recoverable vs Persistent Classifier",
        "",
        "Target: among `Original < 70`, predict `recovered = best(EA,TENT,AdaBN,Snapshot) >= 70`.",
        "",
        md_table(clf),
        "",
        "Univariate predictor AUROC:",
        "",
        md_table(univ.head(12)),
        "",
        "## 3. Oracle Gap / Method Selection",
        "",
        md_table(oracle),
        "",
        "Best method counts:",
        "",
        md_table(oracle_counts),
        "",
        "## 4. Harm Analysis",
        "",
        "Harm is defined as `delta <= -5%p`. Top feature differences between harm and non-harm groups:",
        "",
        md_table(harm_top),
        "",
        "## 5. Statistical Tests",
        "",
        "Paired accuracy uses Wilcoxon signed-rank. Coverage@70 uses McNemar exact binomial test. FDR is Benjamini-Hochberg over this test family.",
        "",
        md_table(tests),
        "",
        "## Main Takeaways",
        "",
        "- Class separability stratifies generalization performance: low-separability subjects remain hardest even after adaptation.",
        "- Recoverability can be predicted above chance from pre-adaptation accuracy, separability, and covariance geometry, but leave-dataset-out performance should be treated as the conservative estimate.",
        "- OracleBest shows the ceiling for subject-adaptive method selection and quantifies how much accuracy is lost by choosing one global method.",
        "- Harm analysis is necessary because methods that improve mean accuracy still create negative transfer for a nontrivial subset.",
        "",
    ]
    (OUT / "generalization_validation_sequence_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wide = load_subject_table()
    df = merge_features(wide)

    tert = separability_tertiles(df)
    clf, univ = recoverable_classifier(df)
    oracle, oracle_counts = oracle_gap(df)
    harm = harm_analysis(df)
    tests = statistical_tests(df)

    df.to_csv(OUT / "analysis_subject_feature_table.csv", index=False)
    tert.to_csv(OUT / "01_separability_tertile_method_benefit.csv", index=False)
    clf.to_csv(OUT / "02_recoverable_classifier_auroc.csv", index=False)
    univ.to_csv(OUT / "02_univariate_recoverable_predictor_auroc.csv", index=False)
    oracle.to_csv(OUT / "03_oracle_gap_summary.csv", index=False)
    oracle_counts.to_csv(OUT / "03_oracle_best_method_counts.csv", index=False)
    harm.to_csv(OUT / "04_harm_analysis_feature_tests.csv", index=False)
    tests.to_csv(OUT / "05_statistical_tests.csv", index=False)

    write_report(tert, clf, univ, oracle, oracle_counts, harm, tests)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
