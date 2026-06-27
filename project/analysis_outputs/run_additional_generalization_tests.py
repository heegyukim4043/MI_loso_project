from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import binomtest


OUT = Path("analysis_outputs/additional_generalization_tests")
NPZ_ROOT = Path("moabb_streamed_npz")
LOSO_ACC = Path("analysis_outputs/generalization_methods_separate/separate_method_subject_accuracy_long.csv")
CROSS_SUBJ = Path("crossdata/results/aggregated/crossdataset_subject_level_all.csv")
THRESHOLD = 70.0


CROSS_METHOD_FILES = {
    "CSP-LDA baseline": {
        ("cho2017", "lee2019"): "loso_results_20260604_cross_stdmi_csp_lda_cross_cho2017_to_lee2019_csp_lda.csv",
        ("lee2019", "cho2017"): "loso_results_20260604_cross_stdmi_csp_lda_cross_lee2019_to_cho2017_csp_lda.csv",
    },
    "DatasetEA+SubjectEA+CSP-LDA": {
        ("cho2017", "lee2019"): "loso_results_20260604_cross_stdmi_datasetea_ea_csp_lda_cross_cho2017_to_lee2019_csp_lda.csv",
        ("lee2019", "cho2017"): "loso_results_20260604_cross_stdmi_datasetea_ea_csp_lda_cross_lee2019_to_cho2017_csp_lda.csv",
    },
    "SessionEA+CSP-LDA": {
        ("cho2017", "lee2019"): "loso_results_20260613_sfreq100_sessionea_session_ea_cross_cho2017_to_lee2019_csp_lda.csv",
        ("lee2019", "cho2017"): "loso_results_20260613_sfreq100_sessionea_session_ea_cross_lee2019_to_cho2017_csp_lda.csv",
    },
    "DSA+SEA+SessionEA+SourceWeight+CSPNet": {
        ("cho2017", "lee2019"): "loso_results_20260615_sfreq100_datasetea_subjectea_sessionea_sourceweight_tau5_cross_cho2017_to_lee2019_cspnet.csv",
        ("lee2019", "cho2017"): "loso_results_20260615_sfreq100_datasetea_subjectea_sessionea_sourceweight_tau5_cross_lee2019_to_cho2017_cspnet.csv",
    },
}


def regularize(cov: np.ndarray) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    scale = float(np.trace(cov) / cov.shape[0])
    return cov + np.eye(cov.shape[0]) * max(scale, 1.0) * 1e-6


def invsqrtm_spd(cov: np.ndarray) -> np.ndarray:
    vals, vecs = eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def riemann_dist(a: np.ndarray, b: np.ndarray) -> float:
    vals = eigh(b, a, eigvals_only=True)
    vals = np.clip(vals, 1e-12, None)
    return float(np.linalg.norm(np.log(vals)))


def subject_from_path(path: Path) -> int:
    for part in path.parts[::-1]:
        if part.startswith("sub-"):
            return int(part.split("-", 1)[1])
    raise ValueError(path)


def trial_cov_mean(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - x.mean(axis=2, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", x, x, optimize=True) / max(x.shape[2] - 1, 1)
    traces = np.trace(covs, axis1=1, axis2=2)
    covs = covs / np.maximum(traces[:, None, None], 1e-12)
    return regularize(covs.mean(axis=0))


def load_loso_best() -> pd.DataFrame:
    acc = pd.read_csv(LOSO_ACC)
    subj = acc.groupby(["dataset", "subject", "method"], as_index=False).agg(acc=("acc", "mean"))
    best = subj.groupby(["dataset", "subject"], as_index=False).agg(
        loso_best_acc=("acc", "max"),
        loso_best_method=("method", lambda s: subj.loc[s.index, :].sort_values("acc", ascending=False).iloc[0]["method"]),
    )
    return best


def load_cross_selected() -> pd.DataFrame:
    df = pd.read_csv(CROSS_SUBJ)
    rows = []
    for method, mapping in CROSS_METHOD_FILES.items():
        for (train, test), source_file in mapping.items():
            sub = df[
                df["source_file"].eq(source_file)
                & df["train_dataset"].eq(train)
                & df["test_dataset"].eq(test)
            ]
            if sub.empty:
                raise ValueError(f"Missing cross rows: {method} {source_file}")
            for _, r in sub.iterrows():
                rows.append(
                    {
                        "train_dataset": train,
                        "test_dataset": test,
                        "direction": f"{train}->{test}",
                        "subject": int(r["subject"]),
                        "cross_method": method,
                        "cross_acc": float(r["acc"]),
                    }
                )
    long = pd.DataFrame(rows)
    best = long.groupby(["train_dataset", "test_dataset", "direction", "subject"], as_index=False).agg(
        cross_best_acc=("cross_acc", "max"),
        cross_best_method=("cross_method", lambda s: long.loc[s.index, :].sort_values("cross_acc", ascending=False).iloc[0]["cross_method"]),
    )
    return long, best


def bootstrap_cross_vs_loso(n_boot: int = 20000, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    loso = load_loso_best()
    _, cross_best = load_cross_selected()
    paired = cross_best.merge(
        loso.rename(columns={"dataset": "test_dataset"}),
        on=["test_dataset", "subject"],
        how="inner",
    )
    paired["diff_cross_minus_loso"] = paired["cross_best_acc"] - paired["loso_best_acc"]
    paired["cross_pass70"] = paired["cross_best_acc"] >= THRESHOLD
    paired["loso_pass70"] = paired["loso_best_acc"] >= THRESHOLD

    rng = np.random.default_rng(seed)
    rows = []
    for label, g in [("pooled", paired), *[(d, x) for d, x in paired.groupby("direction")]]:
        diffs = g["diff_cross_minus_loso"].to_numpy()
        boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
        rows.append(
            {
                "scope": label,
                "n": len(g),
                "mean_loso_best": round(float(g["loso_best_acc"].mean()), 3),
                "mean_cross_best": round(float(g["cross_best_acc"].mean()), 3),
                "mean_diff_cross_minus_loso": round(float(diffs.mean()), 3),
                "ci95_low": round(float(np.quantile(boots, 0.025)), 3),
                "ci95_high": round(float(np.quantile(boots, 0.975)), 3),
                "p_boot_cross_ge_loso": round(float((boots >= 0).mean()), 4),
                "cross_ge_loso_subject_pct": round(float((diffs >= 0).mean() * 100), 1),
            }
        )
    return paired, pd.DataFrame(rows)


def mcnemar_loso_vs_cross(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in [("pooled", paired), *[(d, x) for d, x in paired.groupby("direction")]]:
        loso = g["loso_pass70"].to_numpy()
        cross = g["cross_pass70"].to_numpy()
        cross_only = int((cross & ~loso).sum())
        loso_only = int((loso & ~cross).sum())
        both = int((cross & loso).sum())
        neither = int((~cross & ~loso).sum())
        discord = cross_only + loso_only
        p = float(binomtest(min(cross_only, loso_only), discord, 0.5).pvalue) if discord else np.nan
        rows.append(
            {
                "scope": label,
                "n": len(g),
                "both_pass": both,
                "neither_pass": neither,
                "cross_only_pass": cross_only,
                "loso_only_pass": loso_only,
                "mcnemar_p": p,
                "loso_coverage_pct": round(float(loso.mean() * 100), 1),
                "cross_coverage_pct": round(float(cross.mean() * 100), 1),
            }
        )
    return pd.DataFrame(rows)


def load_dataset_subject_covs() -> dict[str, dict[int, tuple[np.ndarray, list[str]]]]:
    out: dict[str, dict[int, tuple[np.ndarray, list[str]]]] = {}
    for dataset, subdir in {"cho2017": "cho", "lee2019": "lee"}.items():
        grouped: dict[int, list[Path]] = {}
        for path in sorted((NPZ_ROOT / subdir).glob("sub-*/*.npz")):
            grouped.setdefault(subject_from_path(path), []).append(path)
        out[dataset] = {}
        for subject, paths in grouped.items():
            cov_sum = None
            n_total = 0
            names_ref = None
            for path in paths:
                with np.load(path, allow_pickle=True) as npz:
                    x = np.asarray(npz["X"])
                    names = [str(c) for c in npz["ch_names"].tolist()]
                cov = trial_cov_mean(x)
                if cov_sum is None:
                    cov_sum = cov * x.shape[0]
                    names_ref = names
                else:
                    if names != names_ref:
                        raise ValueError(f"Channel mismatch {dataset} {subject}")
                    cov_sum += cov * x.shape[0]
                n_total += x.shape[0]
            assert cov_sum is not None and names_ref is not None
            out[dataset][subject] = (regularize(cov_sum / n_total), names_ref)
    return out


def subset_cov(cov: np.ndarray, names: list[str], common: list[str]) -> np.ndarray:
    idx = [names.index(ch) for ch in common]
    return regularize(cov[np.ix_(idx, idx)])


def riemannian_pre_post_datasetea() -> tuple[pd.DataFrame, pd.DataFrame]:
    covs = load_dataset_subject_covs()
    common = sorted(set(next(iter(covs["cho2017"].values()))[1]) & set(next(iter(covs["lee2019"].values()))[1]))
    subj_rows = []
    for dataset, subject_map in covs.items():
        for subject, (cov, names) in subject_map.items():
            subj_rows.append({"dataset": dataset, "subject": subject, "cov": subset_cov(cov, names, common)})

    subj = pd.DataFrame(subj_rows)
    centroids = {}
    invsqrt = {}
    for dataset, g in subj.groupby("dataset"):
        c = regularize(np.stack(g["cov"].to_numpy()).mean(axis=0))
        centroids[dataset] = c
        invsqrt[dataset] = invsqrtm_spd(c)

    pre_centroid_dist = riemann_dist(centroids["cho2017"], centroids["lee2019"])
    aligned_centroids = {}
    aligned_rows = []
    for dataset, g in subj.groupby("dataset"):
        aligned = []
        a = invsqrt[dataset]
        for _, r in g.iterrows():
            acov = regularize(a @ r["cov"] @ a.T)
            aligned.append(acov)
            aligned_rows.append({"dataset": dataset, "subject": r["subject"], "aligned_cov": acov})
        aligned_centroids[dataset] = regularize(np.stack(aligned).mean(axis=0))
    post_centroid_dist = riemann_dist(aligned_centroids["cho2017"], aligned_centroids["lee2019"])

    pair_rows = []
    cho = subj[subj["dataset"].eq("cho2017")]["cov"].to_list()
    lee = subj[subj["dataset"].eq("lee2019")]["cov"].to_list()
    cho_al = [r["aligned_cov"] for r in aligned_rows if r["dataset"] == "cho2017"]
    lee_al = [r["aligned_cov"] for r in aligned_rows if r["dataset"] == "lee2019"]
    pre_d = [riemann_dist(a, b) for a in cho for b in lee]
    post_d = [riemann_dist(a, b) for a in cho_al for b in lee_al]
    pair_summary = pd.DataFrame(
        [
            {
                "metric": "dataset_centroid_distance",
                "n_common_channels": len(common),
                "pre_datasetea": pre_centroid_dist,
                "post_datasetea": post_centroid_dist,
                "reduction_pct": (pre_centroid_dist - post_centroid_dist) / pre_centroid_dist * 100,
            },
            {
                "metric": "all_cross_subject_pair_distance_mean",
                "n_common_channels": len(common),
                "pre_datasetea": float(np.mean(pre_d)),
                "post_datasetea": float(np.mean(post_d)),
                "reduction_pct": (float(np.mean(pre_d)) - float(np.mean(post_d))) / float(np.mean(pre_d)) * 100,
            },
            {
                "metric": "all_cross_subject_pair_distance_median",
                "n_common_channels": len(common),
                "pre_datasetea": float(np.median(pre_d)),
                "post_datasetea": float(np.median(post_d)),
                "reduction_pct": (float(np.median(pre_d)) - float(np.median(post_d))) / float(np.median(pre_d)) * 100,
            },
        ]
    )
    pair_detail = pd.DataFrame({"pre_distance": pre_d, "post_distance": post_d})
    return pair_summary, pair_detail


def source_pool_scaling() -> pd.DataFrame:
    df = pd.read_csv(CROSS_SUBJ)
    rows = []
    for k in [10, 20, 30, 40]:
        for train, test in [("cho2017", "lee2019"), ("lee2019", "cho2017")]:
            token = f"source_select_k{k}_cross_{train}_to_{test}_csp_lda"
            sub = df[df["source_file"].str.contains(token, regex=False, na=False)]
            if sub.empty:
                continue
            rows.append(
                {
                    "family": "CSP-LDA source_select",
                    "source_pool_n": k,
                    "train_dataset": train,
                    "test_dataset": test,
                    "direction": f"{train}->{test}",
                    "n_subjects": len(sub),
                    "mean_acc": float(sub["acc"].mean()),
                    "coverage_ge70_pct": float((sub["acc"] >= THRESHOLD).mean() * 100),
                }
            )
    # all-source baselines for selected strong models.
    all_methods = {
        "CSP-LDA baseline all-source": {
            ("cho2017", "lee2019"): "loso_results_20260604_cross_stdmi_csp_lda_cross_cho2017_to_lee2019_csp_lda.csv",
            ("lee2019", "cho2017"): "loso_results_20260604_cross_stdmi_csp_lda_cross_lee2019_to_cho2017_csp_lda.csv",
        },
        "CSP-LDA DatasetEA+SubjectEA all-source": {
            ("cho2017", "lee2019"): "loso_results_20260604_cross_stdmi_datasetea_ea_csp_lda_cross_cho2017_to_lee2019_csp_lda.csv",
            ("lee2019", "cho2017"): "loso_results_20260604_cross_stdmi_datasetea_ea_csp_lda_cross_lee2019_to_cho2017_csp_lda.csv",
        },
        "CSPNet DSA+SEA+SessionEA+SourceWeight all-source": {
            ("cho2017", "lee2019"): "loso_results_20260615_sfreq100_datasetea_subjectea_sessionea_sourceweight_tau5_cross_cho2017_to_lee2019_cspnet.csv",
            ("lee2019", "cho2017"): "loso_results_20260615_sfreq100_datasetea_subjectea_sessionea_sourceweight_tau5_cross_lee2019_to_cho2017_cspnet.csv",
        },
    }
    for family, mapping in all_methods.items():
        for (train, test), source_file in mapping.items():
            sub = df[df["source_file"].eq(source_file)]
            if sub.empty:
                continue
            rows.append(
                {
                    "family": family,
                    "source_pool_n": "all",
                    "train_dataset": train,
                    "test_dataset": test,
                    "direction": f"{train}->{test}",
                    "n_subjects": len(sub),
                    "mean_acc": float(sub["acc"].mean()),
                    "coverage_ge70_pct": float((sub["acc"] >= THRESHOLD).mean() * 100),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["direction", "family", "source_pool_n"], key=lambda s: s.astype(str))


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.replace({np.nan: ""}).iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(boot: pd.DataFrame, mc: pd.DataFrame, rsum: pd.DataFrame, scaling: pd.DataFrame) -> None:
    lines = [
        "# Additional Generalization Tests",
        "",
        "## 1. Bootstrap CI for Cross >= LOSO",
        "",
        "Subject pairing: each cross-dataset test subject is compared with the same dataset/subject LOSO best result.",
        "",
        md_table(boot),
        "",
        "Interpretation: if the 95% CI for `cross - LOSO` includes 0, cross-dataset best is statistically comparable to LOSO best under this bootstrap. If the CI is entirely below 0, cross remains lower than LOSO.",
        "",
        "## 2. McNemar Test: LOSO Best vs Cross-Dataset Best",
        "",
        "Binary outcome: accuracy >= 70%.",
        "",
        md_table(mc),
        "",
        "## 3. Riemannian Distance Pre/Post DatasetEA",
        "",
        "Common Cho/Lee channels are used. DatasetEA is simulated by whitening each dataset with its own dataset mean covariance.",
        "",
        md_table(rsum),
        "",
        "## 4. Source Pool Scaling",
        "",
        md_table(scaling),
        "",
        "## 5. ECE / Reliability",
        "",
        "Not computed here. Current aggregate CSVs do not store trial-level predicted probabilities/logits. Re-inference with per-trial confidence output is required.",
        "",
    ]
    (OUT / "additional_generalization_tests_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paired, boot = bootstrap_cross_vs_loso()
    mc = mcnemar_loso_vs_cross(paired)
    rsum, rdetail = riemannian_pre_post_datasetea()
    scaling = source_pool_scaling()

    paired.to_csv(OUT / "cross_vs_loso_best_subject_pairs.csv", index=False)
    boot.to_csv(OUT / "bootstrap_cross_ge_loso.csv", index=False)
    mc.to_csv(OUT / "mcnemar_loso_best_vs_cross_best.csv", index=False)
    rsum.to_csv(OUT / "riemannian_pre_post_datasetea_summary.csv", index=False)
    rdetail.to_csv(OUT / "riemannian_pre_post_datasetea_pair_distances.csv", index=False)
    scaling.to_csv(OUT / "source_pool_scaling_summary.csv", index=False)
    write_report(boot, mc, rsum, scaling)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
