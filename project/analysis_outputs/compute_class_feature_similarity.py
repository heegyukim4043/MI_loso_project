from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import spearmanr


NPZ_ROOT = Path("moabb_streamed_npz")
ACC_LONG = Path("analysis_outputs/generalization_methods_separate/separate_method_subject_accuracy_long.csv")
OUT = Path("analysis_outputs/class_feature_similarity")
DATASETS = {"cho2017": "cho", "lee2019": "lee"}


def regularize(cov: np.ndarray) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    scale = float(np.trace(cov) / cov.shape[0])
    return cov + np.eye(cov.shape[0]) * max(scale, 1.0) * 1e-6


def trial_covariances(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - x.mean(axis=2, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", x, x, optimize=True) / max(x.shape[2] - 1, 1)
    traces = np.trace(covs, axis1=1, axis2=2)
    covs = covs / np.maximum(traces[:, None, None], 1e-12)
    return covs


def riemann_dist(a: np.ndarray, b: np.ndarray) -> float:
    vals = eigh(b, a, eigvals_only=True)
    vals = np.clip(vals, 1e-12, None)
    return float(np.linalg.norm(np.log(vals)))


def fit_csp_features(x: np.ndarray, y: np.ndarray, n_components: int = 6) -> np.ndarray:
    covs = trial_covariances(x)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels, got {classes}")
    c0 = regularize(covs[y == classes[0]].mean(axis=0))
    c1 = regularize(covs[y == classes[1]].mean(axis=0))
    vals, vecs = eigh(c0, regularize(c0 + c1))
    order = np.argsort(vals)
    half = n_components // 2
    picks = np.r_[order[:half], order[-half:]]
    w = vecs[:, picks]
    z = np.einsum("kc,nct->nkt", w.T, x.astype(np.float64), optimize=True)
    var = z.var(axis=2)
    var = var / np.maximum(var.sum(axis=1, keepdims=True), 1e-12)
    return np.log(np.maximum(var, 1e-12))


def feature_metrics(features: np.ndarray, y: np.ndarray) -> dict[str, float]:
    classes = np.unique(y)
    f0 = features[y == classes[0]]
    f1 = features[y == classes[1]]
    m0 = f0.mean(axis=0)
    m1 = f1.mean(axis=0)
    diff = m0 - m1
    centroid_dist = float(np.linalg.norm(diff))
    denom = float(np.linalg.norm(m0) * np.linalg.norm(m1))
    cosine = float(np.dot(m0, m1) / denom) if denom > 0 else np.nan
    within = float(f0.var(axis=0).sum() + f1.var(axis=0).sum())
    fisher = float(np.dot(diff, diff) / max(within, 1e-12))
    return {
        "csp_centroid_dist": centroid_dist,
        "csp_centroid_cosine": cosine,
        "csp_fisher_ratio": fisher,
    }


def subject_from_path(path: Path) -> int:
    for part in path.parts[::-1]:
        if part.startswith("sub-"):
            return int(part.split("-", 1)[1])
    raise ValueError(path)


def load_subject_trials(dataset: str, subdir: str) -> list[tuple[int, np.ndarray, np.ndarray, list[str]]]:
    grouped: dict[int, list[Path]] = {}
    for path in sorted((NPZ_ROOT / subdir).glob("sub-*/*.npz")):
        grouped.setdefault(subject_from_path(path), []).append(path)

    rows = []
    for subject, paths in sorted(grouped.items()):
        xs = []
        ys = []
        ch_names = None
        for path in paths:
            with np.load(path, allow_pickle=True) as npz:
                x = np.asarray(npz["X"])
                y = np.asarray(npz["y"])
                names = [str(c) for c in npz["ch_names"].tolist()]
            if ch_names is None:
                ch_names = names
            elif ch_names != names:
                raise ValueError(f"Channel mismatch for {dataset} subject {subject}")
            xs.append(x)
            ys.append(y)
        rows.append((subject, np.concatenate(xs), np.concatenate(ys), ch_names or []))
    return rows


def compute_similarity() -> pd.DataFrame:
    rows = []
    for dataset, subdir in DATASETS.items():
        for subject, x, y, ch_names in load_subject_trials(dataset, subdir):
            covs = trial_covariances(x)
            classes = np.unique(y)
            c0 = regularize(covs[y == classes[0]].mean(axis=0))
            c1 = regularize(covs[y == classes[1]].mean(axis=0))
            feats = fit_csp_features(x, y)
            metrics = feature_metrics(feats, y)
            rows.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "n_trials": len(y),
                    "n_channels": len(ch_names),
                    "class0": int(classes[0]),
                    "class1": int(classes[1]),
                    "class_cov_riemann_dist": riemann_dist(c0, c1),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def correlate_with_accuracy(sim: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ACC_LONG.exists():
        return pd.DataFrame(), pd.DataFrame()
    acc = pd.read_csv(ACC_LONG)
    subject_acc = acc.groupby(["dataset", "subject", "method"], as_index=False).agg(acc=("acc", "mean"))
    original = subject_acc[subject_acc["method"].eq("Original")].rename(columns={"acc": "original_acc"})
    best = subject_acc[subject_acc["method"].ne("Original")].groupby(["dataset", "subject"], as_index=False).agg(best_generalization_acc=("acc", "max"))
    merged = sim.merge(original[["dataset", "subject", "original_acc"]], on=["dataset", "subject"], how="left")
    merged = merged.merge(best, on=["dataset", "subject"], how="left")
    merged["best_delta"] = merged["best_generalization_acc"] - merged["original_acc"]
    merged["original_illiterate"] = merged["original_acc"] < 70
    merged["persistent_fail"] = merged["best_generalization_acc"] < 70

    features = ["class_cov_riemann_dist", "csp_centroid_dist", "csp_centroid_cosine", "csp_fisher_ratio"]
    targets = ["original_acc", "best_generalization_acc", "best_delta"]
    rows = []
    for dataset, g in merged.groupby("dataset"):
        for feat in features:
            for target in targets:
                sub = g[[feat, target]].dropna()
                if len(sub) < 4 or sub[feat].nunique() < 2 or sub[target].nunique() < 2:
                    rho, p = np.nan, np.nan
                else:
                    rho, p = spearmanr(sub[feat], sub[target])
                rows.append({"dataset": dataset, "feature": feat, "target": target, "n": len(sub), "rho": rho, "p": p})
    return merged, pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.replace({np.nan: ""}).iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(sim: pd.DataFrame, merged: pd.DataFrame, corr: pd.DataFrame) -> None:
    summary = sim.groupby("dataset").agg(
        n_subjects=("subject", "nunique"),
        mean_cov_dist=("class_cov_riemann_dist", "mean"),
        mean_csp_centroid_dist=("csp_centroid_dist", "mean"),
        mean_csp_cosine=("csp_centroid_cosine", "mean"),
        mean_csp_fisher=("csp_fisher_ratio", "mean"),
    ).reset_index()
    for col in summary.columns:
        if col != "dataset":
            summary[col] = summary[col].round(3)

    ill_summary = pd.DataFrame()
    if not merged.empty:
        ill_summary = merged.groupby(["dataset", "original_illiterate", "persistent_fail"]).agg(
            n=("subject", "count"),
            mean_original_acc=("original_acc", "mean"),
            mean_best_generalization_acc=("best_generalization_acc", "mean"),
            mean_cov_dist=("class_cov_riemann_dist", "mean"),
            mean_csp_centroid_dist=("csp_centroid_dist", "mean"),
            mean_csp_fisher=("csp_fisher_ratio", "mean"),
        ).reset_index()
        for c in ill_summary.columns:
            if c not in {"dataset", "original_illiterate", "persistent_fail", "n"}:
                ill_summary[c] = ill_summary[c].round(3)

    top_corr = corr.dropna(subset=["rho"]).copy()
    if not top_corr.empty:
        top_corr["abs_rho"] = top_corr["rho"].abs()
        top_corr = top_corr.sort_values("abs_rho", ascending=False).head(12).drop(columns=["abs_rho"])
        top_corr["rho"] = top_corr["rho"].round(3)
        top_corr["p"] = top_corr["p"].round(4)

    lines = [
        "# Class Feature Similarity Analysis",
        "",
        "Features are computed from streamed MOABB NPZ trials.",
        "",
        "Metrics:",
        "- `class_cov_riemann_dist`: Riemannian distance between class mean covariance matrices. Larger means class covariance patterns are less similar.",
        "- `csp_centroid_dist`: Euclidean distance between left/right class centroids in subject-level CSP log-variance feature space. Larger means better separation.",
        "- `csp_centroid_cosine`: cosine similarity between class centroids. Larger means more similar.",
        "- `csp_fisher_ratio`: between-class centroid distance normalized by within-class variance. Larger means better separability.",
        "",
        "## Dataset Summary",
        "",
        md_table(summary),
        "",
        "## Illiteracy/Persistent Failure Groups",
        "",
        md_table(ill_summary),
        "",
        "## Strongest Spearman Correlations",
        "",
        md_table(top_corr),
        "",
        "## Interpretation Guide",
        "",
        "- If class separability metrics correlate positively with `original_acc`, low-performing subjects likely have intrinsically less separable MI features.",
        "- If separability is low in persistent failures, that supports a persistent/physiological component beyond alignment.",
        "- If separability is reasonable but accuracy improves mainly after alignment, that supports recoverable processing/alignment failure.",
        "",
    ]
    (OUT / "class_feature_similarity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sim = compute_similarity()
    merged, corr = correlate_with_accuracy(sim)
    sim.to_csv(OUT / "class_feature_similarity_by_subject.csv", index=False)
    if not merged.empty:
        merged.to_csv(OUT / "class_feature_similarity_with_accuracy.csv", index=False)
    if not corr.empty:
        corr.to_csv(OUT / "class_feature_similarity_spearman.csv", index=False)
    write_report(sim, merged, corr)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
