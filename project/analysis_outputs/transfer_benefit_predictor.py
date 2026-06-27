from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import spearmanr


DATASET_DIRS = {
    "cho2017": "cho",
    "lee2019": "lee",
}


@dataclass
class SubjectCov:
    dataset: str
    subject: int
    ch_names: list[str]
    cov_full: np.ndarray
    cond_full: float
    n_trials: int


def _regularize_cov(cov: np.ndarray) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    scale = float(np.trace(cov) / cov.shape[0])
    eps = max(scale, 1.0) * 1e-6
    return cov + np.eye(cov.shape[0], dtype=cov.dtype) * eps


def _trial_cov_mean(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - x.mean(axis=2, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", x, x, optimize=True) / max(x.shape[2] - 1, 1)
    return _regularize_cov(covs.mean(axis=0))


def _riemann_dist(a: np.ndarray, b: np.ndarray) -> float:
    vals = eigh(b, a, eigvals_only=True)
    vals = np.clip(vals, 1e-12, None)
    return float(np.linalg.norm(np.log(vals)))


def _extract_subject(path: Path) -> int:
    for part in path.parts[::-1]:
        if part.startswith("sub-"):
            return int(part.split("-", 1)[1])
    raise ValueError(f"Cannot infer subject from {path}")


def load_subject_covariances(npz_root: Path) -> dict[tuple[str, int], SubjectCov]:
    rows: dict[tuple[str, int], list[Path]] = {}
    for dataset, subdir in DATASET_DIRS.items():
        for path in sorted((npz_root / subdir).glob("sub-*/*.npz")):
            rows.setdefault((dataset, _extract_subject(path)), []).append(path)

    out: dict[tuple[str, int], SubjectCov] = {}
    for (dataset, subject), paths in sorted(rows.items()):
        cov_sum = None
        n_total = 0
        ch_names: list[str] | None = None
        for path in paths:
            with np.load(path, allow_pickle=True) as npz:
                x = np.asarray(npz["X"])
                names = [str(c) for c in npz["ch_names"].tolist()]
            cov = _trial_cov_mean(x)
            if cov_sum is None:
                cov_sum = cov * x.shape[0]
                ch_names = names
            else:
                if names != ch_names:
                    raise ValueError(f"Channel mismatch within subject {dataset} {subject}: {path}")
                cov_sum += cov * x.shape[0]
            n_total += x.shape[0]
        assert cov_sum is not None and ch_names is not None
        cov_full = _regularize_cov(cov_sum / n_total)
        out[(dataset, subject)] = SubjectCov(
            dataset=dataset,
            subject=subject,
            ch_names=ch_names,
            cov_full=cov_full,
            cond_full=float(np.linalg.cond(cov_full)),
            n_trials=n_total,
        )
    return out


def subset_cov(sc: SubjectCov, channels: list[str]) -> np.ndarray:
    idx = [sc.ch_names.index(ch) for ch in channels]
    return _regularize_cov(sc.cov_full[np.ix_(idx, idx)])


def summarize_distances(distances: list[float], prefix: str) -> dict[str, float]:
    arr = np.asarray(distances, dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_mean_dist": np.nan,
            f"{prefix}_min_dist": np.nan,
            f"{prefix}_knn3_dist": np.nan,
            f"{prefix}_knn5_dist": np.nan,
            f"{prefix}_knn10_dist": np.nan,
            f"{prefix}_sim_weight": np.nan,
        }
    return {
        f"{prefix}_mean_dist": float(arr.mean()),
        f"{prefix}_min_dist": float(arr.min()),
        f"{prefix}_knn3_dist": float(np.sort(arr)[: min(3, arr.size)].mean()),
        f"{prefix}_knn5_dist": float(np.sort(arr)[: min(5, arr.size)].mean()),
        f"{prefix}_knn10_dist": float(np.sort(arr)[: min(10, arr.size)].mean()),
        f"{prefix}_sim_weight": float(np.exp(-arr).sum()),
    }


def compute_within_dataset_features(covs: dict[tuple[str, int], SubjectCov]) -> pd.DataFrame:
    rows = []
    for (dataset, subject), sc in covs.items():
        others = [
            oc.cov_full
            for (odataset, osubject), oc in covs.items()
            if odataset == dataset and osubject != subject
        ]
        distances = [_riemann_dist(sc.cov_full, other) for other in others]
        row = {
            "dataset": dataset,
            "subject": subject,
            "n_trials": sc.n_trials,
            "cov_condition_num": sc.cond_full,
        }
        row.update(summarize_distances(distances, "source_pool"))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_cross_dataset_features(covs: dict[tuple[str, int], SubjectCov]) -> pd.DataFrame:
    by_dataset: dict[str, list[SubjectCov]] = {}
    for sc in covs.values():
        by_dataset.setdefault(sc.dataset, []).append(sc)

    common_channels = sorted(set(by_dataset["cho2017"][0].ch_names) & set(by_dataset["lee2019"][0].ch_names))
    reduced = {
        (sc.dataset, sc.subject): subset_cov(sc, common_channels)
        for sc in covs.values()
    }

    rows = []
    for train_dataset, source_subjects in by_dataset.items():
        for test_dataset, test_subjects in by_dataset.items():
            if train_dataset == test_dataset:
                continue
            for test_sc in test_subjects:
                test_cov = reduced[(test_dataset, test_sc.subject)]
                distances = [
                    _riemann_dist(test_cov, reduced[(train_dataset, src.subject)])
                    for src in source_subjects
                ]
                row = {
                    "train_dataset": train_dataset,
                    "test_dataset": test_dataset,
                    "subject": test_sc.subject,
                    "common_n_channels": len(common_channels),
                }
                row.update(summarize_distances(distances, "source_dataset"))
                rows.append(row)
    return pd.DataFrame(rows)


def spearman_table(df: pd.DataFrame, group_cols: list[str], target: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for feat in features:
            sub = g[[feat, target]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 4 or sub[feat].nunique() < 2 or sub[target].nunique() < 2:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(sub[feat], sub[target])
            rows.append({**base, "feature": feat, "n": len(sub), "rho": rho, "p": p})
    return pd.DataFrame(rows)


def build_loso(
    deltas_path: Path,
    within_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deltas = pd.read_csv(deltas_path)
    deltas["subject"] = deltas["subject"].astype(int)
    deltas = (
        deltas.groupby(["group", "baseline", "candidate", "dataset", "subject"], as_index=False)
        .agg(baseline_acc=("baseline_acc", "mean"), candidate_acc=("candidate_acc", "mean"), delta=("delta", "mean"))
    )
    merged = deltas.merge(within_features, on=["dataset", "subject"], how="left", validate="many_to_one")
    features = [
        "baseline_acc",
        "cov_condition_num",
        "source_pool_mean_dist",
        "source_pool_min_dist",
        "source_pool_knn3_dist",
        "source_pool_knn5_dist",
        "source_pool_knn10_dist",
        "source_pool_sim_weight",
    ]
    corr = spearman_table(
        merged,
        ["group", "baseline", "candidate", "dataset"],
        "delta",
        features,
    )
    return merged, corr


def _select_cross_rows(df: pd.DataFrame, source_token: str) -> pd.DataFrame:
    mask = df["source_file"].str.contains(source_token, regex=False) & df["model"].eq("csp_lda")
    out = df.loc[mask, ["train_dataset", "test_dataset", "subject", "acc", "bac", "source_file"]].copy()
    out["subject"] = out["subject"].astype(int)
    return out


def build_cross(
    cross_path: Path,
    cross_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(cross_path)
    pairs = [
        (
            "CrossCSP_to_SubjectEA",
            "cross_stdmi_csp_lda_cross",
            "cross_stdmi_ea_csp_lda_cross",
        ),
        (
            "CrossCSP_to_DatasetEA_SubjectEA",
            "cross_stdmi_csp_lda_cross",
            "cross_stdmi_datasetea_ea_csp_lda_cross",
        ),
    ]

    rows = []
    for comparison, base_token, cand_token in pairs:
        base = _select_cross_rows(df, base_token).rename(
            columns={"acc": "baseline_acc", "bac": "baseline_bac", "source_file": "baseline_file"}
        )
        cand = _select_cross_rows(df, cand_token).rename(
            columns={"acc": "candidate_acc", "bac": "candidate_bac", "source_file": "candidate_file"}
        )
        merged = base.merge(cand, on=["train_dataset", "test_dataset", "subject"], how="inner")
        merged["comparison"] = comparison
        merged["delta"] = merged["candidate_acc"] - merged["baseline_acc"]
        rows.append(merged)

    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame()
    out = out.merge(cross_features, on=["train_dataset", "test_dataset", "subject"], how="left")

    features = [
        "baseline_acc",
        "source_dataset_mean_dist",
        "source_dataset_min_dist",
        "source_dataset_knn3_dist",
        "source_dataset_knn5_dist",
        "source_dataset_knn10_dist",
        "source_dataset_sim_weight",
    ]
    corr = spearman_table(
        out,
        ["comparison", "train_dataset", "test_dataset"],
        "delta",
        features,
    )
    return out, corr


def _format_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _top_table(corr: pd.DataFrame, group_cols: list[str], max_rows: int = 12) -> str:
    if corr.empty:
        return "(no rows)"
    t = corr.dropna(subset=["rho"]).copy()
    if t.empty:
        return "(no valid correlations)"
    t["abs_rho"] = t["rho"].abs()
    t = t.sort_values(["abs_rho", "n"], ascending=[False, False]).head(max_rows)
    cols = group_cols + ["feature", "n", "rho", "p"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in t.iterrows():
        vals = []
        for c in cols:
            if c == "rho":
                vals.append(f"{r[c]:.3f}")
            elif c == "p":
                vals.append(_format_p(r[c]))
            else:
                vals.append(str(r[c]))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    loso_features: pd.DataFrame,
    loso_corr: pd.DataFrame,
    cross_features: pd.DataFrame,
    cross_corr: pd.DataFrame,
) -> None:
    loso_noea = loso_corr[loso_corr["group"].eq("NoEA_to_EA")].copy()
    cross_dea = cross_corr[cross_corr["comparison"].eq("CrossCSP_to_DatasetEA_SubjectEA")].copy()

    lines = [
        "# Transfer Benefit Predictor / Source Similarity Analysis",
        "",
        "## Inputs",
        f"- LOSO subject deltas: {len(loso_features)} subject-comparison rows after duplicate subject aggregation.",
        f"- Cross-dataset CSP-LDA deltas: {len(cross_features)} subject-direction-comparison rows.",
        "- Covariance features are computed from streamed, downsampled MOABB NPZ files.",
        "- Cross-dataset distances use only common Cho2017/Lee2019 channels.",
        "",
        "## D. Transfer Benefit Predictor",
        "",
        "Target: `delta = candidate_acc - baseline_acc`.",
        "",
        "Strongest LOSO correlations across all method/backbone/dataset comparisons:",
        "",
        _top_table(loso_corr, ["group", "baseline", "candidate", "dataset"]),
        "",
        "Strongest LOSO NoEA→EA correlations:",
        "",
        _top_table(loso_noea, ["baseline", "candidate", "dataset"]),
        "",
        "## E. Source-Subject Similarity vs Benefit",
        "",
        "Cross-dataset similarity features measure how close each test subject is to the source dataset subject pool.",
        "Smaller distance or larger `source_dataset_sim_weight` means a more similar source pool.",
        "",
        "Strongest cross-dataset correlations:",
        "",
        _top_table(cross_corr, ["comparison", "train_dataset", "test_dataset"]),
        "",
        "DatasetEA+SubjectEA focused correlations:",
        "",
        _top_table(cross_dea, ["train_dataset", "test_dataset"]),
        "",
        "## Interpretation Guide",
        "",
        "- Positive rho for `baseline_acc` means already-strong subjects gain more from the candidate method; negative rho means weak baseline subjects benefit more.",
        "- Negative rho for distance features means more source-similar subjects gain more.",
        "- Positive rho for `source_*_sim_weight` means having many close source subjects predicts larger transfer benefit.",
        "- Treat this as correlational evidence; it supports subject vulnerability and source-pool suitability analysis, not causal proof.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-root", type=Path, default=Path("moabb_streamed_npz"))
    parser.add_argument("--loso-deltas", type=Path, default=Path("analysis_outputs/loso_paired_subject_deltas.csv"))
    parser.add_argument(
        "--cross-subject-level",
        type=Path,
        default=Path("crossdata/results/aggregated/crossdataset_subject_level_all.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/transfer_benefit"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    covs = load_subject_covariances(args.npz_root)
    within = compute_within_dataset_features(covs)
    cross_sim = compute_cross_dataset_features(covs)

    within.to_csv(args.out_dir / "subject_covariance_features_loso.csv", index=False)
    cross_sim.to_csv(args.out_dir / "subject_source_similarity_crossdataset.csv", index=False)

    loso_features, loso_corr = build_loso(args.loso_deltas, within)
    loso_features.to_csv(args.out_dir / "transfer_benefit_loso_features.csv", index=False)
    loso_corr.to_csv(args.out_dir / "transfer_benefit_loso_spearman.csv", index=False)

    cross_features, cross_corr = build_cross(args.cross_subject_level, cross_sim)
    cross_features.to_csv(args.out_dir / "transfer_benefit_cross_features.csv", index=False)
    cross_corr.to_csv(args.out_dir / "transfer_benefit_cross_spearman.csv", index=False)

    write_report(
        args.out_dir / "transfer_benefit_predictor_report.md",
        loso_features,
        loso_corr,
        cross_features,
        cross_corr,
    )

    print(f"Wrote outputs to {args.out_dir}")
    print(f"LOSO feature rows: {len(loso_features)}")
    print(f"Cross feature rows: {len(cross_features)}")


if __name__ == "__main__":
    main()
