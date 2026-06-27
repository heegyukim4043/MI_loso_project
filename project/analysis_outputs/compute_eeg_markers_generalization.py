from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr


NPZ_ROOT = Path("moabb_streamed_npz")
ACC = Path("analysis_outputs/class_feature_similarity/class_feature_similarity_with_accuracy.csv")
OUT = Path("analysis_outputs/eeg_markers_generalization")

DATASETS = {"cho2017": "cho", "lee2019": "lee"}
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0), "mu_beta": (8.0, 30.0)}
LEFT_CH = ["FC3", "C3", "CP3"]
RIGHT_CH = ["FC4", "C4", "CP4"]
MID_CH = ["Cz"]


def subject_from_path(path: Path) -> int:
    for part in path.parts[::-1]:
        if part.startswith("sub-"):
            return int(part.split("-", 1)[1])
    raise ValueError(path)


def bandpower(x: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    freqs, psd = welch(x, fs=sfreq, nperseg=min(x.shape[-1], 200), axis=-1)
    mask = (freqs >= band[0]) & (freqs < band[1])
    if not np.any(mask):
        raise ValueError(f"No frequencies in band {band}")
    return np.trapz(psd[..., mask], freqs[mask], axis=-1)


def safe_idx(ch_names: list[str], wanted: list[str]) -> list[int]:
    return [ch_names.index(ch) for ch in wanted if ch in ch_names]


def load_subject(dataset: str, subdir: str, subject: int, paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float]:
    xs, ys, labels = [], [], []
    ch_names = None
    sfreq = None
    for path in paths:
        with np.load(path, allow_pickle=True) as npz:
            x = np.asarray(npz["X"])
            y = np.asarray(npz["y"])
            lab = np.asarray(npz["labels"]).astype(str)
            names = [str(c) for c in npz["ch_names"].tolist()]
            fs = float(np.asarray(npz["sfreq"]).ravel()[0])
        if ch_names is None:
            ch_names = names
            sfreq = fs
        elif ch_names != names:
            raise ValueError(f"Channel mismatch: {dataset} subject {subject}")
        xs.append(x)
        ys.append(y)
        labels.append(lab)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(labels), ch_names or [], float(sfreq)


def compute_subject_metrics(dataset: str, subject: int, x: np.ndarray, labels: np.ndarray, ch_names: list[str], sfreq: float) -> dict[str, float]:
    left_idx = safe_idx(ch_names, LEFT_CH)
    right_idx = safe_idx(ch_names, RIGHT_CH)
    mid_idx = safe_idx(ch_names, MID_CH)
    sm_idx = left_idx + right_idx + mid_idx
    if len(left_idx) == 0 or len(right_idx) == 0:
        raise ValueError(f"Missing C3/C4 groups for {dataset} subject {subject}")

    is_left = labels == "left_hand"
    is_right = labels == "right_hand"
    if not is_left.any() or not is_right.any():
        raise ValueError(f"Expected left_hand/right_hand labels for {dataset} subject {subject}")

    row: dict[str, float] = {
        "dataset": dataset,
        "subject": subject,
        "n_trials": len(labels),
        "n_channels": len(ch_names),
    }

    for band_name, band in BANDS.items():
        bp = bandpower(x, sfreq, band)
        bp = np.maximum(bp, 1e-20)
        log_bp = np.log(bp)

        left_power = log_bp[:, left_idx].mean(axis=1)
        right_power = log_bp[:, right_idx].mean(axis=1)
        mid_power = log_bp[:, mid_idx].mean(axis=1) if mid_idx else np.full(len(labels), np.nan)
        sm_power = log_bp[:, sm_idx].mean(axis=1)
        all_power = log_bp.mean(axis=1)

        asym = right_power - left_power
        asym_left = asym[is_left].mean()
        asym_right = asym[is_right].mean()
        asym_diff = asym_left - asym_right

        # Positive values indicate lower contralateral than ipsilateral band power in the task window.
        contra_left_mi = right_power[is_left]
        ipsi_left_mi = left_power[is_left]
        contra_right_mi = left_power[is_right]
        ipsi_right_mi = right_power[is_right]
        contra_desync = np.r_[ipsi_left_mi - contra_left_mi, ipsi_right_mi - contra_right_mi].mean()

        class_sm_diff = sm_power[is_left].mean() - sm_power[is_right].mean()
        row.update(
            {
                f"{band_name}_hemi_asym_left": float(asym_left),
                f"{band_name}_hemi_asym_right": float(asym_right),
                f"{band_name}_hemi_asym_diff": float(asym_diff),
                f"{band_name}_hemi_asym_absdiff": float(abs(asym_diff)),
                f"{band_name}_contra_desync_index": float(contra_desync),
                f"{band_name}_sensorimotor_logpower": float(sm_power.mean()),
                f"{band_name}_sensorimotor_vs_all": float((sm_power - all_power).mean()),
                f"{band_name}_left_right_balance_abs": float(abs((right_power - left_power).mean())),
                f"{band_name}_class_sensorimotor_absdiff": float(abs(class_sm_diff)),
            }
        )
    return row


def compute_all() -> pd.DataFrame:
    rows = []
    for dataset, subdir in DATASETS.items():
        grouped: dict[int, list[Path]] = {}
        for path in sorted((NPZ_ROOT / subdir).glob("sub-*/*.npz")):
            grouped.setdefault(subject_from_path(path), []).append(path)
        for subject, paths in sorted(grouped.items()):
            x, _, labels, ch_names, sfreq = load_subject(dataset, subdir, subject, paths)
            rows.append(compute_subject_metrics(dataset, subject, x, labels, ch_names, sfreq))
    return pd.DataFrame(rows)


def correlate(markers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ACC.exists():
        return pd.DataFrame(), pd.DataFrame()
    acc = pd.read_csv(ACC)
    merged = markers.merge(
        acc[["dataset", "subject", "original_acc", "best_generalization_acc", "best_delta", "original_illiterate", "persistent_fail"]],
        on=["dataset", "subject"],
        how="left",
    )
    marker_cols = [c for c in markers.columns if c not in {"dataset", "subject", "n_trials", "n_channels"}]
    targets = ["original_acc", "best_generalization_acc", "best_delta"]
    rows = []
    for dataset, g in merged.groupby("dataset"):
        for feat in marker_cols:
            for target in targets:
                sub = g[[feat, target]].replace([np.inf, -np.inf], np.nan).dropna()
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
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(markers: pd.DataFrame, merged: pd.DataFrame, corr: pd.DataFrame) -> None:
    key_cols = [
        "mu_hemi_asym_absdiff",
        "beta_hemi_asym_absdiff",
        "mu_beta_hemi_asym_absdiff",
        "mu_contra_desync_index",
        "beta_contra_desync_index",
        "mu_beta_class_sensorimotor_absdiff",
    ]
    summary = markers.groupby("dataset")[key_cols].mean().reset_index()
    for c in key_cols:
        summary[c] = summary[c].round(4)

    group_summary = pd.DataFrame()
    if not merged.empty:
        group_summary = merged.groupby(["dataset", "original_illiterate", "persistent_fail"]).agg(
            n=("subject", "count"),
            original_acc=("original_acc", "mean"),
            best_generalization_acc=("best_generalization_acc", "mean"),
            mu_hemi_asym_absdiff=("mu_hemi_asym_absdiff", "mean"),
            beta_hemi_asym_absdiff=("beta_hemi_asym_absdiff", "mean"),
            mu_contra_desync_index=("mu_contra_desync_index", "mean"),
            beta_contra_desync_index=("beta_contra_desync_index", "mean"),
            mu_beta_class_sensorimotor_absdiff=("mu_beta_class_sensorimotor_absdiff", "mean"),
        ).reset_index()
        for c in group_summary.columns:
            if c not in {"dataset", "original_illiterate", "persistent_fail", "n"}:
                group_summary[c] = group_summary[c].round(4)

    top = corr.dropna(subset=["rho"]).copy()
    if not top.empty:
        top["abs_rho"] = top["rho"].abs()
        top = top.sort_values("abs_rho", ascending=False).head(20).drop(columns=["abs_rho"])
        top["rho"] = top["rho"].round(3)
        top["p"] = top["p"].round(4)

    lines = [
        "# EEG Marker-Based Generalization Analysis",
        "",
        "These are task-window EEG markers computed from the streamed NPZ trials. Because no pre-cue baseline is stored, `contra_desync_index` is a desynchronization-like lateralization index, not strict baseline-corrected ERD.",
        "",
        "## Marker Definitions",
        "",
        "- `*_hemi_asym_absdiff`: absolute left/right class difference in hemispheric asymmetry `log(right sensorimotor power) - log(left sensorimotor power)`.",
        "- `*_contra_desync_index`: task-window ipsilateral minus contralateral log bandpower. Positive values are consistent with lower contralateral power.",
        "- `*_sensorimotor_vs_all`: sensorimotor channel logpower relative to all-channel logpower.",
        "- `*_class_sensorimotor_absdiff`: class difference in sensorimotor log bandpower.",
        "",
        "## Dataset Means",
        "",
        md_table(summary),
        "",
        "## Illiteracy / Persistent Failure Groups",
        "",
        md_table(group_summary),
        "",
        "## Strongest EEG Marker Correlations",
        "",
        md_table(top),
        "",
        "## Interpretation",
        "",
        "- EEG markers are useful if they separate recovered vs persistent low-performing subjects beyond accuracy alone.",
        "- Lateralized mu/beta asymmetry is the most physiology-aligned marker for left/right MI.",
        "- Use these as construct-validity features alongside CSP separability and covariance geometry.",
        "",
    ]
    (OUT / "eeg_marker_generalization_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    markers = compute_all()
    merged, corr = correlate(markers)
    markers.to_csv(OUT / "eeg_markers_by_subject.csv", index=False)
    if not merged.empty:
        merged.to_csv(OUT / "eeg_markers_with_accuracy.csv", index=False)
    if not corr.empty:
        corr.to_csv(OUT / "eeg_marker_spearman.csv", index=False)
    write_report(markers, merged, corr)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
