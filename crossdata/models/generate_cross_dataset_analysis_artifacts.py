"""Generate cross-dataset statistics, subject figures, EA ablation tables, and t-SNE plots."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS = PROJECT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
PREP_RAW_UNIFIED = PROJECT / "preprocessed_raw_unified"
PROGRESS = PROJECT / "progress.md"

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def _scale_acc(x: float) -> float:
    return x * 100.0 if x <= 1.5 else x


def load_scores(path: Path, metric="acc", dataset=None):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if dataset is not None and row.get("dataset") not in (None, "", dataset):
                continue
            if metric not in row or row[metric] in (None, ""):
                continue
            subject = int(float(row.get("subject", len(rows) + 1)))
            acc = _scale_acc(float(row[metric]))
            kappa = float(row.get(metric.replace("acc", "kappa"), row.get("kappa", "nan")) or "nan")
            rows.append({"subject": subject, "acc": acc, "kappa": kappa})
    return rows


def by_subject(rows):
    return {r["subject"]: r for r in rows}


def paired_arrays(a_rows, b_rows):
    a, b = by_subject(a_rows), by_subject(b_rows)
    subjects = sorted(set(a) & set(b))
    return subjects, np.array([a[s]["acc"] for s in subjects]), np.array([b[s]["acc"] for s in subjects])


def test_pair(name_a, rows_a, name_b, rows_b, context):
    subjects, a, b = paired_arrays(rows_a, rows_b)
    diff = b - a
    if len(subjects) < 3:
        return None
    try:
        wil = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        wil_p = float(wil.pvalue)
        wil_stat = float(wil.statistic)
    except ValueError:
        wil_p = np.nan
        wil_stat = np.nan
    t = stats.ttest_rel(b, a)
    dz = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    return {
        "context": context,
        "baseline": name_a,
        "method": name_b,
        "n": len(subjects),
        "baseline_mean": a.mean(),
        "method_mean": b.mean(),
        "delta_mean": diff.mean(),
        "delta_median": np.median(diff),
        "wilcoxon_stat": wil_stat,
        "wilcoxon_p": wil_p,
        "paired_t": float(t.statistic),
        "paired_t_p": float(t.pvalue),
        "cohens_dz": dz,
        "n_improved": int((diff > 0).sum()),
    }


CROSS = {
    "cho2017->lee2019": {
        "CSP-LDA": RESULTS / "loso_results_20260604_cross_stdmi_csp_lda_cross_cho2017_to_lee2019_csp_lda.csv",
        "EA-CSP-LDA": RESULTS / "loso_results_20260604_cross_stdmi_ea_csp_lda_cross_cho2017_to_lee2019_csp_lda.csv",
        "DatasetEA+SubjectEA+CSP-LDA": RESULTS / "loso_results_20260604_cross_stdmi_datasetea_ea_csp_lda_cross_cho2017_to_lee2019_csp_lda.csv",
        "RawUnified+DatasetEA+SubjectEA+CSP-LDA": RESULTS / "loso_results_20260605_raw_unified_cross_cho2017_to_lee2019_csp_lda.csv",
        "DatasetEA+SubjectEA+CSPNet": RESULTS / "loso_results_20260604_cross_stdmi_datasetea_ea_cspnet_cross_cho2017_to_lee2019_cspnet.csv",
        "SubjectEA+DatasetEA+CSPNet": RESULTS / "loso_results_20260604_cross_stdmi_subject_dataset_ea_cspnet_cross_cho2017_to_lee2019_cspnet.csv",
    },
    "lee2019->cho2017": {
        "CSP-LDA": RESULTS / "loso_results_20260604_cross_stdmi_csp_lda_cross_lee2019_to_cho2017_csp_lda.csv",
        "EA-CSP-LDA": RESULTS / "loso_results_20260604_cross_stdmi_ea_csp_lda_cross_lee2019_to_cho2017_csp_lda.csv",
        "DatasetEA+SubjectEA+CSP-LDA": RESULTS / "loso_results_20260604_cross_stdmi_datasetea_ea_csp_lda_cross_lee2019_to_cho2017_csp_lda.csv",
        "RawUnified+DatasetEA+SubjectEA+CSP-LDA": RESULTS / "loso_results_20260605_raw_unified_cross_lee2019_to_cho2017_csp_lda.csv",
        "DatasetEA+SubjectEA+CSPNet": RESULTS / "loso_results_20260604_cross_stdmi_datasetea_ea_cspnet_cross_lee2019_to_cho2017_cspnet.csv",
        "SubjectEA+DatasetEA+CSPNet": RESULTS / "loso_results_20260604_cross_stdmi_subject_dataset_ea_cspnet_cross_lee2019_to_cho2017_cspnet.csv",
    },
}

LOSO = {
    "cho2017": {
        "CSP-LDA": (RESULTS / "loso_results_csp_lda_cho2017.csv", "acc"),
        "EA-CSP-LDA": (RESULTS / "loso_results_ea_csp_lda_cho2017.csv", "acc"),
        "EA+CSPNet": (RESULTS / "loso_results_ea_cspnet_cho_cspnet.csv", "acc"),
        "EA+AdaBN+Con": (RESULTS / "loso_results_ea_adabn_contrastive_cho_cspnetcontrastive.csv", "adabn_acc"),
    },
    "lee2019": {
        "CSP-LDA": (RESULTS / "loso_results_csp_lda_lee2019.csv", "acc"),
        "EA-CSP-LDA": (RESULTS / "loso_results_ea_csp_lda_lee2019.csv", "acc"),
        "EA+CSPNet": (RESULTS / "loso_results_ea_cspnet_lee_cspnet.csv", "acc"),
        "EA+AdaBN+Con": (RESULTS / "loso_results_ea_adabn_contrastive_lee_cspnetcontrastive.csv", "adabn_acc"),
    },
}


def write_rows(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def markdown_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        vals = []
        for h in headers:
            v = r[h]
            if isinstance(v, float):
                if "p" in h:
                    vals.append(f"{v:.3g}")
                else:
                    vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_stats():
    tests = []
    loaded_cross = {}
    for context, mapping in CROSS.items():
        loaded_cross[context] = {name: load_scores(path) for name, path in mapping.items() if path.exists()}
        pairs = [
            ("CSP-LDA", "DatasetEA+SubjectEA+CSP-LDA"),
            ("EA-CSP-LDA", "DatasetEA+SubjectEA+CSP-LDA"),
            ("DatasetEA+SubjectEA+CSPNet", "DatasetEA+SubjectEA+CSP-LDA"),
            ("DatasetEA+SubjectEA+CSP-LDA", "RawUnified+DatasetEA+SubjectEA+CSP-LDA"),
            ("SubjectEA+DatasetEA+CSPNet", "DatasetEA+SubjectEA+CSPNet"),
        ]
        for a, b in pairs:
            if a in loaded_cross[context] and b in loaded_cross[context]:
                out = test_pair(a, loaded_cross[context][a], b, loaded_cross[context][b], context)
                if out:
                    tests.append(out)

    loaded_loso = {}
    for context, mapping in LOSO.items():
        loaded_loso[context] = {
            name: load_scores(path, metric=metric, dataset=context)
            for name, (path, metric) in mapping.items() if path.exists()
        }
        for a, b in [("CSP-LDA", "EA-CSP-LDA"), ("EA+CSPNet", "EA+AdaBN+Con")]:
            if a in loaded_loso[context] and b in loaded_loso[context]:
                out = test_pair(a, loaded_loso[context][a], b, loaded_loso[context][b], f"LOSO {context}")
                if out:
                    tests.append(out)

    fieldnames = ["context", "baseline", "method", "n", "baseline_mean", "method_mean", "delta_mean", "delta_median", "wilcoxon_stat", "wilcoxon_p", "paired_t", "paired_t_p", "cohens_dz", "n_improved"]
    write_rows(TABLES / "cross_dataset_and_loso_stat_tests.csv", tests, fieldnames)

    summary_rows = []
    for r in tests:
        summary_rows.append({
            "Context": r["context"],
            "Comparison": f"{r['method']} vs {r['baseline']}",
            "N": r["n"],
            "Baseline": r["baseline_mean"],
            "Method": r["method_mean"],
            "Delta": r["delta_mean"],
            "Wilcoxon p": r["wilcoxon_p"],
            "d_z": r["cohens_dz"],
            "Improved": f"{r['n_improved']}/{r['n']}",
        })
    md = "# Statistical Tests\n\n" + markdown_table(summary_rows, ["Context", "Comparison", "N", "Baseline", "Method", "Delta", "Wilcoxon p", "d_z", "Improved"])
    (TABLES / "statistical_tests.md").write_text(md)
    return loaded_cross, loaded_loso, tests


def plot_distribution(name, data, out_prefix, title):
    labels = list(data.keys())
    vals = [[r["acc"] for r in data[k]] for k in labels]
    fig, ax = plt.subplots(figsize=(max(9, 1.35 * len(labels)), 5.2))
    bp = ax.boxplot(vals, patch_artist=True, showfliers=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    rng = np.random.default_rng(2026)
    for i, v in enumerate(vals, 1):
        x = i + rng.normal(0, 0.035, size=len(v))
        ax.scatter(x, v, s=16, alpha=0.55, color="black", linewidths=0)
        ax.text(i, np.mean(v) + 1.2, f"{np.mean(v):.1f}", ha="center", fontsize=8)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = FIGURES / f"{out_prefix}_{name}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def generate_figures(loaded_cross, loaded_loso):
    outputs = []
    for context, data in loaded_cross.items():
        keep = {k: v for k, v in data.items() if k in ["CSP-LDA", "EA-CSP-LDA", "DatasetEA+SubjectEA+CSP-LDA", "RawUnified+DatasetEA+SubjectEA+CSP-LDA", "DatasetEA+SubjectEA+CSPNet"]}
        outputs.append(plot_distribution(context.replace("->", "_to_"), keep, "cross_subject_distribution", f"Cross-dataset subject accuracy: {context}"))
    for context, data in loaded_loso.items():
        outputs.append(plot_distribution(context, data, "loso_subject_distribution", f"LOSO subject accuracy: {context}"))
    return outputs


def generate_ea_ablation(loaded_cross):
    rows = []
    for context, data in loaded_cross.items():
        for method in ["CSP-LDA", "EA-CSP-LDA", "DatasetEA+SubjectEA+CSP-LDA", "SubjectEA+DatasetEA+CSPNet", "DatasetEA+SubjectEA+CSPNet"]:
            if method not in data:
                continue
            vals = np.array([r["acc"] for r in data[method]])
            rows.append({
                "Direction": context,
                "Method": method,
                "N": len(vals),
                "Mean Acc": vals.mean(),
                "Std Acc": vals.std(ddof=0),
                ">=60%": int((vals >= 60).sum()),
                ">=70%": int((vals >= 70).sum()),
            })
    write_rows(TABLES / "ea_order_ablation_summary.csv", rows, ["Direction", "Method", "N", "Mean Acc", "Std Acc", ">=60%", ">=70%"])
    (TABLES / "ea_order_ablation_summary.md").write_text("# EA Order Ablation\n\n" + markdown_table(rows, ["Direction", "Method", "N", "Mean Acc", "Std Acc", ">=60%", ">=70%"] ))
    return rows


def _mean_cov_features(X):
    feats = []
    iu = np.triu_indices(X.shape[1])
    for trial in X:
        z = trial - trial.mean(axis=1, keepdims=True)
        c = z @ z.T
        tr = np.trace(c)
        if tr > 1e-12:
            c = c / tr
        feats.append(c[iu])
    return np.asarray(feats, dtype=np.float32)


def _ea_dataset(X):
    from cross_dataset import apply_dataset_ea
    return apply_dataset_ea(X)


def _ea_subject(X, subjects):
    from eeg_ea import apply_ea_loso
    return apply_ea_loso(X, subjects)


def generate_tsne():
    cho_path = PREP_RAW_UNIFIED / "cho2017.npz"
    lee_path = PREP_RAW_UNIFIED / "lee2019.npz"
    if not cho_path.exists() or not lee_path.exists():
        return None
    cho = np.load(cho_path, allow_pickle=True)
    lee = np.load(lee_path, allow_pickle=True)
    Xc, sc = cho["X"].astype(np.float32), cho["subjects"].astype(int)
    Xl, sl = lee["X"].astype(np.float32), lee["subjects"].astype(int)
    rng = np.random.default_rng(2026)
    n_each = 700
    ic = rng.choice(len(Xc), size=min(n_each, len(Xc)), replace=False)
    il = rng.choice(len(Xl), size=min(n_each, len(Xl)), replace=False)
    X_before = np.concatenate([Xc[ic], Xl[il]], axis=0)
    y_domain = np.array([0] * len(ic) + [1] * len(il))

    Xc_ea = _ea_subject(_ea_dataset(Xc), sc)[ic]
    Xl_ea = _ea_subject(_ea_dataset(Xl), sl)[il]
    X_after = np.concatenate([Xc_ea, Xl_ea], axis=0)

    F_before = _mean_cov_features(X_before)
    F_after = _mean_cov_features(X_after)
    F = np.vstack([F_before, F_after])
    F = PCA(n_components=min(30, F.shape[1]), random_state=2026).fit_transform(F)
    emb = TSNE(n_components=2, perplexity=35, init="pca", learning_rate="auto", random_state=2026).fit_transform(F)
    before = emb[:len(F_before)]
    after = emb[len(F_before):]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, points, title in [(axes[0], before, "Before EA"), (axes[1], after, "After DatasetEA -> SubjectEA")]:
        for dom, label, color in [(0, "Cho2017", "#1f77b4"), (1, "Lee2019", "#d62728")]:
            mask = y_domain == dom
            ax.scatter(points[mask, 0], points[mask, 1], s=9, alpha=0.55, label=label, c=color)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2, fontsize=8)
    fig.suptitle("t-SNE of covariance features before/after EA")
    fig.tight_layout()
    out = FIGURES / "tsne_raw_unified_ea_before_after.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    loaded_cross, loaded_loso, tests = generate_stats()
    figs = generate_figures(loaded_cross, loaded_loso)
    ea_rows = generate_ea_ablation(loaded_cross)
    tsne_out = generate_tsne()

    block = [
        "\n\n## Cross-Dataset Statistical/Figure Artifacts",
        "",
        "- Status: `completed`",
        f"- Statistical tests: `{TABLES / 'cross_dataset_and_loso_stat_tests.csv'}`",
        f"- Statistical test table: `{TABLES / 'statistical_tests.md'}`",
        f"- EA order ablation table: `{TABLES / 'ea_order_ablation_summary.md'}`",
    ]
    for fig in figs:
        block.append(f"- Subject-level figure: `{fig}`")
    if tsne_out:
        block.append(f"- t-SNE figure: `{tsne_out}`")
    block.append("")
    PROGRESS.write_text(PROGRESS.read_text() + "\n".join(block))

    print("Generated artifacts:")
    print(TABLES / "cross_dataset_and_loso_stat_tests.csv")
    print(TABLES / "statistical_tests.md")
    print(TABLES / "ea_order_ablation_summary.md")
    for fig in figs:
        print(fig)
    if tsne_out:
        print(tsne_out)


if __name__ == "__main__":
    main()
