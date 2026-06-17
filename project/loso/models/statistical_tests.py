"""
Subject-level statistical tests for LOSO results.

Runs paired Wilcoxon signed-rank tests between methods at subject level.

Comparison groups:
  A. EA 효과 체인
     CSP-LDA → EA-CSP-LDA             (EA 효과, classical)
     EA-CSP-LDA → EA+CSPNet           (동일 EA, DL vs classical)   ← option 2
     CSP-LDA → EA+CSPNet              (EA+DL 총 효과)

  B. TTA 단계별 기여
     EA+CSPNet → EA+AdaBN             (AdaBN 단독)
     EA+CSPNet → EA+TENT              (TENT 단독)
     EA+CSPNet → EA+AdaBN+Con         (AdaBN+Con 총 효과)
     EA+AdaBN  → EA+AdaBN+Con         (Contrastive 추가 기여)

  C. Snapshot Ensemble
     EA+CSPNet → EA+AdaBN+Snapshot    (Snapshot 총 효과)
     EA+AdaBN+Con → EA+AdaBN+Snapshot (best model 비교)

Usage:
    python statistical_tests.py
    python statistical_tests.py --dataset cho2017
    python statistical_tests.py --group A
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# {method_name: {dataset: (csv_filename, metric_col)}}
COMBINED_FILES: dict[str, dict[str, tuple[str, str]]] = {
    "EA+CSPNet": {
        "cho2017": ("loso_results_ea_cspnet_cho_cspnet.csv",                   "acc"),
        "lee2019": ("loso_results_ea_cspnet_lee_cspnet.csv",                   "acc"),
    },
    "EA+AdaBN": {
        "cho2017": ("loso_results_ea_adabn_cspnet.csv",                        "adabn_acc"),
        "lee2019": ("loso_results_ea_adabn_cspnet.csv",                        "adabn_acc"),
    },
    "EA+TENT": {
        "cho2017": ("loso_results_ea_tent_cspnet.csv",                         "tent_acc"),
        "lee2019": ("loso_results_ea_tent_cspnet.csv",                         "tent_acc"),
    },
    "EA+AdaBN+Con": {
        "cho2017": ("loso_results_ea_adabn_contrastive_cho_cspnetcontrastive.csv", "adabn_acc"),
        "lee2019": ("loso_results_ea_adabn_contrastive_lee_cspnetcontrastive.csv", "adabn_acc"),
    },
    "EA+AdaBN+Snapshot": {
        "cho2017": ("loso_results_ea_adabn_snapshot_cspnet.csv",               "snap_adabn_acc"),
        "lee2019": ("loso_results_ea_adabn_snapshot_cspnet.csv",               "snap_adabn_acc"),
    },
    "EA+AdaBN+Snapshot(x3)": {
        "cho2017": ("loso_results_ea_adabn_snapshot_x3_cspnet_cspnet.csv",     "snap_adabn_acc"),
        "lee2019": ("loso_results_ea_adabn_snapshot_x3_cspnet_cspnet.csv",     "snap_adabn_acc"),
    },
    "EA+AdaBN+Snapshot(x4)": {
        "cho2017": ("loso_results_ea_adabn_snapshot_x4_cspnet_cspnet.csv",     "snap_adabn_acc"),
        "lee2019": ("loso_results_ea_adabn_snapshot_x4_cspnet_cspnet.csv",     "snap_adabn_acc"),
    },
    "EA+AdaBN+SubjClust": {
        "cho2017": ("loso_results_ea_adabn_subjclust_tau1_cspnet_cspnet.csv",  "adabn_acc"),
        "lee2019": ("loso_results_ea_adabn_subjclust_tau1_cspnet_cspnet.csv",  "adabn_acc"),
    },
}

# Comparison groups: (method_A, method_B, description, group_label)
COMPARISONS = [
    # ── Group A: EA 효과 체인 ─────────────────────────────────────────
    ("CSP-LDA",       "EA-CSP-LDA",        "CSP-LDA → EA-CSP-LDA (EA 효과)",     "A"),
    ("EA-CSP-LDA",    "EA+CSPNet",         "EA-CSP-LDA → EA+CSPNet (DL 효과)",   "A"),
    ("CSP-LDA",       "EA+CSPNet",         "CSP-LDA → EA+CSPNet (총 효과)",       "A"),

    # ── Group B: TTA 단계별 기여 ──────────────────────────────────────
    ("EA+CSPNet",     "EA+AdaBN",          "EA+CSPNet → EA+AdaBN",               "B"),
    ("EA+CSPNet",     "EA+TENT",           "EA+CSPNet → EA+TENT",                "B"),
    ("EA+CSPNet",     "EA+AdaBN+Con",      "EA+CSPNet → EA+AdaBN+Con",           "B"),
    ("EA+AdaBN",      "EA+AdaBN+Con",      "EA+AdaBN → EA+AdaBN+Con (Con 기여)", "B"),

    # ── Group C: Snapshot Ensemble ────────────────────────────────────
    ("EA+CSPNet",     "EA+AdaBN+Snapshot", "EA+CSPNet → EA+AdaBN+Snapshot",      "C"),
    ("EA+AdaBN+Con",  "EA+AdaBN+Snapshot", "EA+AdaBN+Con → EA+AdaBN+Snapshot",   "C"),

    # ── Group D: Snapshot cycles (x6 vs x3 vs x4) ────────────────────
    ("EA+AdaBN+Snapshot",     "EA+AdaBN+Snapshot(x3)", "x6 → x3 (fewer snapshots)", "D"),
    ("EA+AdaBN+Snapshot",     "EA+AdaBN+Snapshot(x4)", "x6 → x4 (fewer snapshots)", "D"),

    # ── Group E: Subject Clustering ───────────────────────────────────
    ("EA+AdaBN",      "EA+AdaBN+SubjClust", "EA+AdaBN → EA+AdaBN+SubjClust",     "E"),
    ("EA+AdaBN+Con",  "EA+AdaBN+SubjClust", "EA+AdaBN+Con → EA+AdaBN+SubjClust", "E"),
]

GROUP_LABELS = {
    "A": "A. EA 효과 체인 (classical → EA → DL)",
    "B": "B. TTA 단계별 기여",
    "C": "C. Snapshot Ensemble",
    "D": "D. Snapshot cycle 수 비교",
    "E": "E. Subject Clustering 가중 학습",
}


def load_subject_scores(filepath: str, metric_col: str,
                        dataset_filter: str | None = None) -> dict[int, float]:
    path = RESULTS_DIR / filepath
    if not path.exists():
        return {}
    scores: dict[int, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if dataset_filter and row.get("dataset", "") != dataset_filter:
                continue
            try:
                subj = int(row["subject"])
                val  = float(row[metric_col])
                scores[subj] = val
            except (KeyError, ValueError):
                pass
    return scores


def get_scores(method: str, dataset: str) -> dict[int, float]:
    """Return {subject: metric_value} for a given method and dataset."""
    if method == "CSP-LDA":
        return load_subject_scores(f"loso_results_csp_lda_{dataset}.csv",
                                   "acc", dataset_filter=dataset)
    if method == "EA-CSP-LDA":
        return load_subject_scores(f"loso_results_ea_csp_lda_{dataset}.csv",
                                   "acc", dataset_filter=dataset)
    if method in COMBINED_FILES:
        fname, col = COMBINED_FILES[method][dataset]
        return load_subject_scores(fname, col, dataset_filter=dataset)
    return {}


def paired_wilcoxon(a_scores: dict, b_scores: dict):
    """Paired Wilcoxon on common subjects. Returns (statistic, p, n, mean_diff)."""
    common = sorted(set(a_scores) & set(b_scores))
    if len(common) < 10:
        return None, None, len(common), None
    a = np.array([a_scores[s] for s in common])
    b = np.array([b_scores[s] for s in common])
    stat, p = wilcoxon(a, b, alternative="two-sided")
    return stat, p, len(common), float((b - a).mean())


def effect_size_r(stat: float, n: int) -> float:
    """r = Z / sqrt(N) approximation from Wilcoxon T statistic."""
    z = abs(stat - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return float(z / np.sqrt(n))


def stars(p) -> str:
    if p is None:  return "n/a"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"


def run(datasets: list[str], filter_group: str | None = None):
    for dataset in datasets:
        print(f"\n{'='*72}")
        print(f"  {dataset.upper()}")
        print(f"{'='*72}")

        current_group = None
        for method_a, method_b, desc, grp in COMPARISONS:
            if filter_group and grp != filter_group:
                continue

            if grp != current_group:
                current_group = grp
                print(f"\n  {GROUP_LABELS[grp]}")
                print(f"  {'─'*68}")
                print(f"  {'비교':<40} {'n':>3} {'Δacc':>8}  {'p':>8}  {'sig':>3}  {'r':>5}")
                print(f"  {'─'*68}")

            a = get_scores(method_a, dataset)
            b = get_scores(method_b, dataset)

            if not a:
                print(f"  {desc:<40} — {method_a} 없음")
                continue
            if not b:
                print(f"  {desc:<40} — {method_b} 없음")
                continue

            stat, p, n, mean_diff = paired_wilcoxon(a, b)
            if p is None:
                print(f"  {desc:<40} n={n:>2}  (피험자 부족)")
                continue

            r = effect_size_r(stat, n)
            diff_str = f"{mean_diff*100:+.2f}%p"
            p_str = f"p={p:.4f}" if p >= 0.0001 else "p<0.0001"
            print(f"  {desc:<40} {n:>3}  {diff_str:>8}  {p_str:>8}  {stars(p):>3}  {r:.3f}")

        # 각 방법 mean±std
        print(f"\n  {'방법':<30} {'n':>3}  {'Acc':>12}  {'κ 참고'}")
        print(f"  {'─'*60}")
        methods_order = ["CSP-LDA", "EA-CSP-LDA", "EA+CSPNet",
                         "EA+AdaBN", "EA+TENT", "EA+AdaBN+Con", "EA+AdaBN+Snapshot"]
        for method in methods_order:
            sc = get_scores(method, dataset)
            if sc:
                vals = list(sc.values())
                print(f"  {method:<30} {len(vals):>3}  "
                      f"{np.mean(vals)*100:.2f}% ±{np.std(vals)*100:.2f}")


def load_cross_scores(filepath: str, acc_col: str = "acc") -> dict[int, float]:
    path = RESULTS_DIR / filepath
    if not path.exists():
        return {}
    scores: dict[int, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                scores[int(row["subject"])] = float(row[acc_col])
            except (KeyError, ValueError):
                pass
    return scores


def run_cross_dataset_sessionea():
    """Wilcoxon test: DatasetEA+SubjectEA+CSP-LDA vs SessionEA+CSP-LDA (cross-dataset)."""
    directions = [
        ("cho2017_to_lee2019", "Cho→Lee (n=54)"),
        ("lee2019_to_cho2017", "Lee→Cho (n=52)"),
    ]
    # Use the "metrics_final" baseline (20260608), which matches the published summary (68.96/65.10%)
    baseline_tmpl = "loso_results_20260608_metrics_final_base_cross_{dir}_csp_lda.csv"
    session_tmpl  = "loso_results_20260608_metrics_final_session_ea_cross_{dir}_csp_lda.csv"

    print(f"\n{'='*72}")
    print("  CROSS-DATASET: DatasetEA+SubjectEA+CSP-LDA  vs  SessionEA+CSP-LDA")
    print(f"{'='*72}")
    print(f"  {'방향':<22} {'n':>3}  {'Δacc':>8}  {'p':>8}  {'sig':>3}  {'r':>5}  {'baseline':>10}  {'sessionea':>10}")
    print(f"  {'─'*80}")

    for tag, label in directions:
        base_scores    = load_cross_scores(baseline_tmpl.format(dir=tag))
        session_scores = load_cross_scores(session_tmpl.format(dir=tag))

        if not base_scores:
            print(f"  {label:<22} — baseline 없음")
            continue
        if not session_scores:
            print(f"  {label:<22} — sessionea 없음")
            continue

        common = sorted(set(base_scores) & set(session_scores))
        n = len(common)
        if n < 10:
            print(f"  {label:<22} n={n:>2}  (피험자 부족)")
            continue

        a = np.array([base_scores[s]    for s in common])
        b = np.array([session_scores[s] for s in common])
        stat, p = wilcoxon(a, b, alternative="two-sided")
        r = effect_size_r(stat, n)
        mean_diff = float((b - a).mean())
        diff_str = f"{mean_diff:+.2f}%p"
        p_str = f"p={p:.4f}" if p >= 0.0001 else "p<0.0001"

        base_mean    = float(a.mean())
        session_mean = float(b.mean())
        print(f"  {label:<22} {n:>3}  {diff_str:>8}  {p_str:>8}  {stars(p):>3}  {r:.3f}  "
              f"{base_mean:>9.2f}%  {session_mean:>9.2f}%")

    print()
    print("  Note: Cliff's delta not computed here (use r as effect proxy).")
    print("  r interpretation: <0.1=negligible, <0.3=small, <0.5=medium, ≥0.5=large")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cho2017", "lee2019", "both"],
                        default="both")
    parser.add_argument("--group",   choices=["A", "B", "C"],
                        help="특정 그룹만 출력")
    parser.add_argument("--cross", action="store_true",
                        help="Cross-dataset SessionEA vs DatasetEA+SubjectEA 검정")
    args = parser.parse_args()
    if args.cross:
        run_cross_dataset_sessionea()
    else:
        datasets = ["cho2017", "lee2019"] if args.dataset == "both" else [args.dataset]
        run(datasets, filter_group=args.group)


if __name__ == "__main__":
    main()
