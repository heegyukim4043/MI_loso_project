#!/usr/bin/env python3
"""
update_summary.py
  실험 결과 CSV 파싱 → subject별 성능 테이블 업데이트 + PROJECT_SUMMARY.md 반영

Usage:
    python update_summary.py           # 모든 실험 처리
    python update_summary.py --dry-run # 출력만, 파일 수정 없음
    python update_summary.py --exp EA+AdaBN  # 특정 실험만
"""

import os, csv, re, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("/home/hkim/MI_test/results")
PROJECT_DIR = Path("/home/hkim/MI_test/MI_loso_project")
SUMMARY_MD  = PROJECT_DIR / "PROJECT_SUMMARY.md"

SUBJ_FILES = {
    "cho2017": {
        "acc":   PROJECT_DIR / "subject_results_cho2017_acc.csv",
        "kappa": PROJECT_DIR / "subject_results_cho2017_kappa.csv",
    },
    "lee2019": {
        "acc":   PROJECT_DIR / "subject_results_lee2019_acc.csv",
        "kappa": PROJECT_DIR / "subject_results_lee2019_kappa.csv",
    },
    "physionet": {
        "acc":   PROJECT_DIR / "subject_results_physionet_acc.csv",
        "kappa": PROJECT_DIR / "subject_results_physionet_kappa.csv",
    },
}
N_SUBJECTS = {"cho2017": 52, "lee2019": 54, "physionet": 106}

# ─────────────────────────────────────────────────────────────────
# 실험 정의
# cho/lee: 결과 파일명(RESULTS_DIR 기준). 같은 파일이면 dataset 컬럼으로 분리.
# metric: "base"=acc/kappa, "adabn"=adabn_acc/adabn_kappa, "tent"=tent_acc/tent_kappa
# ─────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "name":   "EA+CSPNet",
        "cho":    "loso_results_ea_cspnet_cho_cspnet.csv",
        "lee":    "loso_results_ea_cspnet_lee_cspnet.csv",
        "metric": "base",
    },
    {
        "name":   "EA+Contrastive",
        "cho":    "loso_results_ea_contrastive_cho_cspnetcontrastive.csv",
        "lee":    "loso_results_ea_contrastive_lee_cspnetcontrastive.csv",
        "metric": "base",
    },
    {
        "name":   "AdaBN",
        "cho":    "loso_results_adabn_cspnet_cho_cspnet.csv",
        "lee":    "loso_results_adabn_cspnet_lee_cspnet.csv",
        "metric": "adabn",
    },
    {
        "name":   "AdaBN+Con",
        "cho":    "loso_results_adabn_contrastive_cho_cspnetcontrastive.csv",
        "lee":    "loso_results_adabn_contrastive_lee_cspnetcontrastive.csv",
        "metric": "adabn",
    },
    {
        "name":   "EA+AdaBN",
        "cho":    "loso_results_ea_adabn_cspnet.csv",
        "lee":    "loso_results_ea_adabn_cspnet.csv",
        "metric": "adabn",
    },
    {
        "name":   "EA+AdaBN+Con",
        "cho":    "loso_results_ea_adabn_contrastive_cho_cspnetcontrastive.csv",
        "lee":    "loso_results_ea_adabn_contrastive_lee_cspnetcontrastive.csv",
        "metric": "adabn",
    },
    {
        "name":   "EA+TENT",
        "cho":    "loso_results_ea_tent_cspnet.csv",
        "lee":    "loso_results_ea_tent_cspnet.csv",
        "metric": "tent",
    },
    {
        "name":   "EA+AdaBN+Con+TENT",
        "cho":    "loso_results_ea_adabn_contrastive_tent_cspnetcontrastive.csv",
        "lee":    "loso_results_ea_adabn_contrastive_tent_cspnetcontrastive.csv",
        "metric": "tent",
    },
    {
        "name":   "EA+Con+CORAL(0.1)",
        "cho":    "loso_results_ea_coral_l01_cspnetcontrastive.csv",
        "lee":    "loso_results_ea_coral_l01_cspnetcontrastive.csv",
        "metric": "base",
    },
    # ── Backbone 실험 ─────────────────────────────────────────────────────
    {
        "name":   "EEGNet+EA+TENT",
        "cho":    "loso_results_eegnet_ea_tent_eegnet.csv",
        "lee":    "loso_results_eegnet_ea_tent_eegnet.csv",
        "metric": "tent",
    },
    {
        "name":   "Conformer+EA+TENT",
        "cho":    "loso_results_conformer_ea_tent_conformer.csv",
        "lee":    "loso_results_conformer_ea_tent_conformer.csv",
        "metric": "tent",
    },
    {
        "name":   "Conformer+EA+AdaBN+Con+TENT",
        "cho":    "loso_results_conformer_ea_adabn_tent_conformer.csv",
        "lee":    "loso_results_conformer_ea_adabn_tent_conformer.csv",
        "metric": "adabn",
    },
    # ── Ensemble 실험 ─────────────────────────────────────────────────────
    {
        "name":   "EA+Snapshot(x6)",
        "cho":    "loso_results_ea_snapshot_cspnet.csv",
        "lee":    "loso_results_ea_snapshot_cspnet.csv",
        "metric": "snap",
    },
    {
        "name":   "EA+AdaBN+Snapshot(x6)",
        "cho":    "loso_results_ea_adabn_snapshot_cspnet.csv",
        "lee":    "loso_results_ea_adabn_snapshot_cspnet.csv",
        "metric": "snap_adabn",
    },
    {
        "name":   "EA+AdaBN+Con+Snapshot(x6)",
        "cho":    "loso_results_ea_adabn_con_snapshot_cspnetcontrastive.csv",
        "lee":    "loso_results_ea_adabn_con_snapshot_cspnetcontrastive.csv",
        "metric": "snap_adabn",
    },
    # ── FBCSP 실험 ────────────────────────────────────────────────────────
    {
        "name":   "EA+FBCSP+TENT",
        "cho":    "loso_results_ea_fbcsp_tent_cspnet.csv",
        "lee":    "loso_results_ea_fbcsp_tent_cspnet.csv",
        "metric": "tent",
    },
    {
        "name":   "EA+LabelSmooth+AdaBN",
        "cho":    "loso_results_ea_ls_adabn_cspnet_ls0.1.csv",
        "lee":    "loso_results_ea_ls_adabn_cspnet_ls0.1.csv",
        "metric": "adabn",
    },
    # ── PhysioNet 실험 ────────────────────────────────────────────────────
    {
        "name":   "[PHY] EA+CSPNet",
        "cho":    "loso_results_physionet_ea_cspnet.csv",
        "lee":    "loso_results_physionet_ea_cspnet.csv",
        "metric": "base",
        "dataset": "physionet",
    },
    {
        "name":   "[PHY] EA+AdaBN+Con",
        "cho":    "loso_results_physionet_ea_adabn_con_cspnetcontrastive.csv",
        "lee":    "loso_results_physionet_ea_adabn_con_cspnetcontrastive.csv",
        "metric": "adabn",
        "dataset": "physionet",
    },
    {
        "name":   "[PHY] EA+AdaBN+Snapshot",
        "cho":    "loso_results_physionet_ea_adabn_snapshot_cspnet.csv",
        "lee":    "loso_results_physionet_ea_adabn_snapshot_cspnet.csv",
        "metric": "snap_adabn",
        "dataset": "physionet",
    },
]

METRIC_COLS = {
    "base":      ("acc",            "kappa"),
    "adabn":     ("adabn_acc",      "adabn_kappa"),
    "tent":      ("tent_acc",       "tent_kappa"),
    "snap":      ("snap_acc",       "snap_kappa"),
    "snap_adabn":("snap_adabn_acc", "snap_adabn_kappa"),
}


# ─────────────────────────────────────────────────────────────────
# CSV 파싱
# ─────────────────────────────────────────────────────────────────
def load_results(filepath, dataset_filter=None):
    """결과 CSV → {subject_int: {col: value}} dict 반환."""
    path = RESULTS_DIR / filepath
    if not path.exists():
        return None

    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row.get("dataset", "")
            if dataset_filter and ds != dataset_filter:
                continue
            try:
                subj = int(row["subject"])
            except (KeyError, ValueError):
                continue
            rows[subj] = row
    return rows if rows else None


def get_metric(row, metric, col_type):
    """row에서 metric 값 추출. 비어있으면 base acc로 fallback."""
    acc_col, kap_col = METRIC_COLS[metric]
    col = acc_col if col_type == "acc" else kap_col
    val = row.get(col, "").strip()
    if not val:
        # fallback to base
        col = "acc" if col_type == "acc" else "kappa"
        val = row.get(col, "").strip()
    try:
        return float(val)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────
# subject_results CSV 업데이트
# ─────────────────────────────────────────────────────────────────
def create_subject_csv(path, dataset):
    """빈 subject_results CSV 생성 (Subject 컬럼만)."""
    n_total = N_SUBJECTS[dataset]
    if dataset == "physionet":
        # subjects 1-109 excluding 88, 92, 100
        skip = {88, 92, 100}
        subj_ids = [s for s in range(1, 110) if s not in skip]
    elif dataset == "cho2017":
        subj_ids = list(range(1, 53))
    else:  # lee2019
        subj_ids = list(range(1, 55))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject"])
        for s in subj_ids:
            writer.writerow([f"S{s}"])
    print(f"  {path.name} 생성 ({len(subj_ids)}명)")


def update_subject_csv(dataset, col_name, subj_data, col_type, dry_run):
    """subject_results_{dataset}_{acc|kappa}.csv 에 col_name 컬럼 추가/업데이트."""
    path = SUBJ_FILES[dataset][col_type]
    if not path.exists():
        if dry_run:
            print(f"  [dry-run] {path.name} 없음 → 생성 예정")
            return
        create_subject_csv(path, dataset)

    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    col_exists = col_name in header
    col_idx = header.index(col_name) if col_exists else len(header)

    if not col_exists:
        header.append(col_name)

    # subject 행 업데이트
    updated = 0
    for i, row in enumerate(rows[1:], 1):
        # Subject 컬럼: S01, S02, ...
        try:
            subj_num = int(row[0].lstrip("S"))
        except (ValueError, IndexError):
            continue

        val = subj_data.get(subj_num)
        if val is not None:
            pct = round(val * 100, 1) if val <= 1.0 else round(val, 1)
            entry = str(pct)
        else:
            entry = ""

        if col_exists:
            while len(row) <= col_idx:
                row.append("")
            row[col_idx] = entry
        else:
            row.append(entry)
        updated += 1

    if dry_run:
        print(f"  [dry-run] {path.name}: '{col_name}' 컬럼 {'업데이트' if col_exists else '추가'} ({updated}행)")
    else:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"  {path.name}: '{col_name}' 컬럼 {'업데이트' if col_exists else '추가'} ({updated}행)")


# ─────────────────────────────────────────────────────────────────
# 실험 통계 계산
# ─────────────────────────────────────────────────────────────────
def compute_stats(subj_data_acc, subj_data_kap, n_total):
    """mean acc, std, mean kappa 반환."""
    accs  = [v for v in subj_data_acc.values() if v is not None]
    kaps  = [v for v in subj_data_kap.values() if v is not None]
    n     = len(accs)
    if n == 0:
        return None
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    mean_kap = np.mean(kaps) if kaps else float("nan")
    return {"n": n, "total": n_total,
            "acc": mean_acc, "std": std_acc, "kappa": mean_kap}


def fmt_stat(s):
    if s is None:
        return "— | — | —"
    pct   = s["acc"] * 100 if s["acc"] <= 1 else s["acc"]
    std   = s["std"] * 100 if s["std"] <= 1 else s["std"]
    kap   = s["kappa"]
    note  = f" ({s['n']}/{s['total']})" if s["n"] < s["total"] else ""
    return f"{pct:.2f}%{note} | ±{std:.2f} | {kap:.3f}"


# ─────────────────────────────────────────────────────────────────
# PROJECT_SUMMARY.md 업데이트
# ─────────────────────────────────────────────────────────────────
def build_ea_table(results_map):
    """EA 계열 결과 테이블 마크다운 생성."""
    header = (
        "| 모델 | Cho2017 Acc | Cho2017 κ | Lee2019 Acc | Lee2019 κ | 비고 |\n"
        "|------|------------|-----------|------------|-----------|------|\n"
    )
    rows = ""
    for name, stats in results_map.items():
        sc = stats.get("cho2017")
        sl = stats.get("lee2019")

        def cell_acc(s):
            if s is None: return "—"
            pct = s["acc"] * 100 if s["acc"] <= 1 else s["acc"]
            note = f" ({s['n']}/{s['total']})" if s["n"] < s["total"] else ""
            return f"{pct:.2f}%{note} ±{s['std']*100:.2f}" if s["std"] <= 1 else f"{pct:.2f}%{note} ±{s['std']:.2f}"

        def cell_kap(s):
            if s is None: return "—"
            return f"{s['kappa']:.3f}"

        def status(sc, sl):
            if sc is None and sl is None: return "❌"
            cho_done = sc and sc["n"] == sc["total"]
            lee_done = sl and sl["n"] == sl["total"]
            if cho_done and lee_done: return "✅ 완료"
            parts = []
            if sc: parts.append(f"Cho {sc['n']}/{sc['total']}")
            if sl: parts.append(f"Lee {sl['n']}/{sl['total']}")
            return "🔄 " + ", ".join(parts)

        rows += f"| {name} | {cell_acc(sc)} | {cell_kap(sc)} | {cell_acc(sl)} | {cell_kap(sl)} | {status(sc,sl)} |\n"

    return header + rows


def update_summary_md(all_stats, dry_run):
    """PROJECT_SUMMARY.md의 EA 계열 테이블 섹션을 업데이트."""
    with open(SUMMARY_MD, "r") as f:
        content = f.read()

    # 섹션 7 (EA 계열) 테이블 교체
    ea_names = [e["name"] for e in EXPERIMENTS
                if "EA" in e["name"] and "dataset" not in e]
    ea_stats = {n: all_stats[n] for n in ea_names if n in all_stats}

    new_table = build_ea_table(ea_stats)

    # EA 계열 섹션 아래의 테이블 블록 교체
    pattern = r'(### 2\. EA 계열[^\n]*\n\n)((?:\|.*\n)+)'
    def replacer(m):
        return m.group(1) + new_table

    new_content, n = re.subn(pattern, replacer, content)
    if n == 0:
        print("  [WARN] PROJECT_SUMMARY.md 섹션 7 패턴 불일치, 수동 업데이트 필요")
        return

    # 날짜 업데이트
    today = datetime.now().strftime("%Y-%m-%d")
    new_content = re.sub(r'\*\*최종 업데이트: .*?\*\*',
                         f'**최종 업데이트: {today}**', new_content)

    if dry_run:
        print(f"  [dry-run] PROJECT_SUMMARY.md 업데이트 예정 (섹션 7 테이블)")
    else:
        with open(SUMMARY_MD, "w") as f:
            f.write(new_content)
        print(f"  PROJECT_SUMMARY.md 업데이트 완료 (날짜: {today})")


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--exp", help="특정 실험 이름만 처리")
    args = parser.parse_args()

    exps = EXPERIMENTS
    if args.exp:
        exps = [e for e in EXPERIMENTS if e["name"] == args.exp]
        if not exps:
            print(f"실험 '{args.exp}' 없음. 가능한 이름: {[e['name'] for e in EXPERIMENTS]}")
            return

    all_stats = {}

    for exp in exps:
        name   = exp["name"]
        metric = exp["metric"]
        print(f"\n{'='*50}")
        print(f"  {name}  (metric: {metric})")
        print(f"{'='*50}")

        exp_stats = {}

        if "dataset" in exp:
            datasets_to_run = [exp["dataset"]]
        else:
            datasets_to_run = ["cho2017", "lee2019"]

        for dataset in datasets_to_run:
            if dataset == "physionet":
                file_key = "cho"
            else:
                file_key = "cho" if dataset == "cho2017" else "lee"
            fname = exp[file_key]
            rows = load_results(fname, dataset_filter=dataset)

            if rows is None:
                print(f"  [{dataset}] 결과 없음 ({fname})")
                continue

            n_total  = N_SUBJECTS[dataset]
            n_done   = len(rows)
            acc_col, kap_col = METRIC_COLS[metric]

            subj_acc = {s: get_metric(r, metric, "acc")   for s, r in rows.items()}
            subj_kap = {s: get_metric(r, metric, "kappa") for s, r in rows.items()}

            stats = compute_stats(subj_acc, subj_kap, n_total)
            exp_stats[dataset] = stats

            pct = stats["acc"] * 100 if stats["acc"] <= 1 else stats["acc"]
            std = stats["std"] * 100 if stats["std"] <= 1 else stats["std"]
            print(f"  [{dataset}] {n_done}/{n_total}명 완료 | "
                  f"Acc={pct:.2f}% ±{std:.2f} | κ={stats['kappa']:.3f}")

            # subject_results CSV 업데이트
            update_subject_csv(dataset, name, subj_acc, "acc",   args.dry_run)
            update_subject_csv(dataset, name, subj_kap, "kappa", args.dry_run)

        all_stats[name] = exp_stats

    # PROJECT_SUMMARY.md 업데이트
    print(f"\n{'='*50}")
    print("  PROJECT_SUMMARY.md 업데이트")
    update_summary_md(all_stats, args.dry_run)

    print("\n완료.")


if __name__ == "__main__":
    main()
