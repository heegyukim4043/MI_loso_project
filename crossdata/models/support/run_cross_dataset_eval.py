import argparse
import copy
import csv
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

import train_loso as tl
from adabn import apply_adabn
from trial_selection import score_to_weights


def reorder_to_common_channels(X: np.ndarray, ch_names: list[str]) -> tuple[np.ndarray, list[str]]:
    target_order = [ch for ch in tl.COMMON_CHO_LEE_CHANNELS if ch in set(ch_names)]
    idx = [ch_names.index(ch) for ch in target_order]
    return X[:, idx, :], target_order


def resample_time_axis(X: np.ndarray, new_len: int) -> np.ndarray:
    if X.shape[2] == new_len:
        return X.astype(np.float32)
    old_x = np.linspace(0.0, 1.0, X.shape[2], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, new_len, dtype=np.float32)
    out = np.empty((X.shape[0], X.shape[1], new_len), dtype=np.float32)
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            out[i, c] = np.interp(new_x, old_x, X[i, c]).astype(np.float32)
    return out


def make_loader(X, y=None, weights=None, shuffle=False):
    items = [torch.from_numpy(X.astype(np.float32))]
    if y is not None:
        items.append(torch.from_numpy(y.astype(np.int64)))
    if weights is not None:
        items.append(torch.from_numpy(weights.astype(np.float32)))
    ds = TensorDataset(*items)
    return DataLoader(ds, batch_size=tl.BATCH_SIZE, shuffle=shuffle, drop_last=False)


def split_source(X, y, subjects, seed):
    rng = np.random.default_rng(seed)
    subj_ids = np.unique(subjects)
    val_subj = rng.choice(subj_ids)
    train_mask = subjects != val_subj
    val_mask = subjects == val_subj
    return (
        X[train_mask], y[train_mask], subjects[train_mask],
        X[val_mask], y[val_mask], int(val_subj),
    )


def split_subject_trials(X: np.ndarray, y: np.ndarray, seed: int, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        cls_idx = cls_idx.copy()
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_ratio)))
        if n_val >= len(cls_idx):
            n_val = max(1, len(cls_idx) // 2)
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    train_idx = np.array(sorted(train_idx), dtype=np.int64)
    val_idx = np.array(sorted(val_idx), dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Subject train/val split failed; not enough trials.")
    return train_idx, val_idx


def to_cpu_state_dict(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def compute_cov_signature(X: np.ndarray) -> np.ndarray:
    cov_sum = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for trial in X:
        xc = trial - trial.mean(axis=1, keepdims=True)
        cov = (xc @ xc.T) / max(1, xc.shape[1] - 1)
        tr = np.trace(cov)
        if tr > 0:
            cov = cov / tr
        cov_sum += cov
    return (cov_sum / len(X)).astype(np.float32).reshape(-1)


def compute_bn_signature(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=(0, 2), dtype=np.float64)
    std = X.std(axis=(0, 2), dtype=np.float64)
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def compute_subject_signature(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "bn":
        return compute_bn_signature(X)
    return compute_cov_signature(X)


def similarity_weights(signatures: list[np.ndarray], target_signature: np.ndarray, tau: float, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    dists = np.array([np.linalg.norm(sig - target_signature) for sig in signatures], dtype=np.float32)
    top_k = max(1, min(top_k, len(dists)))
    keep_idx = np.argsort(dists)[:top_k]
    scaled = -dists[keep_idx] / max(tau, 1e-6)
    scaled -= scaled.max()
    weights = np.exp(scaled)
    weights /= weights.sum().clip(min=1e-8)
    return keep_idx, weights.astype(np.float32)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, X_norm: np.ndarray) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_norm.astype(np.float32))),
        batch_size=tl.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    outputs = []
    for (xb,) in loader:
        xb = xb.to(tl.DEVICE)
        out = model(xb)
        if isinstance(out, (tuple, list)):
            out = out[0]
        outputs.append(out.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    return (
        accuracy_score(y_true, y_pred),
        balanced_accuracy_score(y_true, y_pred),
        cohen_kappa_score(y_true, y_pred),
    )


def checkpoint_dir(args) -> str:
    mode = "subject_ensemble" if args.subject_ensemble else "single_source"
    path = os.path.join(
        args.out_dir,
        f"cross_dataset_ckpts_{args.run_id}_{args.source_dataset}_to_{args.target_dataset}_{mode}",
    )
    os.makedirs(path, exist_ok=True)
    return path


def train_weighted_source_model(args, source_name, X_src, y_src, subj_src, ch_names):
    X_train, y_train, _, X_val, y_val, val_subj = split_source(X_src, y_src, subj_src, tl.SEED)
    n_channels = X_train.shape[1]
    n_times = X_train.shape[2]
    n_filters = min(tl.N_FILTERS, n_channels)

    X_train_full = X_train.copy()
    y_train_full = y_train.copy()

    mu_sel = X_train_full.mean(axis=(0, 2), keepdims=True)
    std_sel = X_train_full.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train_sel = (X_train_full - mu_sel) / std_sel
    X_val_sel = (X_val - mu_sel) / std_sel

    selector_train_loader = make_loader(X_train_sel, y_train_full, shuffle=True)
    selector_val_loader = make_loader(X_val_sel, y_val, shuffle=False)

    print(
        f"[cross-dataset] source={source_name} val_subj={val_subj} "
        f"selector=uncertainty epochs={args.selector_epochs}",
        flush=True,
    )
    selector_model, _, _ = tl.train_one_fold_model(
        model_name=args.model,
        n_channels=n_channels,
        n_times=n_times,
        n_filters=n_filters,
        X_train_prenorm=X_train_full,
        y_train_prenorm=y_train_full,
        train_loader=selector_train_loader,
        val_loader=selector_val_loader,
        test_loader=selector_val_loader,
        y_val=y_val,
        y_test=y_val,
        augment=None,
        epochs=args.selector_epochs,
    )
    scores = tl.compute_quality_entropy_scores(
        selector_model,
        X_train_sel.astype(np.float32),
        X_train_full.astype(np.float32),
        ch_names,
        entropy_lambda=args.uncertainty_lambda,
    )
    train_weights = score_to_weights(
        scores, y_train_full, args.keep_ratio, balanced=True, min_weight=args.min_weight
    )
    print(
        f"[cross-dataset] source={source_name} weighted "
        f"w_min={train_weights.min():.3f} w_mean={train_weights.mean():.3f} "
        f"w_max={train_weights.max():.3f}",
        flush=True,
    )

    mu = X_train_full.mean(axis=(0, 2), keepdims=True)
    std = X_train_full.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train_norm = (X_train_full - mu) / std
    X_val_norm = (X_val - mu) / std

    train_loader = make_loader(X_train_norm, y_train_full, train_weights, shuffle=True)
    val_loader = make_loader(X_val_norm, y_val, shuffle=False)

    model, _, epoch_log = tl.train_one_fold_model(
        model_name=args.model,
        n_channels=n_channels,
        n_times=n_times,
        n_filters=n_filters,
        X_train_prenorm=X_train_full,
        y_train_prenorm=y_train_full,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=val_loader,
        y_val=y_val,
        y_test=y_val,
        augment=None,
        epochs=tl.EPOCHS,
    )
    best_row = max(epoch_log, key=lambda x: x["val_acc"])
    return model, mu, std, best_row


def train_source_subject_model(args, source_subject, X_subj, y_subj, ch_names, ckpt_root):
    ckpt_path = os.path.join(ckpt_root, f"source_subject_{int(source_subject):02d}.pt")
    if args.resume_models and os.path.exists(ckpt_path):
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"[subject-ensemble] reuse source subject {int(source_subject):02d}", flush=True)
        return payload

    train_idx, val_idx = split_subject_trials(
        X_subj, y_subj, seed=tl.SEED + int(source_subject), val_ratio=args.subject_val_ratio
    )
    X_train_full = X_subj[train_idx].copy()
    y_train_full = y_subj[train_idx].copy()
    X_val = X_subj[val_idx].copy()
    y_val = y_subj[val_idx].copy()

    n_channels = X_subj.shape[1]
    n_times = X_subj.shape[2]
    n_filters = min(tl.N_FILTERS, n_channels)

    mu_sel = X_train_full.mean(axis=(0, 2), keepdims=True)
    std_sel = X_train_full.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train_sel = (X_train_full - mu_sel) / std_sel
    X_val_sel = (X_val - mu_sel) / std_sel
    selector_train_loader = make_loader(X_train_sel, y_train_full, shuffle=True)
    selector_val_loader = make_loader(X_val_sel, y_val, shuffle=False)

    selector_model, _, _ = tl.train_one_fold_model(
        model_name=args.model,
        n_channels=n_channels,
        n_times=n_times,
        n_filters=n_filters,
        X_train_prenorm=X_train_full,
        y_train_prenorm=y_train_full,
        train_loader=selector_train_loader,
        val_loader=selector_val_loader,
        test_loader=selector_val_loader,
        y_val=y_val,
        y_test=y_val,
        augment=None,
        epochs=args.selector_epochs,
    )
    scores = tl.compute_quality_entropy_scores(
        selector_model,
        X_train_sel.astype(np.float32),
        X_train_full.astype(np.float32),
        ch_names,
        entropy_lambda=args.uncertainty_lambda,
    )
    train_weights = score_to_weights(
        scores, y_train_full, args.keep_ratio, balanced=True, min_weight=args.min_weight
    )

    mu = X_train_full.mean(axis=(0, 2), keepdims=True)
    std = X_train_full.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train_norm = (X_train_full - mu) / std
    X_val_norm = (X_val - mu) / std
    train_loader = make_loader(X_train_norm, y_train_full, train_weights, shuffle=True)
    val_loader = make_loader(X_val_norm, y_val, shuffle=False)

    model, _, epoch_log = tl.train_one_fold_model(
        model_name=args.model,
        n_channels=n_channels,
        n_times=n_times,
        n_filters=n_filters,
        X_train_prenorm=X_train_full,
        y_train_prenorm=y_train_full,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=val_loader,
        y_val=y_val,
        y_test=y_val,
        augment=None,
        epochs=tl.EPOCHS,
    )
    best_row = max(epoch_log, key=lambda x: x["val_acc"])
    payload = {
        "source_subject": int(source_subject),
        "state_dict": to_cpu_state_dict(model),
        "mu": mu.astype(np.float32),
        "std": std.astype(np.float32),
        "signature": compute_subject_signature(X_subj, args.similarity_metric),
        "best_row": best_row,
        "n_train": int(len(y_train_full)),
        "n_val": int(len(y_val)),
    }
    torch.save(payload, ckpt_path)
    print(
        f"[subject-ensemble] trained source subject {int(source_subject):02d} "
        f"val_acc={best_row['val_acc']:.3f} ckpt={os.path.basename(ckpt_path)}",
        flush=True,
    )
    return payload


def load_subject_model(args, payload, n_channels: int, n_times: int) -> torch.nn.Module:
    model = tl.build_model(args.model, n_channels, n_times, min(tl.N_FILTERS, n_channels))
    model.load_state_dict(payload["state_dict"])
    model.to(tl.DEVICE)
    model.eval()
    return model


def evaluate_target_subjects(args, model, mu, std, target_name, X_tgt, y_tgt, subj_tgt, out_csv, best_row):
    fields = [
        "source_dataset", "target_dataset", "target_subject", "n_train_source", "n_test",
        "acc", "bac", "kappa", "best_epoch", "best_val_acc", "best_val_loss",
        "adabn_acc", "adabn_bac", "adabn_kappa", "time_min",
    ]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    rows = []
    for subject in np.unique(subj_tgt):
        t0 = time.time()
        mask = subj_tgt == subject
        X_test = X_tgt[mask]
        y_test = y_tgt[mask]
        X_test_norm = (X_test - mu) / std
        test_loader = make_loader(X_test_norm, y_test, shuffle=False)
        acc, bac, kappa, _, _ = tl.evaluate(model, test_loader)

        adabn_acc = adabn_bac = adabn_kappa = None
        if args.adabn:
            model_adapt = copy.deepcopy(model)
            apply_adabn(model_adapt, X_test_norm, tl.DEVICE, batch_size=tl.BATCH_SIZE, n_passes=args.adabn_passes)
            adabn_acc, adabn_bac, adabn_kappa, _, _ = tl.evaluate(model_adapt, test_loader)

        row = {
            "source_dataset": args.source_dataset,
            "target_dataset": target_name,
            "target_subject": int(subject),
            "n_train_source": int(args.n_train_source),
            "n_test": int(len(y_test)),
            "acc": acc,
            "bac": bac,
            "kappa": kappa,
            "best_epoch": int(best_row["epoch"]),
            "best_val_acc": float(best_row["val_acc"]),
            "best_val_loss": float(best_row["val_loss"]),
            "adabn_acc": adabn_acc,
            "adabn_bac": adabn_bac,
            "adabn_kappa": adabn_kappa,
            "time_min": round((time.time() - t0) / 60.0, 2),
        }
        rows.append(row)
        with open(out_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(
            f"[cross-dataset] {args.source_dataset}->{target_name} subject={int(subject):02d} "
            f"acc={acc*100:.1f}% k={kappa:.3f}",
            flush=True,
        )
    return rows


def evaluate_target_subjects_subject_ensemble(args, payloads, target_name, X_tgt, y_tgt, subj_tgt, out_csv):
    fields = [
        "source_dataset", "target_dataset", "target_subject", "n_source_models", "n_test",
        "similarity_metric", "top_k", "selected_source_subjects", "selected_weights",
        "acc", "bac", "kappa", "adabn_acc", "adabn_bac", "adabn_kappa", "time_min",
    ]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    signatures = [p["signature"] for p in payloads]
    rows = []
    for target_subject in np.unique(subj_tgt):
        t0 = time.time()
        mask = subj_tgt == target_subject
        X_test = X_tgt[mask]
        y_test = y_tgt[mask]
        target_signature = compute_subject_signature(X_test, args.similarity_metric)
        keep_idx, weights = similarity_weights(signatures, target_signature, args.ensemble_tau, args.ensemble_topk)

        logits_sum = None
        logits_sum_adabn = None
        selected_subjects = []
        for rel_idx, weight in zip(keep_idx, weights):
            payload = payloads[int(rel_idx)]
            selected_subjects.append(int(payload["source_subject"]))
            model = load_subject_model(args, payload, X_test.shape[1], X_test.shape[2])
            mu = payload["mu"]
            std = payload["std"]
            X_test_norm = (X_test - mu) / std
            logits = predict_logits(model, X_test_norm)
            logits_sum = logits * weight if logits_sum is None else logits_sum + logits * weight

            if args.adabn:
                model_adapt = copy.deepcopy(model)
                apply_adabn(model_adapt, X_test_norm, tl.DEVICE, batch_size=tl.BATCH_SIZE, n_passes=args.adabn_passes)
                logits_adapt = predict_logits(model_adapt, X_test_norm)
                logits_sum_adabn = (
                    logits_adapt * weight if logits_sum_adabn is None else logits_sum_adabn + logits_adapt * weight
                )
            del model

        y_pred = logits_sum.argmax(axis=1)
        acc, bac, kappa = metrics_from_preds(y_test, y_pred)

        adabn_acc = adabn_bac = adabn_kappa = None
        if logits_sum_adabn is not None:
            y_pred_adabn = logits_sum_adabn.argmax(axis=1)
            adabn_acc, adabn_bac, adabn_kappa = metrics_from_preds(y_test, y_pred_adabn)

        row = {
            "source_dataset": args.source_dataset,
            "target_dataset": target_name,
            "target_subject": int(target_subject),
            "n_source_models": int(len(payloads)),
            "n_test": int(len(y_test)),
            "similarity_metric": args.similarity_metric,
            "top_k": int(len(selected_subjects)),
            "selected_source_subjects": ";".join(str(s) for s in selected_subjects),
            "selected_weights": ";".join(f"{float(w):.4f}" for w in weights),
            "acc": acc,
            "bac": bac,
            "kappa": kappa,
            "adabn_acc": adabn_acc,
            "adabn_bac": adabn_bac,
            "adabn_kappa": adabn_kappa,
            "time_min": round((time.time() - t0) / 60.0, 2),
        }
        rows.append(row)
        with open(out_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(
            f"[subject-ensemble] {args.source_dataset}->{target_name} subject={int(target_subject):02d} "
            f"topk={len(selected_subjects)} acc={acc*100:.1f}% k={kappa:.3f}",
            flush=True,
        )
    return rows


def summarize_rows(args, rows):
    accs = np.array([r["acc"] for r in rows], dtype=np.float32)
    kappas = np.array([r["kappa"] for r in rows], dtype=np.float32)
    print(
        f"[cross-dataset] summary {args.source_dataset}->{args.target_dataset} "
        f"acc={accs.mean()*100:.2f}±{accs.std()*100:.2f}% "
        f"kappa={kappas.mean():.3f}±{kappas.std():.3f}",
        flush=True,
    )
    if rows and rows[0].get("adabn_acc") is not None:
        adabn_accs = np.array([r["adabn_acc"] for r in rows], dtype=np.float32)
        adabn_kappas = np.array([r["adabn_kappa"] for r in rows], dtype=np.float32)
        print(
            f"[cross-dataset] AdaBN {args.source_dataset}->{args.target_dataset} "
            f"acc={adabn_accs.mean()*100:.2f}±{adabn_accs.std()*100:.2f}% "
            f"kappa={adabn_kappas.mean():.3f}±{adabn_kappas.std():.3f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", choices=["cho2017", "lee2019"], required=True)
    parser.add_argument("--target_dataset", choices=["cho2017", "lee2019"], required=True)
    parser.add_argument("--model", default="cspnet", choices=["cspnet"])
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--out_dir", default="/home/hkim/MI_test/results/runs")
    parser.add_argument("--selector_epochs", type=int, default=60)
    parser.add_argument("--keep_ratio", type=float, default=0.8)
    parser.add_argument("--min_weight", type=float, default=0.5)
    parser.add_argument("--uncertainty_lambda", type=float, default=0.2)
    parser.add_argument("--ch_filter", default="common_cho_lee")
    parser.add_argument("--adabn", action="store_true")
    parser.add_argument("--adabn_passes", type=int, default=3)
    parser.add_argument("--subject_ensemble", action="store_true")
    parser.add_argument("--similarity_metric", choices=["cov", "bn"], default="cov")
    parser.add_argument("--ensemble_topk", type=int, default=5)
    parser.add_argument("--ensemble_tau", type=float, default=0.5)
    parser.add_argument("--subject_val_ratio", type=float, default=0.2)
    parser.add_argument("--resume_models", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(tl.SEED)
    np.random.seed(tl.SEED)

    X_src, y_src, subj_src, ch_names_src, _ = tl.load_data(args.source_dataset, ch_filter=args.ch_filter)
    X_tgt, y_tgt, subj_tgt, ch_names_tgt, _ = tl.load_data(args.target_dataset, ch_filter=args.ch_filter)
    X_src, ch_names_src = reorder_to_common_channels(X_src, ch_names_src)
    X_tgt, ch_names_tgt = reorder_to_common_channels(X_tgt, ch_names_tgt)
    if ch_names_src != ch_names_tgt:
        raise RuntimeError("Source/target channel sets still mismatch after canonical reorder.")
    X_tgt = resample_time_axis(X_tgt, X_src.shape[2])

    if args.subject_ensemble:
        ckpt_root = checkpoint_dir(args)
        payloads = []
        for source_subject in np.unique(subj_src):
            mask = subj_src == source_subject
            payloads.append(
                train_source_subject_model(
                    args,
                    int(source_subject),
                    X_src[mask],
                    y_src[mask],
                    ch_names_src,
                    ckpt_root,
                )
            )
        out_csv = os.path.join(
            args.out_dir,
            f"cross_dataset_results_{args.run_id}_{args.source_dataset}_to_{args.target_dataset}_"
            f"{args.model}_ch{args.ch_filter}_seluncertaintyweighted80quality_entropy_subject_ensemble.csv",
        )
        rows = evaluate_target_subjects_subject_ensemble(
            args, payloads, args.target_dataset, X_tgt, y_tgt, subj_tgt, out_csv
        )
        summarize_rows(args, rows)
        return

    model, mu, std, best_row = train_weighted_source_model(
        args, args.source_dataset, X_src, y_src, subj_src, ch_names_src
    )
    args.n_train_source = len(y_src)

    out_csv = os.path.join(
        args.out_dir,
        f"cross_dataset_results_{args.run_id}_{args.source_dataset}_to_{args.target_dataset}_"
        f"{args.model}_ch{args.ch_filter}_seluncertaintyweighted80quality_entropy.csv",
    )
    rows = evaluate_target_subjects(
        args, model, mu, std, args.target_dataset, X_tgt, y_tgt, subj_tgt, out_csv, best_row
    )
    summarize_rows(args, rows)


if __name__ == "__main__":
    main()
