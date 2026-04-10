"""
LOSO (Leave-One-Subject-Out) training and evaluation for SPDNet.

Usage:
    python train_loso.py                    # both datasets
    python train_loso.py --dataset cho2017  # only Cho2017
    python train_loso.py --dataset lee2019  # only Lee2019_MI

Results are printed per-subject and saved to results/loso_results_*.csv.

Resume support:
  --resume will load existing results for the same run_id and skip
  already-completed subjects. Results are appended per subject.
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from spd_net     import SPDNet
from riemgat_net import RiemGATNet
from min2net     import MIN2Net
from eeg_augment import EEGAugment

# -----------------------------------------------------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR    = r"g:\MI_opendata\preprocessed"
RESULTS_DIR = r"g:\MI_opendata\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
N_FILTERS  = 32
DROPOUT    = 0.5
LR         = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EPOCHS     = 300
SEED       = 2026


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

def load_data(dataset_name: str, ch_filter: str = None):
    """
    Load preprocessed data.
    ch_filter : if set, keep only channels whose name contains this string
                e.g. ch_filter='C' -> C3, Cz, C4, FC3, CP4, ...
    """
    path = os.path.join(SAVE_DIR, f"{dataset_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run preprocess_data.py first."
        )
    d = np.load(path, allow_pickle=True)
    X        = d["X"].astype(np.float32)     # (N, C, T)
    y        = d["y"].astype(np.int64)        # (N,)
    subjects = d["subjects"].astype(np.int64) # (N,)
    sfreq    = float(d["sfreq"])
    ch_names = list(d["ch_names"])

    if ch_filter is not None:
        idx = [i for i, ch in enumerate(ch_names) if ch_filter in ch]
        X        = X[:, idx, :]
        ch_names = [ch_names[i] for i in idx]
        print(f"  Channel filter '{ch_filter}': {len(idx)} channels selected")
        print(f"  -> {ch_names}")

    print(f"  Loaded {dataset_name}: X={X.shape}, "
          f"{len(np.unique(subjects))} subjects, sfreq={sfreq} Hz")
    return X, y, subjects, ch_names, sfreq


def normalize_subject(X_train, X_test):
    """Channel-wise z-score using training statistics."""
    mu  = X_train.mean(axis=(0, 2), keepdims=True)   # (1, C, 1)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_train - mu) / std, (X_test - mu) / std


# -----------------------------------------------------------------------------
# Training / Evaluation
# -----------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, augment=None):
    model.train()
    if augment is not None:
        augment.train()
    total_loss, correct, n = 0.0, 0, 0
    recon_criterion = nn.MSELoss()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if augment is not None:
            xb = augment(xb)
        optimizer.zero_grad()
        out = model(xb)
        # MIN2Net returns (logits, x_recon) during training
        if isinstance(out, tuple):
            logits, x_recon = out
            alpha = getattr(model, "alpha", 0.9)
            loss  = alpha * criterion(logits, yb) + (1 - alpha) * recon_criterion(x_recon, xb)
        else:
            logits = out
            loss   = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        pred = model(xb).argmax(1).cpu().numpy()
        all_pred.extend(pred)
        all_true.extend(yb.numpy())
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    acc  = accuracy_score(y_true, y_pred)
    bac  = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, bac, kappa, y_true, y_pred


# -----------------------------------------------------------------------------
# LOSO loop
# -----------------------------------------------------------------------------

def run_loso(dataset_name: str, ch_filter: str = None, model_name: str = "spdnet",
             use_augment: bool = False, resume: bool = False,
             out_dir: str = RESULTS_DIR, run_id: str = None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tag = f" [ch_filter='{ch_filter}']" if ch_filter else ""
    aug_tag = " +aug" if use_augment else ""
    print(f"\n{'='*60}")
    print(f" LOSO - {dataset_name.upper()}{tag}{aug_tag}")
    print(f"{'='*60}")

    X, y, subjects, ch_names, sfreq = load_data(dataset_name, ch_filter=ch_filter)
    subj_ids = np.unique(subjects)
    n_channels = X.shape[1]
    n_filters  = min(N_FILTERS, n_channels)   # BiMap: c_out must be <= c_in
    if n_filters != N_FILTERS:
        print(f"  n_filters capped to {n_filters} (n_channels={n_channels})")

    # -- Augmentation --------------------------------------------------------
    augment = None
    if use_augment:
        augment = EEGAugment(p=0.5, jitter_ms=50.0, sfreq=sfreq,
                             amp_range=(0.8, 1.2), noise_std=0.05).to(DEVICE)
        print(f"  Augmentation: {augment}")

    results   = []  # per-subject summary
    rng = np.random.default_rng(SEED)

    import time
    t_total_start = time.time()

    # -- Incremental CSV setup ------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    ch_tag2   = f"_ch{ch_filter}" if ch_filter else ""
    aug_tag2  = "_aug" if use_augment else ""
    base_tag  = ch_tag2 + f"_{model_name}" + aug_tag2
    res_fields  = ["dataset", "subject", "n_train", "n_test",
                   "acc", "bac", "kappa", "best_epoch", "time_min"]
    loss_fields = ["dataset", "subject", "epoch",
                   "train_loss", "train_acc",
                   "val_loss",   "val_acc",
                   "test_loss",  "test_acc"]

    done_subjects = set()
    res_csv = None
    loss_csv = None

    if resume:
        if run_id:
            res_csv  = os.path.join(out_dir, f"loso_results_{run_id}{base_tag}.csv")
            loss_csv = os.path.join(out_dir, f"loso_loss_{run_id}{base_tag}.csv")
            if not os.path.exists(res_csv):
                raise FileNotFoundError(
                    f"--resume requested but results not found: {res_csv}"
                )
        else:
            pattern = os.path.join(out_dir, f"loso_results_*{base_tag}.csv")
            existing = sorted(glob.glob(pattern))
            if existing:
                res_csv  = existing[-1]
                loss_csv = res_csv.replace("loso_results_", "loso_loss_")
            else:
                print("  --resume: no existing CSV found, starting fresh.")
                resume = False

    if not resume:
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        res_csv  = os.path.join(out_dir, f"loso_results_{run_id}{base_tag}.csv")
        loss_csv = os.path.join(out_dir, f"loso_loss_{run_id}{base_tag}.csv")
        if os.path.exists(res_csv) or os.path.exists(loss_csv):
            # Allow appending when the same run_id is reused (e.g., dataset=both)
            resume = True
        else:
            with open(res_csv,  "w", newline="") as f:
                csv.DictWriter(f, fieldnames=res_fields).writeheader()
            with open(loss_csv, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=loss_fields).writeheader()

    # If resuming, load completed subjects
    if resume and res_csv:
        with open(res_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("dataset") == dataset_name:
                    try:
                        done_subjects.add(int(row["subject"]))
                    except Exception:
                        pass
        # Ensure loss file has a header
        if loss_csv and (not os.path.exists(loss_csv)):
            with open(loss_csv, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=loss_fields).writeheader()
        print(f"  Resuming: {len(done_subjects)} subjects done -> {res_csv}")

    print(f"  Results -> {res_csv}")

    for i, test_subj in enumerate(subj_ids):
        if test_subj in done_subjects:
            print(f"  [{i+1:02d}/{len(subj_ids)}] S{test_subj:02d} -- skipped (already done)")
            continue
        t_subj_start = time.time()
        # -- Split: test | val (1 random train subj) | train (rest) --------
        train_pool = subj_ids[subj_ids != test_subj]
        val_subj   = rng.choice(train_pool)

        test_mask  = subjects == test_subj
        val_mask   = subjects == val_subj
        train_mask = ~test_mask & ~val_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # -- Normalise (per-channel z-score on training statistics) ---------
        mu  = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2),  keepdims=True) + 1e-8
        X_train = (X_train - mu) / std
        X_val   = (X_val   - mu) / std
        X_test  = (X_test  - mu) / std

        # -- DataLoaders ----------------------------------------------------
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
        test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                  shuffle=False, drop_last=False)

        # -- Model ----------------------------------------------------------
        if model_name == "riemgat":
            model = RiemGATNet(
                n_channels=n_channels,
                n_times=X.shape[2],
                dropout=DROPOUT,
            ).to(DEVICE)
        elif model_name == "min2net":
            # Paper default dropout=0.25; DROPOUT=0.5 is used for other models
            model = MIN2Net(
                n_channels=n_channels,
                n_times=X.shape[2],
                dropout=0.25,
            ).to(DEVICE)
        else:
            model = SPDNet(
                n_channels=n_channels,
                n_filters=n_filters,
                dropout=DROPOUT,
            ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        criterion = nn.CrossEntropyLoss()

        # -- Train ----------------------------------------------------------
        best_val_acc, best_state = -1.0, None
        epoch_log = []

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, augment=augment)
            scheduler.step()

            def _loss_acc(loader, n):
                model.eval()
                total_loss, correct = 0.0, 0
                with torch.no_grad():
                    for xb, yb in loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        logits = model(xb)
                        total_loss += criterion(logits, yb).item() * len(yb)
                        correct    += (logits.argmax(1) == yb).sum().item()
                return total_loss / n, correct / n

            val_loss,  val_acc  = _loss_acc(val_loader,  len(y_val))
            test_loss, test_acc = _loss_acc(test_loader, len(y_test))

            epoch_log.append(dict(
                dataset=dataset_name, subject=int(test_subj), epoch=epoch,
                train_loss=round(tr_loss,  6), train_acc=round(tr_acc,  4),
                val_loss=round(val_loss,   6), val_acc=round(val_acc,   4),
                test_loss=round(test_loss, 6), test_acc=round(test_acc, 4),
            ))

            # best model by validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # -- Evaluate best model --------------------------------------------
        if best_state is None:  # fallback: all epochs produced NaN val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        acc, bac, kappa, y_true, y_pred = evaluate(model, test_loader)

        elapsed = time.time() - t_subj_start

        results.append(dict(
            dataset=dataset_name,
            subject=int(test_subj),
            n_train=int(train_mask.sum()),
            n_test=int(test_mask.sum()),
            acc=acc, bac=bac, kappa=kappa,
            best_epoch=max(epoch_log, key=lambda x: x["val_acc"])["epoch"],
            time_min=round(elapsed / 60, 2),
        ))
        remaining = elapsed * (len(subj_ids) - i - 1 - len(done_subjects))
        best_ep = max(epoch_log, key=lambda x: x["val_acc"])["epoch"]
        print(f"  [{i+1:02d}/{len(subj_ids)}] S{test_subj:02d} "
              f"(val=S{val_subj:02d}) | "
              f"Acc={acc*100:.1f}%  BAcc={bac*100:.1f}%  k={kappa:.3f}  "
              f"best_ep={best_ep}  "
              f"[{elapsed/60:.1f}min | ETA {remaining/60:.0f}min]")

        # -- Incremental save (write immediately after each subject) --------
        with open(res_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=res_fields).writerow(results[-1])
        with open(loss_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=loss_fields).writerows(epoch_log)

    # -- Summary ---------------------------------------------------------------
    total_time = time.time() - t_total_start
    accs  = [r["acc"]  for r in results]
    bacs  = [r["bac"]  for r in results]
    kappas= [r["kappa"] for r in results]

    best_epochs = [r["best_epoch"] for r in results]
    print(f"\n{'-'*60}")
    print(f"  {dataset_name.upper()} LOSO Summary ({len(results)} subjects)")
    print(f"  Accuracy  : {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f} %")
    print(f"  Bal. Acc  : {np.mean(bacs)*100:.2f} ± {np.std(bacs)*100:.2f} %")
    print(f"  Cohen k   : {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")
    print(f"  Best epoch: {np.mean(best_epochs):.0f} ± {np.std(best_epochs):.0f} "
          f"(min={min(best_epochs)}, max={max(best_epochs)})")
    print(f"  Total time: {total_time/60:.1f} min  "
          f"({total_time/len(results)/60:.1f} min/subject)")
    print(f"{'-'*60}\n")

    return results


# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------

def save_results(all_results, all_loss_logs, tag=""):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -- Per-subject summary CSV --------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, f"loso_results_{ts}{tag}.csv")
    fields = ["dataset", "subject", "n_train", "n_test",
              "acc", "bac", "kappa", "best_epoch", "time_min"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)
    print(f"Results saved  -> {csv_path}")

    # -- Per-epoch loss CSV -------------------------------------------------
    loss_path = os.path.join(RESULTS_DIR, f"loso_loss_{ts}{tag}.csv")
    loss_fields = ["dataset", "subject", "epoch",
                   "train_loss", "train_acc",
                   "val_loss",   "val_acc",
                   "test_loss",  "test_acc"]
    with open(loss_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=loss_fields)
        w.writeheader()
        w.writerows(all_loss_logs)
    print(f"Loss log saved -> {loss_path}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["cho2017", "lee2019", "both"],
        default="both",
        help="Which dataset to run LOSO on (default: both)",
    )
    parser.add_argument(
        "--ch_filter", type=str, default=None,
        help="Keep only channels whose name contains this string (e.g. 'C')",
    )
    parser.add_argument(
        "--model", choices=["spdnet", "riemgat", "min2net"], default="spdnet",
        help="Model architecture (default: spdnet)",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable signal-level augmentation (time jitter + amp scaling + noise)",
    )
    parser.add_argument(
        "--out_dir", type=str, default=RESULTS_DIR,
        help="Directory to save results (default: results/)",
    )
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="Run id for output filenames (default: timestamp)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing CSV for the same run_id (or latest if run_id is not set)",
    )
    args = parser.parse_args()

    ch_tag = f"_ch{args.ch_filter}" if args.ch_filter else ""
    model_tag = f"_{args.model}"
    print(f"\nDevice    : {DEVICE}")
    print(f"Model     : {args.model}")
    print(f"Augment   : {'ON (time_jitter+amp_scale+noise)' if args.augment else 'OFF'}")
    print(f"Config    : n_filters={N_FILTERS}, lr={LR}, epochs={EPOCHS}, "
          f"batch={BATCH_SIZE}")
    if args.ch_filter:
        print(f"Ch filter : '{args.ch_filter}' channels only")
    print()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dataset in ("cho2017", "both"):
        run_loso("cho2017", ch_filter=args.ch_filter,
                 model_name=args.model, use_augment=args.augment,
                 resume=args.resume, out_dir=args.out_dir, run_id=run_id)

    if args.dataset in ("lee2019", "both"):
        run_loso("lee2019", ch_filter=args.ch_filter,
                 model_name=args.model, use_augment=args.augment,
                 resume=args.resume, out_dir=args.out_dir, run_id=run_id)
