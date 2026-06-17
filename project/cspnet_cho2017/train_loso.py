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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from spd_net        import SPDNet
from riemgat_net    import RiemGATNet
from min2net        import MIN2Net
from cspnet         import CSPNet, fit_csp_layer
from discriminator_selective import (
    TangentSpaceExtractor,
    TangentAutoEncoder,
    DiscriminatorSelectiveModel,
    discriminator_losses,
)
from eeg_augment    import EEGAugment
from selection_viz  import save_selection_plots
from trial_selection import score_trials, select_trials, score_to_weights

# -----------------------------------------------------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVE_DIR    = os.path.join(_BASE_DIR, "preprocessed")
RESULTS_DIR = os.path.join(_BASE_DIR, "results")
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

def build_model(model_name: str, n_channels: int, n_times: int, n_filters: int):
    """Factory for all supported LOSO models."""
    if model_name == "riemgat":
        return RiemGATNet(
            n_channels=n_channels,
            n_times=n_times,
            dropout=DROPOUT,
        ).to(DEVICE)
    if model_name == "min2net":
        return MIN2Net(
            n_channels=n_channels,
            n_times=n_times,
            dropout=0.25,
        ).to(DEVICE)
    if model_name == "cspnet":
        return CSPNet(
            n_channels=n_channels,
            n_times=n_times,
            n_csp=8,
            F1=8,
            F2=16,
            dropout=0.25,
            trainable_csp=True,
        ).to(DEVICE)
    return SPDNet(
        n_channels=n_channels,
        n_filters=n_filters,
        dropout=DROPOUT,
    ).to(DEVICE)


def init_model_for_fold(model, model_name: str, X_train_prenorm, y_train_prenorm):
    """Fold-specific initialization such as CSP fitting."""
    if model_name == "cspnet":
        fit_csp_layer(model, X_train_prenorm, y_train_prenorm)

def _apply_sample_weights(loss_vec: torch.Tensor, sample_weight=None) -> torch.Tensor:
    """Reduce a per-sample loss vector with optional sample weights."""
    if sample_weight is None:
        return loss_vec.mean()
    weight = sample_weight.to(loss_vec.device).float().view(-1)
    weight = weight / weight.sum().clamp_min(1e-8)
    return (loss_vec.view(-1) * weight).sum()


def train_epoch(model, loader, optimizer, criterion, augment=None):
    model.train()
    if augment is not None:
        augment.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        if len(batch) == 3:
            xb, yb, wb = batch
            wb = wb.to(DEVICE)
        else:
            xb, yb = batch
            wb = None
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if augment is not None:
            xb = augment(xb)
        optimizer.zero_grad()
        out = model(xb)
        # MIN2Net returns (logits, x_recon) during training
        if isinstance(out, tuple):
            logits, x_recon = out
            alpha = getattr(model, "alpha", 0.9)
            cls_loss = F.cross_entropy(logits, yb, reduction="none")
            recon_loss = F.mse_loss(x_recon, xb, reduction="none").mean(dim=(1, 2))
            loss  = alpha * _apply_sample_weights(cls_loss, wb) + \
                    (1 - alpha) * _apply_sample_weights(recon_loss, wb)
        else:
            logits = out
            cls_loss = F.cross_entropy(logits, yb, reduction="none")
            loss = _apply_sample_weights(cls_loss, wb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


def _extract_logits(out):
    """Handle models that may return auxiliary outputs."""
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def compute_classifier_scores(model, X, y, method: str = "trueprob"):
    """
    Score training trials using a trained classifier.

    Higher is better for selection.
    """
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    out_scores = []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = _extract_logits(model(xb))
        prob = torch.softmax(logits, dim=1)
        top2 = torch.topk(prob, k=min(2, prob.shape[1]), dim=1).values
        confidence = prob.max(dim=1).values
        trueprob = prob.gather(1, yb.unsqueeze(1)).squeeze(1)
        if top2.shape[1] == 2:
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = top2[:, 0]

        if method == "confidence":
            score = confidence
        elif method == "margin":
            score = margin
        else:
            score = trueprob

        out_scores.append(score.detach().cpu().numpy())
    return np.concatenate(out_scores, axis=0).astype(np.float32)


def _find_last_linear(model: nn.Module, X_probe: np.ndarray) -> nn.Linear:
    """Return the last nn.Linear that is actually called during a forward pass.

    Uses a probe forward pass so that decoder-only layers (e.g. MIN2Net's
    reconstruction head, which is skipped in eval mode) are not mistakenly
    selected over the true classification head.
    """
    all_linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not all_linears:
        raise RuntimeError("No nn.Linear layer found for rho-loss scoring.")

    last_fired: list = [None]

    def _make_hook(layer):
        def _hook(_module, _inp, _out):
            last_fired[0] = layer
        return _hook

    handles = [lin.register_forward_hook(_make_hook(lin)) for lin in all_linears]
    try:
        model.eval()
        device = next(model.parameters()).device
        x = torch.from_numpy(X_probe[:1].astype(np.float32)).to(device)
        with torch.no_grad():
            _extract_logits(model(x))
    finally:
        for h in handles:
            h.remove()

    if last_fired[0] is None:
        raise RuntimeError("No nn.Linear layer was active during forward pass.")
    return last_fired[0]


@torch.no_grad()
def _collect_last_linear_features_and_logits(model, X: np.ndarray, y: np.ndarray):
    """
    Collect penultimate features entering the final linear layer and logits.

    Returns
    -------
    feat : (N, D) float32
    logits : (N, K) float32
    y : (N,) int64
    """
    model.eval()
    last_linear = _find_last_linear(model, X)
    captured = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach())

    handle = last_linear.register_forward_hook(hook)
    try:
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )
        feats, logits_all, labels = [], [], []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = _extract_logits(model(xb))
            if not captured:
                raise RuntimeError("Failed to capture last-layer features.")
            feat = captured.pop(0)
            feats.append(feat.cpu().numpy())
            logits_all.append(logits.detach().cpu().numpy())
            labels.append(yb.numpy())
    finally:
        handle.remove()

    return (
        np.concatenate(feats, axis=0).astype(np.float32),
        np.concatenate(logits_all, axis=0).astype(np.float32),
        np.concatenate(labels, axis=0).astype(np.int64),
    )


def compute_rho_loss_scores(model, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """
    Approximate reducible hold-out loss with cosine similarity on the final layer.

    We compute:
      score_i ≈ cos(∇_{W,b} L(x_i), ∇_{W,b} L_val)

    using the classifier's final linear layer only. This is much cheaper than
    full-parameter per-trial gradients and still aligns trial ranking with
    validation-subject improvement.
    """
    feat_tr, logits_tr, y_tr = _collect_last_linear_features_and_logits(model, X_train, y_train)
    feat_val, logits_val, yv = _collect_last_linear_features_and_logits(model, X_val, y_val)

    n_classes = logits_tr.shape[1]

    prob_val = torch.softmax(torch.from_numpy(logits_val), dim=1).numpy().astype(np.float32)
    onehot_val = np.eye(n_classes, dtype=np.float32)[yv]
    delta_val = prob_val - onehot_val
    grad_w_val = (delta_val[:, :, None] * feat_val[:, None, :]).mean(axis=0)
    grad_b_val = delta_val.mean(axis=0)
    g_val = np.concatenate([grad_w_val.reshape(-1), grad_b_val.reshape(-1)], axis=0)
    g_val_norm = np.linalg.norm(g_val) + 1e-8

    prob_tr = torch.softmax(torch.from_numpy(logits_tr), dim=1).numpy().astype(np.float32)
    onehot_tr = np.eye(n_classes, dtype=np.float32)[y_tr]
    delta_tr = prob_tr - onehot_tr

    grad_w_tr = delta_tr[:, :, None] * feat_tr[:, None, :]
    grad_b_tr = delta_tr
    g_tr = np.concatenate(
        [grad_w_tr.reshape(len(X_train), -1), grad_b_tr.reshape(len(X_train), -1)],
        axis=1,
    )
    g_tr_norm = np.linalg.norm(g_tr, axis=1) + 1e-8
    cos = (g_tr @ g_val) / (g_tr_norm * g_val_norm)
    # Map to [0, 1] so that higher is always better and consistent with other selectors.
    return ((cos + 1.0) * 0.5).astype(np.float32)


@torch.no_grad()
def compute_tangent_features(X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Compute tangent-space features from raw epochs."""
    extractor = TangentSpaceExtractor().to(DEVICE)
    extractor.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    feats = []
    for (xb,) in loader:
        xb = xb.to(DEVICE)
        feats.append(extractor(xb).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def train_tangent_autoencoder(z_train: np.ndarray, epochs: int = 50,
                              batch_size: int = 256, lr: float = 1e-3):
    """Train the tangent-space autoencoder for reconstruction-based quality.

    Normalises z_train to zero-mean unit-variance before training so that
    all dimensions contribute equally to the reconstruction loss.

    Returns (model, z_mean, z_std) so that compute_quality_labels can apply
    the same normalisation.
    """
    z_mean = z_train.mean(axis=0, keepdims=True)
    z_std  = z_train.std(axis=0,  keepdims=True) + 1e-8
    z_norm = ((z_train - z_mean) / z_std).astype(np.float32)

    input_dim = z_norm.shape[1]
    hidden_dim = min(512, max(128, input_dim // 4))
    latent_dim = min(128, max(32, input_dim // 16))
    model = TangentAutoEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=0.1,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(z_norm)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model.train()
    for _ in range(epochs):
        for (zb,) in loader:
            zb = zb.to(DEVICE)
            optimizer.zero_grad()
            z_hat, _ = model(zb)
            loss = F.mse_loss(z_hat, zb)
            loss.backward()
            optimizer.step()
    return model, z_mean, z_std


@torch.no_grad()
def compute_quality_labels(autoencoder: TangentAutoEncoder, z_train: np.ndarray,
                           z_mean: np.ndarray, z_std: np.ndarray,
                           batch_size: int = 256) -> np.ndarray:
    """Compute soft quality labels from the trained autoencoder.

    z_train is normalised with z_mean / z_std before scoring so that
    the reconstruction error is on the same normalised scale.
    """
    z_norm = ((z_train - z_mean) / z_std).astype(np.float32)
    autoencoder.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(z_norm)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    labels = []
    for (zb,) in loader:
        zb = zb.to(DEVICE)
        labels.append(autoencoder.quality_label(zb).cpu().numpy())
    return np.concatenate(labels, axis=0).astype(np.float32)


def train_discriminator_selector(z_train: np.ndarray,
                                 quality_labels: np.ndarray,
                                 subject_ids: np.ndarray,
                                 epochs: int = 100,
                                 batch_size: int = 256,
                                 lr: float = 1e-3,
                                 lambda_q: float = 1.0,
                                 lambda_d: float = 1.0):
    """Train quality + GRL domain discriminator on tangent-space features.

    Normalise z_train internally so the AE/discriminator see unit-variance
    features regardless of the raw tangent-space scale.
    """
    # Normalise tangent features to zero-mean, unit-variance per dimension
    z_mean = z_train.mean(axis=0, keepdims=True)
    z_std  = z_train.std(axis=0,  keepdims=True) + 1e-8
    z_norm = ((z_train - z_mean) / z_std).astype(np.float32)

    subj_unique = np.unique(subject_ids)
    subj_map = {sid: idx for idx, sid in enumerate(subj_unique.tolist())}
    subj_idx = np.array([subj_map[int(s)] for s in subject_ids], dtype=np.int64)

    input_dim = z_norm.shape[1]
    model = DiscriminatorSelectiveModel(
        input_dim=input_dim,
        n_subjects=len(subj_unique),
        hidden_dim=min(512, max(128, input_dim // 4)),
        embed_dim=min(256, max(64, input_dim // 8)),
        dropout=0.1,
        lambda_grl=1.0,   # GRL strength fixed at 1; lambda_d scales only the loss term
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(z_norm),                             # normalised
            torch.from_numpy(quality_labels.astype(np.float32)),
            torch.from_numpy(subj_idx),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model.train()
    for _ in range(epochs):
        for zb, qlb, sb in loader:
            zb = zb.to(DEVICE)
            qlb = qlb.to(DEVICE)
            sb = sb.to(DEVICE)
            optimizer.zero_grad()
            out = model(zb)
            losses = discriminator_losses(
                quality_logit=out["quality_logit"],
                quality_label=qlb,
                domain_logits=out["domain_logits"],
                subject_id=sb,
                lambda_q=lambda_q,
                lambda_d=lambda_d,
            )
            losses["total"].backward()
            optimizer.step()
    # Return model together with the normalisation stats so that
    # compute_discriminator_scores can apply the same transform.
    return model, z_mean, z_std


@torch.no_grad()
def compute_discriminator_scores(model: DiscriminatorSelectiveModel,
                                 z_train: np.ndarray,
                                 z_mean: np.ndarray,
                                 z_std: np.ndarray,
                                 method: str = "final",
                                 batch_size: int = 256) -> np.ndarray:
    """Score trials with discriminator outputs.

    z_train is normalised with the same z_mean / z_std used during training
    before being fed to the model.
    """
    z_norm = ((z_train - z_mean) / z_std).astype(np.float32)
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(z_norm)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    scores = []
    for (zb,) in loader:
        zb = zb.to(DEVICE)
        out = model(zb)
        if method == "quality":
            score = model.quality_score(out["quality_logit"])
        elif method == "domain":
            score = model.domain_score(out["domain_logits"])
        else:
            score = model.final_score(out["quality_logit"], out["domain_logits"])
        scores.append(score.detach().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float32)


def _loss_acc(model, loader, criterion, n):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / n, correct / n


def train_one_fold_model(model_name, n_channels, n_times, n_filters,
                         X_train_prenorm, y_train_prenorm,
                         train_loader, val_loader, test_loader,
                         y_val, y_test, augment=None, epochs=EPOCHS):
    """Train a fold model and return the best checkpoint and logs."""
    model = build_model(model_name, n_channels, n_times, n_filters)
    init_model_for_fold(model, model_name, X_train_prenorm, y_train_prenorm)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = -1.0, None
    epoch_log = []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, augment=augment)
        scheduler.step()

        val_loss, val_acc = _loss_acc(model, val_loader, criterion, len(y_val))
        test_loss, test_acc = _loss_acc(model, test_loader, criterion, len(y_test))

        epoch_log.append(dict(
            epoch=epoch,
            train_loss=round(tr_loss, 6), train_acc=round(tr_acc, 4),
            val_loss=round(val_loss, 6), val_acc=round(val_acc, 4),
            test_loss=round(test_loss, 6), test_acc=round(test_acc, 4),
        ))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, criterion, epoch_log


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
             out_dir: str = RESULTS_DIR, run_id: str = None,
             keep_ratio: float = 1.0, score_method: str = "combined",
             selection_mode: str = "hard", min_weight: float = 0.25,
             selection_source: str = "heuristic", selector_epochs: int = 100,
             lambda_q: float = 1.0, lambda_d: float = 1.0,
             save_selection_plots_flag: bool = False,
             selection_plot_max_points: int = 4000):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tag = f" [ch_filter='{ch_filter}']" if ch_filter else ""
    aug_tag = " +aug" if use_augment else ""
    sel_tag = ""
    if keep_ratio < 1.0:
        sel_tag = (f" [sel={selection_source}/{selection_mode}:"
                   f"{int(keep_ratio*100)}% {score_method}]")
    print(f"\n{'='*60}")
    print(f" LOSO - {dataset_name.upper()}{tag}{aug_tag}{sel_tag}")
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
    sel_tag2  = (f"_sel{selection_source}{selection_mode}{int(keep_ratio*100)}{score_method}"
                 if keep_ratio < 1.0 else "")
    base_tag  = ch_tag2 + f"_{model_name}" + aug_tag2 + sel_tag2
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

    selection_plot_dir = os.path.join(out_dir, f"selection_plots_{run_id}{base_tag}")
    if save_selection_plots_flag:
        os.makedirs(selection_plot_dir, exist_ok=True)

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
        train_subjects = subjects[train_mask]

        n_before = len(y_train)
        X_train_full = X_train.copy()
        y_train_full = y_train.copy()
        train_weights = None

        # Keep pre-normalised copy for CSP filter fitting and classifier-first scoring
        X_train_prenorm = X_train.copy()
        y_train_prenorm = y_train.copy()

        if keep_ratio < 1.0:
            if selection_source in ("classifier", "rho_loss"):
                # Stage 1: train a baseline classifier on the full train pool.
                mu_sel = X_train_full.mean(axis=(0, 2), keepdims=True)
                std_sel = X_train_full.std(axis=(0, 2), keepdims=True) + 1e-8
                X_train_sel = (X_train_full - mu_sel) / std_sel
                X_val_sel = (X_val - mu_sel) / std_sel
                X_test_sel = (X_test - mu_sel) / std_sel

                selector_train_ds = TensorDataset(
                    torch.from_numpy(X_train_sel), torch.from_numpy(y_train_full)
                )
                selector_val_ds = TensorDataset(
                    torch.from_numpy(X_val_sel), torch.from_numpy(y_val)
                )
                selector_test_ds = TensorDataset(
                    torch.from_numpy(X_test_sel), torch.from_numpy(y_test)
                )

                selector_train_loader = DataLoader(selector_train_ds, batch_size=BATCH_SIZE,
                                                   shuffle=True, drop_last=False)
                selector_val_loader = DataLoader(selector_val_ds, batch_size=BATCH_SIZE,
                                                 shuffle=False, drop_last=False)
                selector_test_loader = DataLoader(selector_test_ds, batch_size=BATCH_SIZE,
                                                  shuffle=False, drop_last=False)

                stage_name = "rho_loss" if selection_source == "rho_loss" else "classifier"
                print(f"    [sel-{stage_name}] training selector model "
                      f"(epochs={selector_epochs}, score={score_method})")
                selector_model, _, _ = train_one_fold_model(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_times=X.shape[2],
                    n_filters=n_filters,
                    X_train_prenorm=X_train_full,
                    y_train_prenorm=y_train_full,
                    train_loader=selector_train_loader,
                    val_loader=selector_val_loader,
                    test_loader=selector_test_loader,
                    y_val=y_val,
                    y_test=y_test,
                    augment=None,
                    epochs=selector_epochs,
                )
                if selection_source == "rho_loss":
                    if score_method != "rho_cosine":
                        print(f"    [WARN] score_method='{score_method}' not valid for "
                              f"rho_loss; using 'rho_cosine'")
                    scores = compute_rho_loss_scores(
                        selector_model,
                        X_train_sel.astype(np.float32),
                        y_train_full,
                        X_val_sel.astype(np.float32),
                        y_val,
                    )
                else:
                    scores = compute_classifier_scores(
                        selector_model, X_train_sel.astype(np.float32), y_train_full, method=score_method
                    )
            elif selection_source == "discriminator":
                print(f"    [sel-discriminator] tangent AE + GRL discriminator "
                      f"(epochs={selector_epochs}, score={score_method})")
                z_train = compute_tangent_features(X_train_full)
                ae_epochs = max(20, selector_epochs // 2)
                autoencoder, ae_mean, ae_std = train_tangent_autoencoder(
                    z_train,
                    epochs=ae_epochs,
                    batch_size=max(BATCH_SIZE, 256),
                    lr=1e-3,
                )
                quality_labels = compute_quality_labels(
                    autoencoder,
                    z_train,
                    z_mean=ae_mean,
                    z_std=ae_std,
                    batch_size=max(BATCH_SIZE, 256),
                )
                discriminator, z_mean, z_std = train_discriminator_selector(
                    z_train=z_train,
                    quality_labels=quality_labels,
                    subject_ids=train_subjects,
                    epochs=selector_epochs,
                    batch_size=max(BATCH_SIZE, 256),
                    lr=1e-3,
                    lambda_q=lambda_q,
                    lambda_d=lambda_d,
                )
                # Validate score_method for discriminator; warn and fall back to "final"
                if score_method not in ("quality", "domain", "final"):
                    print(f"    [WARN] score_method='{score_method}' not valid for "
                          f"discriminator; using 'final'")
                    disc_method = "final"
                else:
                    disc_method = score_method
                scores = compute_discriminator_scores(
                    discriminator,
                    z_train,
                    z_mean=z_mean,
                    z_std=z_std,
                    method=disc_method,
                    batch_size=max(BATCH_SIZE, 256),
                )
            else:
                scores = score_trials(X_train_full, ch_names, method=score_method)

            if save_selection_plots_flag:
                stem = (
                    f"{dataset_name}_S{int(test_subj):02d}_valS{int(val_subj):02d}_"
                    f"{selection_source}_{selection_mode}_{int(keep_ratio * 100)}_{score_method}"
                )
                title_prefix = (
                    f"{dataset_name} S{int(test_subj):02d} "
                    f"{selection_source}/{selection_mode} keep={int(keep_ratio * 100)}%"
                )
                save_selection_plots(
                    scores=scores,
                    y=y_train_full,
                    keep_ratio=keep_ratio,
                    out_dir=selection_plot_dir,
                    stem=stem,
                    balanced=True,
                    title_prefix=title_prefix,
                    max_points=selection_plot_max_points,
                )

            if selection_mode == "hard":
                X_train, y_train = select_trials(
                    X_train_full, y_train_full, scores, keep_ratio, balanced=True
                )
                print(f"    [sel-{selection_source}/hard] {n_before} -> {len(y_train)} trials "
                      f"(kept {len(y_train)/n_before*100:.0f}%, method={score_method})")
            else:
                X_train, y_train = X_train_full, y_train_full
                train_weights = score_to_weights(
                    scores, y_train, keep_ratio, balanced=True, min_weight=min_weight
                )
                print(f"    [sel-{selection_source}/weighted] {n_before} trials "
                      f"(w_min={train_weights.min():.3f}, "
                      f"w_mean={train_weights.mean():.3f}, "
                      f"w_max={train_weights.max():.3f}, method={score_method})")

            X_train_prenorm = X_train.copy()
            y_train_prenorm = y_train.copy()

        # -- Normalise (per-channel z-score on training statistics) ---------
        mu  = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2),  keepdims=True) + 1e-8
        X_train = (X_train - mu) / std
        X_val   = (X_val   - mu) / std
        X_test  = (X_test  - mu) / std

        # -- DataLoaders ----------------------------------------------------
        if train_weights is None:
            train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        else:
            train_ds = TensorDataset(
                torch.from_numpy(X_train),
                torch.from_numpy(y_train),
                torch.from_numpy(train_weights.astype(np.float32)),
            )
        val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
        test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                  shuffle=False, drop_last=False)

        model, criterion, raw_epoch_log = train_one_fold_model(
            model_name=model_name,
            n_channels=n_channels,
            n_times=X.shape[2],
            n_filters=n_filters,
            X_train_prenorm=X_train_prenorm,
            y_train_prenorm=y_train_prenorm,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            y_val=y_val,
            y_test=y_test,
            augment=augment,
            epochs=EPOCHS,
        )
        epoch_log = [
            dict(dataset=dataset_name, subject=int(test_subj), **row)
            for row in raw_epoch_log
        ]

        acc, bac, kappa, y_true, y_pred = evaluate(model, test_loader)

        elapsed = time.time() - t_subj_start

        results.append(dict(
            dataset=dataset_name,
            subject=int(test_subj),
            n_train=int(len(y_train)),
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
        "--model", choices=["spdnet", "riemgat", "min2net", "cspnet"], default="spdnet",
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
    parser.add_argument(
        "--keep_ratio", type=float, default=1.0,
        help="Fraction of train pool trials to keep by quality score "
             "(1.0 = no selection, 0.6 = keep top 60%%). "
             "Class balance is preserved within each fold.",
    )
    parser.add_argument(
        "--score_method",
        choices=["band_power", "laterality", "cov_quality", "combined",
                 "confidence", "trueprob", "margin",
                 "quality", "domain", "final", "rho_cosine"],
        default="combined",
        help="Trial scoring method for --keep_ratio selection. "
             "Heuristic source: band_power/laterality/cov_quality/combined. "
             "Classifier source: confidence/trueprob/margin. "
             "Discriminator source: quality/domain/final. "
             "RHO-LOSS source: rho_cosine. "
             "(default: combined)",
    )
    parser.add_argument(
        "--selection_mode",
        choices=["hard", "weighted"],
        default="hard",
        help="How to use suitability scores when keep_ratio < 1.0 "
             "(default: hard)",
    )
    parser.add_argument(
        "--min_weight", type=float, default=0.25,
        help="Minimum per-trial weight for weighted selection mode "
             "(default: 0.25)",
    )
    parser.add_argument(
        "--selection_source",
        choices=["heuristic", "classifier", "discriminator", "rho_loss"],
        default="heuristic",
        help="Where trial-selection scores come from when keep_ratio < 1.0 "
             "(default: heuristic)",
    )
    parser.add_argument(
        "--selector_epochs", type=int, default=100,
        help="Pretraining epochs for classifier/discriminator-based trial scoring "
             "(default: 100)",
    )
    parser.add_argument(
        "--lambda_q", type=float, default=1.0,
        help="Quality-loss weight for discriminator-based selection (default: 1.0)",
    )
    parser.add_argument(
        "--lambda_d", type=float, default=1.0,
        help="Domain-loss / GRL weight for discriminator-based selection (default: 1.0)",
    )
    parser.add_argument(
        "--save_selection_plots", action="store_true",
        help="Save ranked/distribution PNGs for fold-level selection scores "
             "(only used when keep_ratio < 1.0)",
    )
    parser.add_argument(
        "--selection_plot_max_points", type=int, default=4000,
        help="Maximum ranked points to draw in selection plots (default: 4000)",
    )
    args = parser.parse_args()

    ch_tag = f"_ch{args.ch_filter}" if args.ch_filter else ""
    model_tag = f"_{args.model}"
    print(f"\nDevice    : {DEVICE}")
    print(f"Model     : {args.model}")
    print(f"Augment   : {'ON (time_jitter+amp_scale+noise)' if args.augment else 'OFF'}")
    print(f"Selection : "
          f"{'source=' + args.selection_source + ' mode=' + args.selection_mode + ' keep=' + str(int(args.keep_ratio*100)) + '% method=' + args.score_method if args.keep_ratio < 1.0 else 'OFF (keep_ratio=1.0)'}")
    print(f"Config    : n_filters={N_FILTERS}, lr={LR}, epochs={EPOCHS}, "
          f"batch={BATCH_SIZE}")
    if args.ch_filter:
        print(f"Ch filter : '{args.ch_filter}' channels only")
    print()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dataset in ("cho2017", "both"):
        run_loso("cho2017", ch_filter=args.ch_filter,
                 model_name=args.model, use_augment=args.augment,
                 resume=args.resume, out_dir=args.out_dir, run_id=run_id,
                 keep_ratio=args.keep_ratio, score_method=args.score_method,
                 selection_mode=args.selection_mode, min_weight=args.min_weight,
                 selection_source=args.selection_source,
                 selector_epochs=args.selector_epochs,
                 lambda_q=args.lambda_q, lambda_d=args.lambda_d,
                 save_selection_plots_flag=args.save_selection_plots,
                 selection_plot_max_points=args.selection_plot_max_points)

    if args.dataset in ("lee2019", "both"):
        run_loso("lee2019", ch_filter=args.ch_filter,
                 model_name=args.model, use_augment=args.augment,
                 resume=args.resume, out_dir=args.out_dir, run_id=run_id,
                 keep_ratio=args.keep_ratio, score_method=args.score_method,
                 selection_mode=args.selection_mode, min_weight=args.min_weight,
                 selection_source=args.selection_source,
                 selector_epochs=args.selector_epochs,
                 lambda_q=args.lambda_q, lambda_d=args.lambda_d,
                 save_selection_plots_flag=args.save_selection_plots,
                 selection_plot_max_points=args.selection_plot_max_points)
