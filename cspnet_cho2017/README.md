# CSP-Net-2: Motor Imagery EEG Classification with LOSO Evaluation

Cross-subject motor imagery (MI) EEG classification using **CSP-Net-2** — a differentiable Common Spatial Pattern network — evaluated under a strict **Leave-One-Subject-Out (LOSO)** protocol on the **Cho2017** dataset (52 subjects, 64 channels).

**Reference:** Lu, J. et al. "CSP-Net: Common Spatial Pattern Empowered Neural Networks for EEG-Based Motor Imagery Classification." *Knowledge-Based Systems* (2024). DOI: [10.1016/j.knosys.2024.112668](https://doi.org/10.1016/j.knosys.2024.112668) · [arXiv 2411.11879](https://arxiv.org/abs/2411.11879)

---

## Quick Start

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate mi_spdnet

# 2. Preprocess Cho2017 (downloads automatically via MOABB)
python preprocess_data.py --dataset cho2017

# 3. Run LOSO training with CSPNet
python train_loso.py --model cspnet --dataset cho2017
```

Results are written to `results/loso_results_<timestamp>_cspnet.csv`.

---

## Dataset

**Cho2017** is downloaded automatically by `preprocess_data.py` via [MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Cho2017.html).

| Item | Value |
|---|---|
| Subjects | 52 |
| EEG channels | 64 |
| Original sampling rate | 512 Hz → **128 Hz** (after preprocess) |
| Epoch window | [0.5, 2.5] s relative to MI onset |
| Classes | Left hand / Right hand MI |
| LOSO splits | 52 (one held-out subject per fold) |

Preprocessed data is saved to `../preprocessed/cho2017.npz` (excluded from git).

---

## Environment

```bash
conda env create -f environment.yml
conda activate mi_spdnet
```

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| MNE | 1.7.0 |
| MOABB | 1.5.0 |
| mne-icalabel | 0.6.0 |
| scipy | ≥ 1.10 |

> **GPU:** Tested on NVIDIA Quadro RTX 6000 (24 GB, CUDA 12.4). A single GPU with ≥ 6 GB VRAM is sufficient.

---

## Preprocessing

```bash
python preprocess_data.py --dataset cho2017
```

Steps applied to each subject:
1. `raw.pick("eeg")` — retain 64 EEG channels only
2. Common Average Reference (CAR)
3. ICA (Extended Infomax) — eye and muscle artifacts removed via ICLabel (threshold 0.90)
4. Bandpass filter: 8–30 Hz
5. Resample to 128 Hz
6. Epoch extraction: [0.5, 2.5] s relative to MI onset
7. Save `X (N, C, T)`, `y (N,)`, `subjects (N,)` to `.npz`

---

## Model — CSP-Net-2 (`cspnet.py`)

```
Input (B, C, T)
  └─ unsqueeze(1)
  └─ Block 1: Temporal Conv  Conv2d(1, F1, (1, kernLen)) + BN
                              → (B, F1, C, T)
  └─ Block 2: CSP Spatial    W ∈ R^(n_csp × C), differentiable
              shared across temporal filters
              + BN + ELU + AvgPool(4) + Dropout
                              → (B, F1·n_csp, 1, T//4)
  └─ Block 3: Separable Conv depthwise + pointwise + BN + ELU
              + AvgPool(8) + Dropout
                              → (B, F2, 1, T//32)
  └─ Classifier: flatten → Linear → n_classes
```

| Hyperparameter | Default | Description |
|---|---|---|
| `n_csp` | 8 | Number of CSP spatial filters |
| `F1` | 8 | Number of temporal filters |
| `F2` | 16 | Number of separable conv output channels |
| `kernel_length` | `max(16, T//4)` | Temporal conv kernel length |
| `dropout` | 0.25 | Dropout probability |
| `trainable_csp` | True | Whether CSP weights are updated by backprop |

**Total parameters:** ~3,600 (Cho2017, 64 ch, 256 time samples)

### CSP Layer Initialization

Before the gradient-descent training loop, `fit_csp_layer()` initializes `W` via generalized eigenvalue decomposition (CSP):

```
Σ_0 · w = λ · Σ_c · w,   Σ_c = Σ_0 + Σ_1
```

The `n_csp/2` smallest-λ and `n_csp/2` largest-λ eigenvectors are selected as the initial spatial filters. These weights remain fully differentiable and are updated by Adam during training.

---

## Training

```bash
# Basic CSPNet LOSO
python train_loso.py --model cspnet --dataset cho2017

# With augmentation
python train_loso.py --model cspnet --dataset cho2017 --augment

# Resume an interrupted run
python train_loso.py --model cspnet --dataset cho2017 \
    --run_id <previous_run_id> --resume
```

### Default Training Config

| Setting | Value |
|---|---|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=300) |
| Epochs | 300 |
| Batch size | 64 |
| Gradient clipping | max_norm=1.0 |
| Model selection | Best validation accuracy across epochs |
| Normalization | Per-channel z-score using training statistics |

---

## Trial Selection (Optional)

The training pool can be filtered per fold to focus on more transferable trials.

```bash
# Keep top 80% by combined heuristic score
python train_loso.py --model cspnet --dataset cho2017 \
    --keep_ratio 0.8 --selection_mode hard --score_method combined

# RHO-LOSS: validation-gradient cosine similarity (recommended)
python train_loso.py --model cspnet --dataset cho2017 \
    --keep_ratio 0.8 --selection_source rho_loss --selector_epochs 100

# Sweep keep-ratio (0.2 / 0.4 / 0.6 / 0.8 / 1.0)
python run_trial_selection_sweep.py \
    --dataset cho2017 --model cspnet \
    --score_method combined --selection_mode hard
```

| `--selection_source` | Scoring criterion |
|---|---|
| `heuristic` (default) | band_power / laterality / cov_quality / combined |
| `classifier` | trueprob / confidence / margin from a pre-trained fold model |
| `rho_loss` | Validation-gradient cosine — directly measures trial's contribution to cross-subject generalization |

> **Note:** `keep_ratio=1.0` (default) disables all selection — no performance change.

---

## Compared Baselines

This repository also includes additional models for comparison:

| Model | CLI | Notes |
|---|---|---|
| SPDNet | `--model spdnet` | Riemannian manifold (SPD matrices) |
| RiemGATNet | `--model riemgat` | Covariance + graph attention |
| MIN2Net | `--model min2net` | EEGNet encoder + reconstruction loss |
| MRFBCSP + LDA | `mrfbcsp_loso.py` | Traditional CSP baseline |

---

## Output Files

```
results/
  loso_results_<timestamp>_cspnet.csv      # Per-subject: acc, bac, kappa, time
  loso_loss_<timestamp>_cspnet.csv         # Per-epoch: train/val/test loss and acc
```

CSV columns (per-subject results): `subject`, `test_acc`, `test_bac`, `test_kappa`, `best_epoch`, `time_s`

---

## File Structure

```
cspnet_cho2017/
├── cspnet.py                   # CSP-Net-2 model + fit_csp_layer()
├── spd_net.py                  # SPDNet + shared covariance layers
├── riemgat_net.py              # RiemGATNet
├── min2net.py                  # MIN2Net
├── train_loso.py               # LOSO training loop (all models)
├── mrfbcsp_loso.py             # MRFBCSP + LDA standalone script
├── preprocess_data.py          # Preprocessing pipeline (MOABB)
├── eeg_augment.py              # Signal augmentation (jitter, scaling, noise)
├── trial_selection.py          # Trial scoring + hard/weighted selection
├── selection_viz.py            # Score visualization utilities
├── discriminator_selective.py  # Tangent AE + GRL discriminator (for trial selection)
├── run_trial_selection_sweep.py# Keep-ratio sweep automation
├── environment.yml             # Conda environment spec
└── README.md                   # This file
```

---

## References

- **CSP-Net:** Lu, J. et al. *Knowledge-Based Systems* (2024). [arXiv 2411.11879](https://arxiv.org/abs/2411.11879)
- **Cho2017:** Cho, H. et al. "EEG datasets for motor imagery brain–computer interface." *GigaScience* 6.7 (2017).
- **MOABB:** Jayaram, V. & Barachant, A. "MOABB: Trustworthy algorithm benchmarking for BCIs." *J. Neural Eng.* (2018).
- **EEGNet:** Lawhern, V.J. et al. "EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces." *J. Neural Eng.* (2018).
