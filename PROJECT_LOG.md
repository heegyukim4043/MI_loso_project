# MI (Motor Imagery) EEG Classification ??Project Log

## Objective

Evaluate Riemannian geometry-based deep learning models for Motor Imagery BCI classification
using publicly available EEG datasets (Cho2017, Lee2019_MI), and compare various architectures
and preprocessing strategies under a rigorous LOSO (Leave-One-Subject-Out) protocol.

---

## Datasets

| Item | Cho2017 | Lee2019_MI |
|---|---|---|
| Subjects | 52 | 54 |
| Original sampling rate | 512 Hz | 1000 Hz |
| After preprocessing | 128 Hz | 100 Hz |
| EEG channels | 64 | 62 |
| Sessions | 1 | 2 |
| Classes | Left / Right hand MI | Left / Right hand MI |
| Download | MOABB (`Cho2017`) | MOABB (`Lee2019_MI`) |
| Saved path | `g:\MI_opendata\preprocessed\cho2017.npz` | `g:\MI_opendata\preprocessed\lee2019.npz` |

---

## Environment

- **Conda env**: `mi_spdnet` (miniforge3)
- Python 3.10 / PyTorch 2.1.2+cu118 / MNE 1.7.0 / MOABB 1.1.0 / mne-icalabel 0.6.0
- GPU: NVIDIA GTX 1080 Ti (CUDA 11.8)
- Execution prefix: `C:/Users/Bio_lab_HG/miniforge3/envs/mi_spdnet/python.exe`

---

## Preprocessing Pipeline (`preprocess_data.py`)

1. Load raw EEG via MOABB
2. `raw.pick("eeg")` ??retain EEG channels only (drop EMG, Stim)
3. Apply CAR (Common Average Reference)
4. **Run ICA before resampling** ??avoids h_freq > Nyquist error
   - Cho2017: concatenate per subject, then ICA
   - Lee2019: separate ICA per session (session1 / session2 independently)
   - Extended Infomax; ICLabel removes eye/muscle components >= 90% probability
5. Resample: Cho2017 -> 128 Hz, Lee2019 -> 100 Hz
6. Bandpass filter: 8-30 Hz
7. Extract epochs: **[0.5, 2.5]s** relative to MI stimulus onset
8. Save: `X(N, C, T)`, `y(N,)`, `subjects(N,)`, `sfreq`, `ch_names` -> `.npz`

### Key Fixes

| Fix | Reason |
|---|---|
| Epoch window `[2.0, 4.0]s` -> `[0.5, 2.5]s` | Cho2017 MI stimulus spans 0-3s; the old window captured 1s MI + 1s rest -> **+5% accuracy** |
| Added `raw.pick("eeg")` | Preprocessed npz contained 69 channels (64 EEG + 4 EMG + 1 Stim) |
| ICA before resampling | Bandpass upper limit exceeded Nyquist after resampling |

---

## Model Architectures

### 1. SPDNet (`spd_net.py`)

Classification on the Riemannian manifold of Symmetric Positive Definite (SPD) matrices.

```
Input (B, C, T)
  -> CovarianceLayer        : (B, C, C)  ??trace-normalized Tikhonov regularization (eps=1e-5)
  -> BiMapLayer x N         : (B, c_out, c_out)  ??bilinear map W^T Sigma W
  -> ReEigLayer             : eigenvalue clamp (eps=1e-4)
  -> LogMapLayer            : matrix logarithm (eps=1e-7)
  -> VectorizeLayer         : upper triangle -> flat vector
  -> MLP (128 -> 64 -> 2)  : BN + ELU + Dropout
```

- Constraint: `n_filters = min(N_FILTERS, n_channels)` ??BiMap requires c_out <= c_in

### 2. RiemGATNet (`riemgat_net.py`)

Riemannian Covariance + Dynamic Graph Attention Network. ~245K parameters.

```
Input (B, C, T)
  -> TemporalEncoder        : shared 1D CNN per channel -> (B, C, d_node=64)
  -> CovarianceLayer        : (B, C, C)
  -> LogMapLayer            : log-covariance as attention bias
  -> DynamicGATLayer x 3   : multi-head attention (n_heads=4) + residual + LayerNorm
  -> DeepConvBlock          : 1D CNN (B, C, d_gat) -> (B, d_conv=128)
  -> Linear -> 2 classes
```

- `adj_scale` initialized to 0 (no structural prior; fully learned)
- Gradient clipping: `max_norm=1.0`

### 3. MIN2Net (`min2net.py`)

Multi-task learning model; EEGNet-based encoder shared between classifier and reconstructor heads.
Reference: Autthasan et al., IEEE TNSRE 2022.

```
Input (B, C, T)
  -> Block1 (temporal conv, kernel ~T/2)  : (B, F1, C, T)
  -> Block2 (depthwise spatial conv)      : (B, D*F1, 1, T//4)
  -> Block3 (separable conv)              : (B, F2, 1, T//32)
  -> Bottleneck FC                        : (B, latent_dim=64)
       |---> Classifier head             : (B, n_classes)
       |---> Decoder MLP                 : (B, C, T)   [training only]

Loss = alpha * CE(classifier) + (1 - alpha) * MSE(reconstructed, original)
       default alpha = 0.9
```

- ~8.5M parameters (decoder FC dominates for large C x T)
- Integrated into `train_loso.py` via `--model min2net`

### 4. MRFBCSP + LDA (`mrfbcsp_loso.py`)

Multi-Resolution Filter Bank CSP with LDA classifier. Traditional ML baseline.
Reference: "Selective Subject Pooling for BCI", Sensors 2021 (MDPI).

```
Input (N, C, T)
  -> Filter bank (10 sub-bands, 4 Hz wide, 2 Hz step, 8-30 Hz):
       [8-12, 10-14, 12-16, 14-18, 16-20, 18-22, 20-24, 22-26, 24-28, 26-30]
  -> CSP per band (n_components=4)        : log-variance features
  -> Concatenate all bands                : (N, 10 x n_csp = 40)
  -> LDA (lsqr + Ledoit-Wolf shrinkage)  : 2-class decision
```

- Standalone script; does not use GPU
- ~seconds per subject

---

## Training Strategy (`train_loso.py`)

- **Evaluation**: LOSO (Leave-One-Subject-Out)
- **Validation**: 1 subject randomly held out from training pool for model selection
- **Model selection**: based on `val_acc` (robust to NaN val_loss)
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=300)
- EPOCHS=300, BATCH_SIZE=64, SEED=2026
- Gradient clipping: `max_norm=1.0`
- Supported models via `--model`: `spdnet`, `riemgat`, `min2net`
- Augmentation enabled via `--augment` flag (EEGAugment applied per batch during training)
- MIN2Net multi-task loss handled inside `train_epoch` (detects tuple output automatically)
- MIN2Net uses `dropout=0.25` (paper default); other models use `DROPOUT=0.5`
- `best_epoch` in CSV is selected by `val_acc` (not test_acc); test loss/acc per epoch is logged for analysis only and does not affect model selection or weights

---

## Augmentation (`eeg_augment.py`)

Signal-level stochastic augmentation applied during training (each technique independently at p=0.5):

| Technique | Implementation | Parameters |
|---|---|---|
| Time Jitter | Cyclic roll along time axis | +/-50 ms (+/-6 samples @ 128 Hz) |
| Per-channel Amplitude Scaling | U(0.8, 1.2) per channel | Alters correlation structure even after trace normalization |
| Noise Injection | Additive N(0, 0.05^2) | Stabilizes minimum eigenvalue of SPD matrix |
| Frequency Shift | **Not applied** | Destroys mu/beta band structure essential for MI |

`RiemannianMixup` class also implemented (geodesic interpolation on the SPD manifold):

```
Sigma_mix = A^(1/2) (A^(-1/2) B A^(-1/2))^t A^(1/2),   t ~ Beta(0.4, 0.4)
Loss = t * CE(y_a) + (1-t) * CE(y_b)
```

---

## ERD Analysis (`analyze_erd.py`)

- TFR multitaper, baseline=(-1, 0)s, active=(0.5, 2.5)s
- ERD strength computed for mu (8-13 Hz) and beta (13-30 Hz) bands per subject
- ERD vs. LOSO accuracy correlation: **r = 0.059, p = 0.675** (no significant correlation)
  - Likely reason: all subject accuracies clustered near ~55% (low variance), limiting correlation

---

## Results

### Cho2017 (LOSO, 52 subjects, epoch [0.5, 2.5]s)

| Model | Accuracy (mean +/- std) | Cohen kappa | Notes |
|---|---|---|---|
| CSP + LDA | 55.79 +/- 9.53% | — | Incorrect window [2.0, 4.0]s |
| CSP + LDA | 60.55 +/- 10.78% | — | Corrected window, baseline |
| SPDNet | ~55-57% | ~0.10-0.15 | Early runs (BiMap bug) |
| RiemGATNet | 55.18 +/- 7.84% | — | No augmentation |
| SPDNet | 57.17 +/- 9.53% | — | After BiMap fix |
| **SPDNet + Augment** | **61.23 +/- 10.08%** | **0.225** | Current best (Cho2017) |

### Lee2019 / OpenBMI (LOSO, 54 subjects, epoch [0.5, 2.5]s)

| Model | Accuracy (mean +/- std) | Cohen kappa | Total time | Literature SOTA |
|---|---|---|---|---|
| **MIN2Net** | **69.73 +/- 12.51%** | **0.395** | 4947.6 min | 72.03 +/- 14.04% |
| MRFBCSP+LDA | 67.74 +/- 13.14% | 0.355 | 210.5 min | 69.36% |

Both results are within ~2% of reported literature SOTA. Protocol differences (epoch window, ICA, channel set) account for most of the gap.

---

## File Structure

```
g:\MI_opendata\
????? preprocess_data.py          # Full preprocessing pipeline
????? preprocess_lee_missing.py   # Supplemental preprocessing for missing Lee2019 subjects
????? spd_net.py                  # SPDNet model
????? riemgat_net.py              # RiemGATNet model
????? min2net.py                  # MIN2Net model (EEGNet encoder + multi-task loss)
????? eeg_augment.py              # Signal-level (EEGAugment) and Riemannian (RiemannianMixup) augmentation
????? train_loso.py               # LOSO training and evaluation loop (spdnet / riemgat / min2net)
????? mrfbcsp_loso.py             # MRFBCSP+LDA standalone LOSO script
????? analyze_erd.py              # ERD strength analysis
????? download_cho2017.py         # MOABB download script
????? download_lee2019_mi.py      # MOABB download script
????? environment.yml             # Conda environment definition
????? PROJECT_LOG.md              # This file
????? preprocessed/
??  ????? cho2017.npz             # X(10520, 64, 257), 52 subjects
??  ????? lee2019.npz             # X(10800, 62, 201), 54 subjects
????? results/
    ????? loso_results_*.csv      # Per-subject: acc, bac, kappa, time  (train_loso.py)
    ????? loso_loss_*.csv         # Per-epoch: train/val/test loss and acc  (train_loso.py)
    ????? mrfbcsp_results_*.csv   # Per-subject: acc, bac, kappa, time  (mrfbcsp_loso.py)
    ????? erd_raw_*.csv           # Per-subject ERD raw values
    ????? erd_summary_*.csv       # ERD summary statistics
```

---

## Bug Fix History

| Error | Cause | Fix |
|---|---|---|
| best_epoch uses test_acc | CSV best_epoch referenced test_acc even though model selection used val_acc | Switched best_epoch to val_acc |
| `h_freq > Nyquist` | Bandpass applied after resampling | Run ICA and bandpass before resampling |
| BiMapLayer einsum error | Redundant lines in forward pass | Simplified to single `W.t().unsqueeze(0) @ x @ W.unsqueeze(0)` |
| `best_state = None` | NaN val_loss caused model selection to never update | Switch model selection to `val_acc` |
| UnicodeEncodeError (cp949) | Box-drawing characters in print statements | Replaced with ASCII equivalents |
| Lee2019: only 41/54 subjects loaded | Corrupted .mat files from partial downloads | Re-downloaded 20 files; preprocessed subjects 42-54 separately and merged |
| `n_filters > n_channels` | BiMap c_out <= c_in constraint violated | `n_filters = min(N_FILTERS, n_channels)` |
| 69 channels in npz (EEG+EMG+Stim) | Missing `raw.pick("eeg")` call | Added to `process_session()`, re-preprocessed both datasets |
| Epoch window [2.0, 4.0]s | Mismatch with Cho2017 MI stimulus range (0-3s) | Fixed to [0.5, 2.5]s; CSP+LDA improved by ~5% |
| `name in ep` checking channel types | Should check event_id dict | Fixed to `name in ep.event_id` |

---

## Pending Tasks

- [x] Run MRFBCSP+LDA LOSO on Lee2019 — 67.74 +/- 13.14%
- [x] Run MIN2Net LOSO on Lee2019 — 69.73 +/- 12.51%
- [ ] Run MRFBCSP+LDA LOSO on Cho2017
- [ ] Run MIN2Net LOSO on Cho2017
- [ ] RiemGATNet + Augmentation on Cho2017
- [ ] SPDNet + Augment on Lee2019
- [ ] Integrate RiemannianMixup (manifold-level) into training loop
- [ ] Cross-dataset comparison summary (all models)

---

## LOSO Baselines (Project + Literature)

### Cho2017 (2-class, LOSO)

Internal project baseline (same preprocessing + LOSO):
- CSP + LDA, epoch [0.5, 2.5]s: 60.55 +/- 10.78% (current baseline)
- SPDNet + Augment, epoch [0.5, 2.5]s: 61.23 +/- 10.08% (current best)

Literature note:
- No clear 2-class LOSO baseline found for Cho2017. Most published cross-subject results use 4-class and 10-fold cross-subject splits, which are not directly comparable.

### Lee2019_MI / OpenBMI (2-class, LOSO)

Literature baselines (subject-leave-one-out, 2-class):
- EEGNet: 68.84 +/- 13.87
- DeepConvNet: 68.33 +/- 15.33
- MIN2Net: 72.03 +/- 14.04
- MSAENet: 68.31 +/- 14.28

Project status:
- LOSO baseline on Lee2019 not yet run; planned to reproduce with the same preprocessing and eval protocol used for Cho2017.

---

## High-Impact Next Attempts (To Improve LOSO)

1. Enforce strict fold-wise preprocessing (ICA, normalization, resampling, covariance stats) to avoid any test-subject leakage.
2. Add filterbank covariance features (e.g., 4-6 sub-bands between 8-30 Hz) and fuse in SPDNet (late fusion or channel concat before CovLayer).
3. Add subject-wise batch mixing and domain generalization loss (CORAL/MMD) on log-cov features.
4. Integrate RiemannianMixup into train_loso.py and compare to current signal-level augmentation only.
5. Try session-wise alignment for Lee2019 (EA or re-centering of covariance) before SPDNet.
6. Use model selection with val_acc + val_bac (or kappa) to reduce bias from small val sets.
7. Run RiemGATNet with the same augmentation pipeline as SPDNet for a fair comparison.

---

## LOSO SOTA Summary (Literature, English Only)

### Cho2017 (SI-BCI / LOSOCV)

Best reported subject-independent result found (traditional CSP family):
- Method: MRFBCSP (10 filter pairs) + selective subject pooling (alpha=0.005), 50 trials
- Accuracy: 0.6442 (64.42%)
- Protocol: subject-independent (LOSOCV), Table 2
- Source: https://www.mdpi.com/1424-8220/21/16/5436

Project results (this work, Cho2017):
- SPDNet + Augment: **61.23 +/- 10.08%** (κ=0.225) — current best
- vs. literature SOTA 64.42%: gap of ~3.2% (protocol differences apply)

Note:
- Deep learning LOSO results for Cho2017 are not consistently reported in a directly comparable 2-class LOSO setting in the surveyed literature; many works use 4-class or other cross-subject splits.

Recommended model to reproduce (LOSO/SI-BCI):
- MRFBCSP (10) + selective subject pooling (alpha approx 0.005)
- Implemented in `mrfbcsp_loso.py` (without selective pooling; standard LOSO)

### Lee2019 (OpenBMI, LOSO / subject-independent)

Best reported subject-independent result found (deep learning):
- Method: MIN2Net
- Accuracy: 72.03 +/- 14.04 (OpenBMI subject-independent)
- Protocol: subject-independent comparison, Table 3
- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10377689/

Traditional SI-BCI reference (LOSOCV):
- Method: MRFBCSP (10) + selective subject pooling (alpha=0.01), 150 trials
- Accuracy: 0.6936 (69.36%)
- Protocol: subject-independent (LOSOCV), Table 2
- Source: https://www.mdpi.com/1424-8220/21/16/5436

Project results (this work, Lee2019):
- MIN2Net: **69.73 +/- 12.51%** (κ=0.395) — vs. literature 72.03%, gap ~2.3%
- MRFBCSP+LDA: **67.74 +/- 13.14%** (κ=0.355) — vs. literature 69.36%, gap ~1.6%

Recommended model to reproduce (LOSO/SI-BCI):
- MIN2Net
- Implemented in `min2net.py`; run via `train_loso.py --model min2net --dataset lee2019`

Caveats:
- These results are protocol-sensitive. Differences in windowing, channel selection, filtering, and split strategy can materially change accuracy.
- Cho2017 literature is less consistent in 2-class LOSO reporting than OpenBMI; direct SOTA comparison is limited.
