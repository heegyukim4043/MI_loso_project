# Preprocessing Methods and Model Parameters

> Last updated: 2026-06-08  
> Project: `/home/hkim/MI_test/`  
> Purpose: Paper writing reference — dataset stats, preprocessing pipelines, model architectures, training settings, results

---

## 1. Datasets

### 1.1 Summary

| Dataset | Subjects | Trials/Subject | Channels | Orig. sfreq | Saved Shape (N, C, T) | sfreq |
|---------|:--------:|:--------------:|:--------:|:-----------:|:---------------------:|:-----:|
| Cho2017 | 52 | ~202 | 64 | 512 Hz | (10520, 64, 257) | 128 Hz |
| Lee2019 | 54 | ~200 | 62 | 1000 Hz | (10800, 62, 201) | 100 Hz |
| PhysioNet | 106 | ~45 | 64 | 160 Hz | (4768, 64, 257) | 128 Hz |
| BCI IV 2a | 9 | 288 | 22 | 250 Hz | (2592, 22, 257) | 128 Hz |

- Binary classification: **left hand (0) vs right hand (1)**
- All saved to `preprocessed/<dataset>.npz` with fields: `X`, `y`, `subjects`, `sfreq`, `ch_names`

---

### 1.2 Cho2017

- **Source**: MOABB `Cho2017`, 52 subjects, 1 session
- **Original**: 64 EEG channels, 512 Hz
- **Script**: `MI_loso_project/preprocess_data.py`

| Step | Detail |
|------|--------|
| Channel selection | `raw.pick("eeg")` — removes EMG, Stim |
| ICA fitting filter | 1–100 Hz bandpass + Common Average Reference (CAR) |
| ICA algorithm | Extended Infomax, `n_components=20`, `random_state=42` |
| Artifact removal | ICLabel ≥ 90% probability: `eye blink`, `muscle artifact` |
| ICA unit | Subject-wise (all sessions concatenated — 1 session only) |
| Resample | 512 Hz → **128 Hz** |
| Bandpass | **8–30 Hz** (mu + beta band) |
| Epoch window | **[0.5, 2.5] s** post-cue (2 s window) |
| Baseline correction | None |
| Final T | **257 samples** @ 128 Hz |

---

### 1.3 Lee2019

- **Source**: MOABB `Lee2019_MI`, 54 subjects, 2 sessions per subject
- **Original**: 62 EEG channels, 1000 Hz
- **Script**: `MI_loso_project/preprocess_data.py`

| Step | Detail |
|------|--------|
| Channel selection | `raw.pick("eeg")` |
| ICA fitting filter | 1–100 Hz bandpass + CAR |
| ICA algorithm | Extended Infomax, `n_components=20`, `random_state=42` |
| Artifact removal | ICLabel ≥ 90%: `eye blink`, `muscle artifact` |
| ICA unit | **Session-wise** (session 1 and session 2 fitted independently) |
| Resample | 1000 Hz → **100 Hz** |
| Bandpass | **8–30 Hz** |
| Epoch window | **[0.5, 2.5] s** post-cue (2 s window) |
| Baseline correction | None |
| Final T | **201 samples** @ 100 Hz |

> **Implementation note**: MOABB 1.5.0 has a session key bug (`"0"` vs integer `1`) that silently drops the first session in `get_data()`. Workaround: `_get_single_subject_data()` + manual `SetRawAnnotations` applied to each run.

---

### 1.4 PhysioNet Motor Imagery

- **Source**: MOABB `PhysionetMI(imagined=True)`, 109 subjects → **3 removed** → **106 subjects**
- **Removed**: subjects {88, 92, 100} (known noisy recordings)
- **Original**: 64 EEG channels, 160 Hz
- **Script**: `MI_loso_project/preprocess_physionet.py`

| Step | Detail |
|------|--------|
| Channel selection | `pick_types(eeg=True, stim=False)` → 64 ch |
| ICA fitting filter | 1–40 Hz bandpass + CAR + projection applied |
| ICA algorithm | Extended Infomax, `n_components=20` |
| Artifact removal | ICLabel ≥ 90%: `eye blink`, `muscle artifact` |
| ICA unit | Subject-wise (all sessions concatenated) |
| Resample | 160 Hz → **128 Hz** |
| Bandpass | **8–30 Hz** |
| Epoch window | **[0.5, 2.5] s** post-cue (2 s window) |
| Event ID | `left_hand=2`, `right_hand=3` |
| Baseline correction | None |
| Final T | **257 samples** @ 128 Hz |

---

### 1.5 BCI Competition IV Dataset 2a

- **Source**: MOABB `BNCI2014_001`, 9 subjects, 2 sessions, originally 4-class
- **Classes used**: left hand (1) and right hand (2) only → **binary**
- **Original**: 22 EEG + 3 EOG channels, 250 Hz
- **Script**: `MI_loso_project/preprocess_bciciv2a.py`

| Step | Detail |
|------|--------|
| Channel selection | `pick_types(eeg=True, eog=False, stim=False)` → **22 EEG channels** |
| ICA | **Not applied** (small dataset; minimal preprocessing policy) |
| Resample | 250 Hz → **128 Hz** |
| Bandpass | **8–30 Hz** (IIR) |
| Epoch window | **[0.0, 2.0] s** post-cue (2 s window) |
| Event ID | `left_hand=1`, `right_hand=2` |
| Session handling | session 1 + session 2 concatenated → 288 trials/subject |
| Baseline correction | None |
| Final T | **257 samples** @ 128 Hz |

> **Note**: 22 channels causes automatic cap: `n_filters = min(32, 22) = 22` in CSPNet.

---

## 2. Preprocessing Pipeline Comparison

```
Dataset       Orig. sfreq  ICA   Bandpass   Epoch window    Channels  → Final (C×T)
────────────────────────────────────────────────────────────────────────────────────
Cho2017        512 Hz       ✓    8–30 Hz    [0.5, 2.5] s    64 ch    →  64 × 257
Lee2019       1000 Hz       ✓    8–30 Hz    [0.5, 2.5] s    62 ch    →  62 × 201
PhysioNet      160 Hz       ✓    8–30 Hz    [0.5, 2.5] s    64 ch    →  64 × 257
BCI IV 2a      250 Hz       ✗    8–30 Hz    [0.0, 2.0] s    22 ch    →  22 × 257
```

**Shared conventions**: binary left/right hand; `0=left, 1=right`; no baseline correction; no trial rejection except `epochs.drop_bad()`.

---

## 3. Euclidean Alignment (EA)

- **Reference**: He et al., "Transfer Learning for BCI: A Euclidean Space Data Alignment Approach," IEEE TBME 2019.
- **Script**: `MI_loso_project/eeg_ea.py`
- Applied **after** preprocessing, **before** the LOSO fold split, **independently per subject**.

### Algorithm

```
For each subject s:
    R_s = (1/N) Σ_i  x_i x_i^T          # mean trial covariance  (C × C)
    R_s = V Λ V^T                         # symmetric eigendecomposition
    R_s^{−1/2} = V Λ^{−1/2} V^T
    x_i^{EA} = R_s^{−1/2} x_i            # whiten each trial
```

| Parameter | Value |
|-----------|-------|
| Covariance formula | `x x^T` (not divided by T — follows paper) |
| Regularization ε | 1e-8 (eigenvalue floor) |
| Labels required | **None** (unsupervised) |
| Applied to test subject | Yes — test subject's own R used (no leakage) |

---

## 4. Model Architectures

### 4.1 CSPNet *(primary model)*

- **Reference**: "CSP-Net: Common Spatial Pattern Empowered Neural Networks for EEG-Based Motor Imagery Classification," KBS 2024.
- **Script**: `MI_loso_project/cspnet.py`
- EEGNet backbone with learnable CSP spatial filter replacing depthwise conv.

#### Architecture

```
Input (B, C, T)
  → unsqueeze → (B, 1, C, T)
  → Block 1:  Conv2d(1→F1, kernel=(1, kernel_length), same-pad) + BN2d
  → Block 2:  CSPLayer(C→n_csp per F1-map) + BN2d + ELU + AvgPool(1,4) + Dropout(p)
  → Block 3:  DepthwiseConv2d(mid→mid, sep_kern) + PointwiseConv2d(mid→F2)
              + BN2d + ELU + AvgPool(1,8) + Dropout(p)
  → Flatten → Linear(n_flat, n_classes)
```

where `mid = F1 × n_csp`.

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| `n_csp` | 8 (capped to channel count if C < 8) |
| `F1` | 8 |
| `F2` | 16 |
| `kernel_length` | `max(16, T // 4)`, forced odd |
| `sep_kern` | `max(8, T // 16)`, forced odd |
| `dropout` | 0.25 |
| `trainable_csp` | True |

#### CSP Initialization (per fold, before training)

Generalized eigenvalue problem: `Σ_0 w = λ Σ_c w`, `Σ_c = Σ_0 + Σ_1`

Select eigenvectors with n_csp/2 smallest + n_csp/2 largest eigenvalues.

Regularization: `Σ_c += ε·I`, `ε = 1e-6 · tr(Σ_c) / C`

#### Trainable Parameter Counts

| Dataset | (C, T) | Params |
|---------|--------|-------:|
| Cho2017 | (64, 257) | **3,578** |
| Lee2019 | (62, 201) | **3,130** |
| PhysioNet | (64, 257) | **3,578** |
| BCI IV 2a | (22, 257) | **3,242** |

---

### 4.2 CSPNetContrastive

- **Script**: `MI_loso_project/cspnet_contrastive.py`
- Shared CSPNet encoder + MI classifier head + supervised contrastive projection head.

#### Architecture

```
Shared encoder (identical to CSPNet body)
  ├── MI head:          Linear(n_flat → 2)           [CE loss]
  └── Projection head:  Linear(n_flat → 128) + ReLU
                        + Linear(128 → 64) + L2-norm  [SupCon loss]
```

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Encoder | CSPNet (n_csp=8, F1=8, F2=16, dropout=0.25) |
| `proj_hidden` | 128 |
| `proj_dim` | 64 |
| SupCon temperature | 0.07 |
| `λ_con` | 0.5 |

#### Two-stage Training

| Stage | Epochs | Loss |
|-------|--------|------|
| Stage 1 | 1–150 | CE only |
| Stage 2 | 151–300 | CE + 0.5 × SupCon |

#### Trainable Parameter Counts

| Dataset | (C, T) | Params |
|---------|--------|-------:|
| Cho2017 | (64, 257) | **28,346** |
| Lee2019 | (62, 201) | **23,802** |
| PhysioNet | (64, 257) | **28,346** |
| BCI IV 2a | (22, 257) | **28,010** |

---

### 4.3 EEGNet *(comparison baseline)*

- **Script**: `MI_loso_project/eegnet.py`

| Parameter | Value |
|-----------|-------|
| `F1` | 8 |
| `D` (depth multiplier) | 2 |
| `F2` | 16 |
| `kernel_length` | `max(16, T // 4)`, forced odd |
| `sep_kern` | `max(8, T // 16)`, forced odd |
| `dropout` | 0.5 |

| Dataset | (C, T) | Params |
|---------|--------|-------:|
| Cho2017 | (64, 257) | **2,410** |
| Lee2019 | (62, 201) | **2,138** |
| PhysioNet | (64, 257) | **2,410** |
| BCI IV 2a | (22, 257) | **1,738** |

---

### 4.4 EEG-Conformer *(comparison baseline)*

- **Script**: `MI_loso_project/conformer.py`

| Parameter | Value |
|-----------|-------|
| `F1` | 40 |
| `D` | 2 |
| `temp_kern` | 25 |
| `pool` | 8 |
| `dropout` | 0.5 |
| `nhead` | 8 |
| `n_layers` | 2 |
| `ff_dim` | 256 |
| `attn_dropout` | 0.3 |

| Dataset | (C, T) | Params |
|---------|--------|-------:|
| Cho2017 | (64, 257) | **141,754** |
| Lee2019 | (62, 201) | **141,594** |
| PhysioNet | (64, 257) | **141,754** |
| BCI IV 2a | (22, 257) | **138,394** |

---

### 4.5 CSP-LDA *(classical baseline)*

- **Script**: `MI_loso_project/loso_csp_lda.py`

| Parameter | Value |
|-----------|-------|
| CSP n_components | 8 |
| CSP criterion | Generalized eigenvalue: `Σ_0 w = λ Σ_c w` |
| Feature | Log-variance of CSP-filtered signals |
| Feature scaling | `StandardScaler` (fit on train, apply to test) |
| Classifier | `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` |
| Labels required | Train labels only (test labels never used) |

---

## 5. Training Configuration

- **Script**: `MI_loso_project/train_loso.py`

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Epochs | 300 |
| Gradient clipping | max_norm = 1.0 |
| Scheduler (default) | CosineAnnealingLR (T_max=300) |
| Random seed | 2026 |
| Input normalization | Per-channel z-score (train mean/std → applied to val/test) |
| cuDNN | Disabled (`torch.backends.cudnn.enabled = False`) |

### LOSO Split

```
For each test_subject s:
    val_subject  = random.choice(remaining)      # 1 subject for validation
    train_pool   = remaining \ {val_subject}     # all others for training
```

| Dataset | N_train | N_val | N_test |
|---------|---------|-------|--------|
| Cho2017 | 50 subj | 1 subj | 1 subj |
| Lee2019 | 52 subj | 1 subj | 1 subj |
| PhysioNet | 104 subj | 1 subj | 1 subj |
| BCI IV 2a | 7 subj | 1 subj | 1 subj |

Early stopping: best val_acc checkpoint saved and restored after training.

---

## 6. Test-Time Adaptation Methods

### 6.1 AdaBN

- **Reference**: Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation," ICLR Workshop 2017.
- **Script**: `MI_loso_project/adabn.py`

| Parameter | Value |
|-----------|-------|
| Updated parameters | `running_mean`, `running_var` of all BN layers |
| Model weights | **Frozen** (no gradient) |
| Momentum | `None` → cumulative moving average |
| Passes over test data | 3 |
| Batch size | 64 |
| Labels required | **None** |
| BN layers in CSPNet | 3 (temporal_conv BN, bn2, bn3) |

### 6.2 TENT

- **Reference**: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization," ICLR 2021.
- **Script**: `MI_loso_project/tent.py`

| Parameter | Value |
|-----------|-------|
| Updated parameters | BN affine: γ (weight), β (bias) |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Gradient steps | 1 (one full pass over test data) |
| Loss | Shannon entropy: −Σ p·log(p + 1e-8) |
| AdaBN warm-start | Yes (3 passes before TENT optimization) |

---

## 7. Snapshot Ensemble

- **Script**: `train_one_fold_model()` in `MI_loso_project/train_loso.py`

| Parameter | x6 (default) | x4 | x3 |
|-----------|:------------:|:--:|:--:|
| Scheduler | CosineAnnealingWarmRestarts | ← same | ← same |
| `T_0` (cycle length) | 50 ep | 75 ep | 100 ep |
| `eta_min` | LR × 1e-2 = 1e-5 | ← same | ← same |
| Snapshots per 300 ep | **6** | **4** | **3** |
| Save condition | `epoch % T_0 == 0` | ← same | ← same |
| Inference | Uniform average of per-snapshot softmax | ← same | ← same |
| +AdaBN | Independent AdaBN (3 passes) per snapshot → average | ← same | ← same |

---

## 8. Subject Clustering Weighted Learning *(novel method)*

- **Script**: `MI_loso_project/train_loso.py` (`--subject_weight` flag)
- Weights training samples by Riemannian similarity between each training subject and the test subject.

| Step | Detail |
|------|--------|
| Covariance estimation | pyriemann `covariances(estimator='oas')` on **pre-EA** EEG |
| Distance metric | Riemannian affine-invariant: `d_R(A, B)` |
| Weight function | `w_s ∝ exp(−d_R(C_s, C_test) / τ)`, softmax-normalized |
| Applied as | Per-sample weighted CE loss |
| Labels required | **None** for covariance/weights (unsupervised) |

| τ | Weight distribution (Cho2017 S1 example) |
|---|------------------------------------------|
| 1.0 | [0.0005, 0.192] — concentrated on nearest subject |
| 5.0 | [0.0105, 0.0351] — smoother, near-uniform |

CLI: `--subject_weight --subject_weight_tau <τ>`

---

## 9. Experimental Results

### 9.1 Cho2017 LOSO (52 subjects, binary L/R)

| Method | Acc (mean ± std) | Cohen's κ |
|--------|:----------------:|:---------:|
| CSP-LDA | 60.66 ± 11.23% | 0.213 |
| EA + CSP-LDA | 66.75 ± 11.79% | 0.335 |
| EA + CSPNet | 71.43 ± 12.66% | 0.429 |
| EA + AdaBN | 71.04 ± 12.58% | 0.420 |
| EA + TENT | 71.45 ± 12.70% | 0.431 |
| EA + AdaBN + Contrastive | **71.81 ± 12.78%** | 0.424 |
| EA + AdaBN + Snapshot (×6) | 70.63 ± 12.14% | 0.421 |

### 9.2 Lee2019 LOSO (54 subjects, binary L/R)

| Method | Acc (mean ± std) | Cohen's κ |
|--------|:----------------:|:---------:|
| CSP-LDA | 62.78 ± 11.01% | 0.256 |
| EA + CSP-LDA | 64.10 ± 11.83% | 0.282 |
| EA + CSPNet | 71.69 ± 12.77% | 0.434 |
| EA + AdaBN | 72.22 ± 13.43% | 0.420 |
| EA + TENT | 72.51 ± 12.71% | 0.426 |
| EA + AdaBN + Contrastive | 72.33 ± 12.90% | 0.418 |
| EA + AdaBN + Snapshot (×6) | **73.50 ± 12.33%** | 0.426 |

### 9.3 PhysioNet LOSO (106 subjects, binary L/R)

| Method | Acc (mean ± std) | Cohen's κ |
|--------|:----------------:|:---------:|
| EA + CSPNet | 68.39 ± 13.78% | 0.367 |
| EA + AdaBN + Contrastive | **69.63 ± 13.81%** | 0.352 |
| EA + AdaBN + Snapshot (×6) | 69.25 ± 14.31% | 0.363 |

### 9.4 BCI IV 2a LOSO (9 subjects, binary L/R)

| Method | Acc (mean ± std) | Cohen's κ |
|--------|:----------------:|:---------:|
| EA + CSPNet | 73.96 ± 10.25% | 0.479 |
| EA + AdaBN + Contrastive | **74.58 ± 11.25%** | 0.462 |

### 9.5 Key Findings

| Finding | Cho2017 | Lee2019 |
|---------|---------|---------|
| EA effect (CSP-LDA → EA+CSPNet) | +10.77%p | +8.91%p |
| EA effect only (CSP-LDA → EA-CSP-LDA) | +6.09%p | +1.32%p |
| DL effect (EA-CSP-LDA → EA+CSPNet) | +4.68%p*** | +7.59%p*** |
| AdaBN vs EA+CSPNet | −0.39%p (ns) | +0.53%p (ns) |
| TENT vs EA+CSPNet | +0.02%p (ns) | +0.82%p (ns) |
| Contrastive vs EA+AdaBN | +0.77%p (ns) | +0.11%p (ns) |
| Snapshot×6 vs EA+CSPNet | −0.80%p (ns) | +1.81%p*** |

Statistical tests: paired Wilcoxon signed-rank, two-sided, subject-level (*** p<0.001, ns p≥0.05).

---

## 10. Software Environment

| Package | Version |
|---------|---------|
| Python | 3.13 |
| PyTorch | (see `environments.md`) |
| MNE | 1.12.0 |
| MOABB | 1.5.0 |
| pyriemann | 0.10 |
| scikit-learn | — |
| mne-icalabel | — |

Hardware: NVIDIA Quadro RTX 6000 (24 GB) × 2 used (GPU 0 excluded — cuBLAS hardware fault).

---

## 11. File Reference

| File | Role |
|------|------|
| `MI_loso_project/preprocess_data.py` | Cho2017, Lee2019 전처리 |
| `MI_loso_project/preprocess_physionet.py` | PhysioNet 전처리 |
| `MI_loso_project/preprocess_bciciv2a.py` | BCI IV 2a 전처리 |
| `MI_loso_project/eeg_ea.py` | Euclidean Alignment |
| `MI_loso_project/cspnet.py` | CSPNet 모델 + CSP 초기화 |
| `MI_loso_project/cspnet_contrastive.py` | CSPNetContrastive + SupCon loss |
| `MI_loso_project/eegnet.py` | EEGNet |
| `MI_loso_project/conformer.py` | EEG-Conformer |
| `MI_loso_project/adabn.py` | AdaBN |
| `MI_loso_project/tent.py` | TENT |
| `MI_loso_project/loso_csp_lda.py` | CSP-LDA / EA-CSP-LDA 베이스라인 |
| `MI_loso_project/train_loso.py` | 전체 LOSO 학습 루프 |
| `MI_loso_project/statistical_tests.py` | Wilcoxon 검정 (A–E groups) |
| `preprocessed/` | 전처리된 `.npz` 파일들 |
| `results/` | LOSO 결과 `.csv` 파일들 |
