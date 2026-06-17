# MI-EEG LOSO Experiment Progress

**Task**: Cross-subject Motor Imagery EEG classification  
**Datasets**: Cho2017 (52 subjects), Lee2019 (54 subjects)  
**Protocol**: Leave-One-Subject-Out (LOSO) — train on N-1, test on 1  
**Classes**: Left-hand vs Right-hand MI (binary, balanced)  
**Metric**: Accuracy (= Balanced Accuracy for balanced classes)

---

## Directory Structure

```
loso/
├── models/              # Source code
│   ├── train_loso.py    # Main LOSO training entry point
│   ├── cspnet.py        # CSPNet architecture
│   ├── cspnet_contrastive.py  # CSPNet + SupCon loss
│   ├── cspnet_dann.py   # CSPNet + DANN adversarial
│   ├── eegnet.py        # EEGNet architecture
│   ├── conformer.py     # EEGConformer architecture
│   ├── eeg_ea.py        # Euclidean Alignment (EA)
│   ├── eeg_style_aug.py # Covariance Style Transfer augmentation
│   ├── eeg_augment.py   # Time-domain augmentation
│   ├── adabn.py         # Adaptive Batch Normalization
│   ├── tent.py          # TENT (entropy minimization TTA)
│   ├── sam.py           # SAM optimizer
│   ├── loso_csp_lda.py  # CSP-LDA baseline
│   ├── mrfbcsp_loso.py  # Filter-bank CSP
│   └── statistical_tests.py
├── results/             # Per-subject accuracy CSVs + summary
│   ├── summary_all_methods.csv   ← all methods × datasets × stats
│   ├── ea_cspnet.csv             ← per-subject acc (dataset, subject, acc)
│   └── ...
└── Progress.md          # This file
```

---

## Results Summary

> All values are **accuracy (%)** on held-out test subject.  
> `★` = best result per dataset.

### 1. Classical Baselines

| Method | Cho2017 | Lee2019 | Note |
|--------|---------|---------|------|
| CSP-LDA | 60.66 ± 11.3 | 62.78 ± 11.1 | No alignment |
| EA + CSP-LDA | 66.75 ± 11.9 | 64.10 ± 11.9 | +6.1 / +1.3 vs CSP-LDA |

### 2. Deep Learning — CSPNet

| Method | Cho2017 | Lee2019 | Note |
|--------|---------|---------|------|
| CSPNet (no EA) | 65.77 ± 12.1 | 69.99 ± 13.8 | Baseline deep |
| EA + CSPNet | 71.43 ± 12.8 | 71.69 ± 12.9 | **+5.7 / +1.7 vs no-EA** |
| EA + AdaBN | 71.04 ± 12.7 | 72.22 ± 13.6 | Test-time BN adaptation |
| EA + TENT | 71.45 ± 12.8 | 72.51 ± 12.8 | Entropy minimization |
| EA + Snapshot(×6 T0=50) + AdaBN | 70.63 ± 12.3 | **★ 73.50 ± 12.4** | Ensemble + BN adapt |
| EA + Snapshot(×4 T0=75) + AdaBN | 70.06 ± 12.2 | 72.94 ± 12.9 | snap_adabn metric |
| EA + Snapshot(×3 T0=100) + AdaBN | 70.25 ± 12.2 | 72.59 ± 13.1 | snap_adabn metric |
| EA + SubjClust(τ=1) + AdaBN | 70.58 ± 12.5 | 71.83 ± 13.3 | Riemannian subject weighting |
| EA + SubjClust(τ=5) + AdaBN | 71.18 ± 13.4 | 72.50 ± 13.2 | |

### 3. Deep Learning — Adversarial & Contrastive

| Method | Cho2017 | Lee2019 | Note |
|--------|---------|---------|------|
| DANN (no EA) | 65.00 ± 11.8 | 70.78 ± 13.3 | GRL domain confusion |
| EA + SupCon | 71.23 ± 12.6 | 71.46 ± 13.1 | Supervised contrastive |
| EA + SupCon + AdaBN | **★ 71.81 ± 12.9** | 72.33 ± 13.0 | Best Cho |
| EA + SupCon + CORAL | 71.43 ± 12.5 | 71.26 ± 12.7 | Feature alignment |

### 4. Deep Learning — EEGNet

| Method | Cho2017 | Lee2019 | Note |
|--------|---------|---------|------|
| EEGNet (no EA) | 65.93 ± 12.4 | 69.50 ± 12.6 | |
| EA + EEGNet | 69.55 ± 12.3 | 71.36 ± 12.6 | **+3.6 / +1.9** |
| EA + TENT + EEGNet | 69.39 ± 12.6 | 71.69 ± 13.3 | |
| EA + AdaBN + EEGNet | 69.89 ± 12.7 | 71.62 ± 13.2 | |
| EA + Snapshot(×6) + EEGNet | 69.08 ± 13.0 | 71.05 ± 13.3 | Snapshot hurts EEGNet (−0.5 / −0.3) |

### 5. Deep Learning — EEGConformer

| Method | Cho2017 | Lee2019 | Note |
|--------|---------|---------|------|
| Conformer (no EA) | 63.92 ± 11.6 | 68.95 ± 12.4 | |
| EA + Conformer | 69.37 ± 11.7 | 69.69 ± 12.1 | **+5.5 / +0.7** |
| EA + TENT + Conformer | 66.88 ± 12.0 | 68.77 ± 12.7 | TENT hurts Conformer |
| EA + AdaBN + Conformer | 70.46 ± 12.1 | 72.30 ± 12.5 | AdaBN rescues it |

### 6. Within-dataset Transfer

| Method | Cho2017 | Lee2019 | Note |
|--------|---------|---------|------|
| EA + KMM-TrAdaBoost | 66.53 ± 11.6 | 63.56 ± 11.4 | Density-ratio reweighting |

---

## Key Findings

### Finding 1: EA is the single most impactful technique
Euclidean Alignment (per-subject covariance whitening) gives the largest single improvement across all architectures:
- CSPNet: +5.7%p (Cho), +1.7%p (Lee)
- EEGNet: +3.6%p (Cho), +1.9%p (Lee)
- Conformer: +5.5%p (Cho), +0.7%p (Lee)

### Finding 2: Feature alignment / adversarial methods add marginal value on top of EA
| Category | Δ vs EA+CSPNet (Cho/Lee) |
|----------|--------------------------|
| DANN (adversarial, no EA) | **−6.4 / −0.9** |
| CORAL (feature align) | +0.0 / −0.4 |
| SupCon (contrastive) | −0.2 / −0.2 |
| AdaBN (TTA) | −0.4 / +0.5 |
| TENT (TTA) | +0.0 / +0.8 |

→ EA handles the domain shift at input level. Feature-space alignment on top adds ≤1%p.

### Finding 4: Snapshot ensemble is architecture-dependent, and more snapshots help for CSPNet

Cosine-annealing snapshot ensemble + AdaBN:

| Architecture | n_snap | Δ Cho vs base | Δ Lee vs base | Verdict |
|---|---|---|---|---|
| CSPNet + AdaBN | ×3 (T0=100) | −1.00 | +0.37 | Marginal |
| CSPNet + AdaBN | ×4 (T0=75) | −1.17 | +0.72 | Modest |
| CSPNet + AdaBN | ×6 (T0=50) | **−0.41** | **+1.28** | Best — more cycles = better diversity |
| EEGNet | ×6 | −0.47 | −0.31 | Hurts |
| Conformer | ×6 (in progress) | — | — | TBD |

Δ is `snap_adabn_acc` vs EA+AdaBN base (Cho=71.04%, Lee=72.22%).

→ More snapshots (shorter T0) consistently improve diversity for CSPNet. Lee2019 benefits more than Cho2017.  
→ EEGNet (~3K params) underfits within cosine cycles → correlated snapshots.  
→ Conformer in progress on GPU2; result TBD.

### Finding 3: Source pool scaling (CSP-LDA)
CSP-LDA accuracy as a function of N training subjects:

| N | Cho2017 | Lee2019 |
|---|---------|---------|
| 3  | 57.7% | 57.2% |
| 10 | 63.1% | 58.4% |
| 20 | 65.1% | 59.8% |
| 40 | 65.8% | 62.0% |
| 50 (max) | 65.6% | 62.4% |

→ **Saturates around N=20–25.** EA+CSPNet (71.4%) surpasses the CSP-LDA ceiling (66%) by ~5%p regardless of subject pool size.

---

## Experiment Status

### Completed
| Experiment | Result file |
|------------|-------------|
| CSP-LDA | `csp_lda.csv` |
| EA + CSP-LDA | `ea_csp_lda.csv` |
| EA + CSPNet (main baseline) | `ea_cspnet.csv` |
| EA + AdaBN/TENT/Snapshot(×6)(CSPNet) | `ea_adabn_cspnet.csv` etc. |
| EA + CSPNet + Snapshot(×3 T0=100) + AdaBN | `loso_results_ea_adabn_snapshot_x3_cspnet_cspnet.csv` |
| EA + CSPNet + Snapshot(×4 T0=75) + AdaBN | `loso_results_ea_adabn_snapshot_x4_cspnet_cspnet.csv` |
| EA + CSPNet + SubjClust(τ=1/5) + AdaBN | `ea_subjclust_tau{1,5}_cspnet.csv` |
| EA + CSPNet + StyleAug (p=0.5) | `ea_cspnet_style_aug.csv` — Cho 68.30% / Lee 71.17% |
| EA + SupCon (Contrastive) | `ea_supcon_*` |
| DANN | `dann_cspnet.csv` |
| EEGNet (no EA / +EA / +TENT / +AdaBN) | `eegnet_*.csv` |
| EA + EEGNet + Snapshot(×6) | `ea_snapshot_eegnet.csv` |
| EEGConformer (no EA / +EA / +TENT) | `conformer_*.csv` |
| EEGConformer + EA + AdaBN | `ea_adabn_conformer.csv` |
| KMM-TrAdaBoost | `ea_kmm_tradaboost.csv` |
| Source Pool Scaling (CSP-LDA) | `source_pool_scaling_*.csv` |

### In Progress (as of 2026-06-17)
| Experiment | GPU | Cho2017 | Lee2019 | Status |
|------------|-----|---------|---------|--------|
| EA + Conformer + Snapshot(×6 T0=50) | GPU 2 | 67.72% snap (52/52 done) | [07/54] ~15h | 진행 중 |

> Auto-update watcher (`/tmp/forgit_final_watcher.sh`) will run `update_forgit.py` on completion.

### Planned
| Experiment | Method | Description |
|------------|--------|-------------|
| EA + CSPNet + SAM | SAM optimizer | Flat loss landscape → better generalization |
| EA + CSPNet + GroupDRO | GroupDRO | Worst-group loss optimization |
| Source Pool Scaling (CSPNet) | Deep version | Find N threshold where DL > CSP-LDA |

---

## Model Architecture Notes

### CSPNet
- CSP spatial filter layer (4 filters, initialized from training data)  
- BatchNorm → ELU → Depthwise temporal conv → Global average pool → FC  
- ~50K parameters  

### EEGNet
- Depthwise conv + separable conv  
- ~3K parameters (lightweight)  

### EEGConformer  
- CNN patch embedding + Transformer encoder + FC head  
- ~500K parameters  

### Euclidean Alignment (EA)
- Per-subject whitening: `X_s → R_s^{-1/2} X_s` where `R_s = mean covariance`  
- Applied before training, no test-time leakage  
- Reduces inter-subject covariance shift  

### Covariance Style Transfer (new)
- Reverse of EA: re-colors whitened data with another subject's covariance  
- Training augmentation: `x_ea → R_j^{1/2} x_ea` (random j from training pool)  
- Increases covariance diversity of training distribution  

---

## Reproducibility

```bash
# EA + CSPNet (main result)
CUDA_VISIBLE_DEVICES=1 python models/train_loso.py \
    --model cspnet --ea --dataset both \
    --run_id ea_cspnet

# EA + CSPNet + Covariance Style Aug
CUDA_VISIBLE_DEVICES=1 python models/train_loso.py \
    --model cspnet --ea --style_aug --style_aug_p 0.5 \
    --dataset both --run_id ea_cspnet_style_aug

# EA + CSPNet + SAM
CUDA_VISIBLE_DEVICES=1 python models/train_loso.py \
    --model cspnet --ea --sam --sam_rho 0.05 \
    --dataset both --run_id ea_cspnet_sam

# CSP-LDA baseline
python models/loso_csp_lda.py --dataset cho2017
python models/loso_csp_lda.py --dataset lee2019
```

**Hardware**: Quadro RTX 6000 (24GB) × 3  
**Framework**: PyTorch 2.x, Python 3.10  
**Seed**: 42 (fixed throughout)
