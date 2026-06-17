# forgit Model Performance Summary

Generated from `/home/hkim/MI_test/forgit/loso` and `/home/hkim/MI_test/forgit/crossdata`.

## LOSO Summary

| Method | Description | Cho2017 | Lee2019 | Avg | Cho n | Lee n |
|---|---|---|---|---|---|---|
| ea_supcon_adabn_cspnet | EA + SupCon + AdaBN | 71.81 | 72.33 | 72.07 | 52 | 54 |
| ea_snapshot_adabn_cspnet | EA + CSPNet + Snapshot(x6) + AdaBN | 70.63 | 73.50 | 72.06 | 52 | 54 |
| ea_tent_cspnet | EA + CSPNet + TENT | 71.45 | 72.51 | 71.98 | 52 | 54 |
| ea_subjclust_tau5_cspnet | EA + CSPNet + Subject Clustering (τ=5) + AdaBN | 71.18 | 72.50 | 71.84 | 52 | 54 |
| ea_adabn_cspnet | EA + CSPNet + AdaBN (test-time BN update) | 71.04 | 72.22 | 71.63 | 52 | 54 |
| ea_cspnet | EA + CSPNet (best baseline) | 71.43 | 71.69 | 71.56 | 52 | 54 |
| ea_snapshot_adabn_x4_cspnet | ea_snapshot_adabn_x4_cspnet | 70.06 | 72.94 | 71.50 | 52 | 54 |
| ea_snapshot_adabn_x3_cspnet | ea_snapshot_adabn_x3_cspnet | 70.25 | 72.59 | 71.42 | 52 | 54 |
| ea_adabn_conformer | EA + EEGConformer + AdaBN (test-time BN update) | 70.46 | 72.30 | 71.38 | 52 | 54 |
| ea_supcon_coral_cspnet | EA + SupCon + CORAL feature alignment (λ=0.1) | 71.43 | 71.26 | 71.34 | 52 | 54 |
| ea_supcon_cspnet | EA + SupCon (Supervised Contrastive) + CSPNet | 71.23 | 71.46 | 71.34 | 52 | 54 |
| ea_subjclust_tau1_cspnet | EA + CSPNet + Subject Clustering (τ=1) + AdaBN | 70.58 | 71.83 | 71.20 | 52 | 54 |

## Cross-Dataset Summary

| Method | Cho->Lee Acc | Cho->Lee k | Lee->Cho Acc | Lee->Cho k | Avg Acc |
|---|---|---|---|---|---|
| DSA+SEA+SessionEA+SourceWeight+CSPNet (sfreq100) | 73.24 | 0.465 | 69.21 | 0.384 | 71.22 |
| Conformer + DSA+SEA+SessionEA+SourceWeight (sfreq100) | 71.98 | 0.440 | 69.40 | 0.388 | 70.69 |
| DSA+SEA+SourceWeight+CSPNet (sfreq100) | 71.93 | 0.439 | 69.12 | 0.383 | 70.52 |
| EEGNet + DSA+SEA+SessionEA+SourceWeight (sfreq100) | 71.38 | 0.428 | 68.77 | 0.375 | 70.07 |
| Conformer + DSA+SEA+AdaBN (sfreq100) | 71.35 | 0.427 | 68.46 | 0.369 | 69.91 |
| Conformer + DSA+SEA+AdaBN+Con (sfreq100) | 71.35 | 0.427 | 68.46 | 0.369 | 69.91 |
| EEGNet + DSA+SEA+AdaBN (sfreq100) | 70.72 | 0.414 | 68.90 | 0.378 | 69.81 |
| EEGNet + DSA+SEA+AdaBN+Con (sfreq100) | 70.72 | 0.414 | 68.90 | 0.378 | 69.81 |
| Conformer + DSA+SEA (sfreq100) | 71.62 | 0.432 | 67.97 | 0.359 | 69.79 |
| Conformer + DSA+SEA+SourceWeight (sfreq100) | 70.35 | 0.407 | 69.07 | 0.381 | 69.71 |
| EEGNet + DSA+SEA (sfreq100) | 70.81 | 0.416 | 68.56 | 0.371 | 69.68 |
| EEGNet + DSA+SEA+TENT (sfreq100) | 70.17 | 0.403 | 68.48 | 0.370 | 69.32 |
| EEGNet + DSA+SEA+SourceWeight (sfreq100) | 70.02 | 0.400 | 67.75 | 0.355 | 68.88 |
| RawUnified+DatasetEA+SubjectEA+CSP-LDA | 68.96 | 0.379 | 65.10 | 0.302 | 67.03 |
| Conformer + DSA+SEA+TENT (sfreq100) | 70.04 | 0.401 | 62.82 | 0.257 | 66.43 |
| DatasetEA+SubjectEA+CSP-LDA | 68.10 | 0.362 | 63.95 | 0.279 | 66.03 |
| EA+AdaBN+CSPNet (sfreq100) | 66.93 | 0.339 | 64.24 | 0.285 | 65.58 |
| EA+TENT+CSPNet (sfreq100) | 66.59 | 0.332 | 64.13 | 0.283 | 65.36 |
| Conformer + SubjEA+AdaBN (sfreq100) | 65.76 | 0.315 | 64.54 | 0.291 | 65.15 |
| EEGNet + SubjEA+AdaBN (sfreq100) | 63.95 | 0.279 | 64.74 | 0.295 | 64.35 |
| EEGNet + SubjEA+Snapshotx6 (sfreq100) | 62.70 | 0.254 | 63.04 | 0.261 | 62.87 |
| EEGNet + SubjectEA only (sfreq100) | 61.84 | 0.237 | 63.71 | 0.274 | 62.78 |
| EA+Snapshotx6+CSPNet (sfreq100) | 63.81 | 0.276 | 61.36 | 0.227 | 62.58 |
| Conformer + SubjEA+Snapshotx6 (sfreq100) | 60.68 | 0.213 | 63.43 | 0.269 | 62.06 |
| Conformer + SubjectEA only (sfreq100) | 63.49 | 0.270 | 60.45 | 0.209 | 61.97 |

- Best LOSO average: `ea_supcon_adabn_cspnet` avg `72.07%`.
- Best selected cross-dataset: `DSA+SEA+SessionEA+SourceWeight+CSPNet (sfreq100)` avg `71.22%`.
- Full cross subject-level rows: `crossdata/results/aggregated/crossdataset_subject_level_all.csv`.
