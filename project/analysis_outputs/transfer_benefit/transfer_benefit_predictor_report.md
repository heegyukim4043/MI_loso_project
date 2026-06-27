# Transfer Benefit Predictor / Source Similarity Analysis

## Inputs
- LOSO subject deltas: 1590 subject-comparison rows after duplicate subject aggregation.
- Cross-dataset CSP-LDA deltas: 212 subject-direction-comparison rows.
- Covariance features are computed from streamed, downsampled MOABB NPZ files.
- Cross-dataset distances use only common Cho2017/Lee2019 channels.

## D. Transfer Benefit Predictor

Target: `delta = candidate_acc - baseline_acc`.

Strongest LOSO correlations across all method/backbone/dataset comparisons:

| group | baseline | candidate | dataset | feature | n | rho | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NoEA_to_EA | cspnet_noea | ea_cspnet | cho2017 | source_pool_knn10_dist | 52 | 0.419 | 0.002 |
| NoEA_to_EA | cspnet_noea | ea_cspnet | cho2017 | source_pool_sim_weight | 52 | -0.413 | 0.002 |
| NoEA_to_EA | cspnet_noea | ea_cspnet | cho2017 | source_pool_knn5_dist | 52 | 0.399 | 0.003 |
| NoEA_to_EA | cspnet_noea | ea_cspnet | cho2017 | source_pool_mean_dist | 52 | 0.385 | 0.005 |
| NoEA_to_EA | cspnet_noea | ea_cspnet | cho2017 | source_pool_knn3_dist | 52 | 0.385 | 0.005 |
| NoEA_to_EA | cspnet_noea | ea_cspnet | cho2017 | source_pool_min_dist | 52 | 0.382 | 0.005 |
| EA_to_Snapshot | ea_cspnet | ea_snapshot_adabn_cspnet | cho2017 | baseline_acc | 52 | -0.368 | 0.007 |
| EA_to_Snapshot | ea_cspnet | ea_snapshot_adabn_cspnet | lee2019 | baseline_acc | 54 | -0.353 | 0.009 |
| NoEA_to_EA | cspnet_noea | ea_cspnet | lee2019 | baseline_acc | 54 | -0.347 | 0.010 |
| EA_to_Snapshot | ea_conformer | ea_snapshot_conformer | cho2017 | source_pool_knn10_dist | 52 | 0.334 | 0.015 |
| NoEA_to_EA | cspnet_noea | ea_cspnet | cho2017 | cov_condition_num | 52 | 0.333 | 0.016 |
| EA_to_Snapshot | ea_conformer | ea_snapshot_conformer | cho2017 | source_pool_knn3_dist | 52 | 0.323 | 0.020 |

Strongest LOSO NoEA→EA correlations:

| baseline | candidate | dataset | feature | n | rho | p |
| --- | --- | --- | --- | --- | --- | --- |
| cspnet_noea | ea_cspnet | cho2017 | source_pool_knn10_dist | 52 | 0.419 | 0.002 |
| cspnet_noea | ea_cspnet | cho2017 | source_pool_sim_weight | 52 | -0.413 | 0.002 |
| cspnet_noea | ea_cspnet | cho2017 | source_pool_knn5_dist | 52 | 0.399 | 0.003 |
| cspnet_noea | ea_cspnet | cho2017 | source_pool_mean_dist | 52 | 0.385 | 0.005 |
| cspnet_noea | ea_cspnet | cho2017 | source_pool_knn3_dist | 52 | 0.385 | 0.005 |
| cspnet_noea | ea_cspnet | cho2017 | source_pool_min_dist | 52 | 0.382 | 0.005 |
| cspnet_noea | ea_cspnet | lee2019 | baseline_acc | 54 | -0.347 | 0.010 |
| cspnet_noea | ea_cspnet | cho2017 | cov_condition_num | 52 | 0.333 | 0.016 |
| conformer_noea | ea_conformer | cho2017 | baseline_acc | 52 | -0.321 | 0.020 |
| conformer_noea | ea_conformer | lee2019 | source_pool_knn10_dist | 54 | 0.320 | 0.018 |
| cspnet_noea | ea_cspnet | lee2019 | source_pool_knn3_dist | 54 | 0.318 | 0.019 |
| cspnet_noea | ea_cspnet | lee2019 | source_pool_knn5_dist | 54 | 0.317 | 0.020 |

## E. Source-Subject Similarity vs Benefit

Cross-dataset similarity features measure how close each test subject is to the source dataset subject pool.
Smaller distance or larger `source_dataset_sim_weight` means a more similar source pool.

Strongest cross-dataset correlations:

| comparison | train_dataset | test_dataset | feature | n | rho | p |
| --- | --- | --- | --- | --- | --- | --- |
| CrossCSP_to_SubjectEA | cho2017 | lee2019 | baseline_acc | 54 | -0.688 | <0.001 |
| CrossCSP_to_DatasetEA_SubjectEA | lee2019 | cho2017 | source_dataset_mean_dist | 52 | 0.501 | <0.001 |
| CrossCSP_to_DatasetEA_SubjectEA | lee2019 | cho2017 | source_dataset_min_dist | 52 | 0.501 | <0.001 |
| CrossCSP_to_DatasetEA_SubjectEA | lee2019 | cho2017 | source_dataset_knn3_dist | 52 | 0.501 | <0.001 |
| CrossCSP_to_DatasetEA_SubjectEA | lee2019 | cho2017 | source_dataset_knn5_dist | 52 | 0.501 | <0.001 |
| CrossCSP_to_DatasetEA_SubjectEA | lee2019 | cho2017 | source_dataset_knn10_dist | 52 | 0.501 | <0.001 |
| CrossCSP_to_DatasetEA_SubjectEA | lee2019 | cho2017 | source_dataset_sim_weight | 52 | -0.501 | <0.001 |
| CrossCSP_to_SubjectEA | lee2019 | cho2017 | source_dataset_mean_dist | 52 | 0.315 | 0.023 |
| CrossCSP_to_SubjectEA | lee2019 | cho2017 | source_dataset_min_dist | 52 | 0.315 | 0.023 |
| CrossCSP_to_SubjectEA | lee2019 | cho2017 | source_dataset_knn3_dist | 52 | 0.315 | 0.023 |
| CrossCSP_to_SubjectEA | lee2019 | cho2017 | source_dataset_knn5_dist | 52 | 0.315 | 0.023 |
| CrossCSP_to_SubjectEA | lee2019 | cho2017 | source_dataset_knn10_dist | 52 | 0.315 | 0.023 |

DatasetEA+SubjectEA focused correlations:

| train_dataset | test_dataset | feature | n | rho | p |
| --- | --- | --- | --- | --- | --- |
| lee2019 | cho2017 | source_dataset_mean_dist | 52 | 0.501 | <0.001 |
| lee2019 | cho2017 | source_dataset_min_dist | 52 | 0.501 | <0.001 |
| lee2019 | cho2017 | source_dataset_knn3_dist | 52 | 0.501 | <0.001 |
| lee2019 | cho2017 | source_dataset_knn5_dist | 52 | 0.501 | <0.001 |
| lee2019 | cho2017 | source_dataset_knn10_dist | 52 | 0.501 | <0.001 |
| lee2019 | cho2017 | source_dataset_sim_weight | 52 | -0.501 | <0.001 |
| lee2019 | cho2017 | baseline_acc | 52 | 0.206 | 0.144 |
| cho2017 | lee2019 | baseline_acc | 54 | 0.195 | 0.157 |
| cho2017 | lee2019 | source_dataset_min_dist | 54 | 0.043 | 0.756 |
| cho2017 | lee2019 | source_dataset_mean_dist | 54 | 0.029 | 0.836 |
| cho2017 | lee2019 | source_dataset_knn5_dist | 54 | 0.027 | 0.845 |
| cho2017 | lee2019 | source_dataset_sim_weight | 54 | -0.026 | 0.851 |

## Interpretation Guide

- Positive rho for `baseline_acc` means already-strong subjects gain more from the candidate method; negative rho means weak baseline subjects benefit more.
- Negative rho for distance features means more source-similar subjects gain more.
- Positive rho for `source_*_sim_weight` means having many close source subjects predicts larger transfer benefit.
- Treat this as correlational evidence; it supports subject vulnerability and source-pool suitability analysis, not causal proof.
