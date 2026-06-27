# Five-Method Generalization Metric Comparison

Compared LOSO method families: `No EA`, `EA`, `EA+TENT`, `EA+AdaBN`, `EA+Snapshot`.

## 1. Metric Availability

| metric_family | availability | files | use_in_paper |
| --- | --- | --- | --- |
| Subject robustness / coverage | Directly available | loso_subject_robustness_metrics.csv, coverage_threshold_metrics.csv | Main table: mean, SD, P10, worst, >=70% coverage |
| Transfer benefit metrics | Directly available | loso_paired_subject_deltas.csv | Responder, >=5pp benefit, harm rate, delta distribution |
| Riemannian domain-gap / predictor | Available as predictor, not as full pre/post gap for every method | transfer_benefit_loso_spearman.csv | Covariance geometry predicts which subjects benefit |
| Generalization gap | Partial | coverage_threshold_metrics.csv, crossdataset transfer outputs | LOSO vs cross-dataset gap is available; within-subject/session gap requires matching results |
| Calibration / risk-coverage | Not available from current aggregate CSV | Requires trial-level probability/logit/confidence outputs | Future/optional analysis unless inference probabilities are regenerated |

## 2. Subject Robustness / Coverage

Backbone-averaged LOSO summary across EEGNet, CSPNet, and Conformer.

| scope | dataset | method_family | n_backbones | mean_acc | sd_acc | p10_acc | q1_acc | min_acc | coverage_ge_60_pct | coverage_ge_70_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOSO | all | EA | 3 | 70.52 | 12.34 | 54.83 | 62.33 | 46.83 | 80.2 | 47.13 |
| LOSO | all | EA+AdaBN | 3 | 71.27 | 12.75 | 54.83 | 61.33 | 47.5 | 80.83 | 51.9 |
| LOSO | all | EA+Snapshot | 3 | 71.04 | 12.64 | 55.3 | 61.8 | 46.0 | 79.82 | 50.58 |
| LOSO | all | EA+TENT | 3 | 70.13 | 12.69 | 54.0 | 60.33 | 47.83 | 75.47 | 48.43 |
| LOSO | all | No EA | 3 | 67.39 | 12.61 | 52.67 | 57.83 | 47.17 | 68.57 | 35.2 |

Interpretation: the strongest deployment-oriented readout is `coverage_ge_70_pct`, because it approximates how many subjects pass a practical MI-BCI threshold.

## 3. Transfer Benefit Metrics

`EA` is compared against `NoEA`. `EA+TENT`, `EA+AdaBN`, and `EA+Snapshot` are compared against their corresponding `EA` baseline.

| scope | delta_reference | method_family | dataset | n_backbones | mean_delta | median_delta | p10_delta | responder_rate_pct | large_benefit_ge5pp_pct | harm_rate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOSO | NoEA for EA; EA for TENT/AdaBN/Snapshot | EA | all | 3 | 3.17 | 2.8 | -2.93 | 67.5 | 34.2 | 27.6 |
| LOSO | NoEA for EA; EA for TENT/AdaBN/Snapshot | EA+TENT | all | 3 | -0.4 | -0.09 | -4.67 | 44.2 | 5.9 | 49.5 |
| LOSO | NoEA for EA; EA for TENT/AdaBN/Snapshot | EA+AdaBN | all | 3 | 0.74 | 0.54 | -3.61 | 50.9 | 15.1 | 42.9 |
| LOSO | NoEA for EA; EA for TENT/AdaBN/Snapshot | EA+Snapshot | all | 3 | 0.21 | -0.04 | -4.52 | 47.1 | 11.2 | 48.2 |

Interpretation: EA gives the dominant transfer benefit. TENT/AdaBN/Snapshot are smaller incremental effects on top of EA and should be reported with harm rate, not only mean delta.

## 4. Riemannian Domain-Gap / Benefit Predictor

This table lists the strongest subject-level Spearman predictor per method family and dataset.

| candidate_family | dataset | baseline | candidate | feature | n | rho | p | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EA | cho2017 | cspnet_noea | ea_cspnet | source_pool_knn10_dist | 52 | 0.419 | 0.002 | larger covariance distance predicts larger benefit |
| EA | lee2019 | cspnet_noea | ea_cspnet | baseline_acc | 54 | -0.347 | 0.01 | weaker baseline subjects benefit more |
| EA+TENT | cho2017 | ea_eegnet | ea_tent_eegnet | source_pool_knn3_dist | 52 | 0.18 | 0.2023 | larger covariance distance predicts larger benefit |
| EA+TENT | lee2019 | ea_eegnet | ea_tent_eegnet | baseline_acc | 54 | 0.249 | 0.0694 | feature predicts subject-level benefit |
| EA+AdaBN | cho2017 | ea_conformer | ea_adabn_conformer | source_pool_mean_dist | 52 | 0.308 | 0.0265 | larger covariance distance predicts larger benefit |
| EA+AdaBN | lee2019 | ea_conformer | ea_adabn_conformer | source_pool_knn10_dist | 54 | -0.196 | 0.1558 | feature predicts subject-level benefit |
| EA+Snapshot | cho2017 | ea_cspnet | ea_snapshot_adabn_cspnet | baseline_acc | 52 | -0.368 | 0.0073 | weaker baseline subjects benefit more |
| EA+Snapshot | lee2019 | ea_cspnet | ea_snapshot_adabn_cspnet | baseline_acc | 54 | -0.353 | 0.0089 | weaker baseline subjects benefit more |

Interpretation: covariance geometry is useful as a vulnerability/benefit predictor. For EA, larger source-pool distance often predicts larger benefit, so the safer claim is not 'similar sources always help' but 'source-target covariance geometry predicts alignment benefit'.

## 5. Generalization Gap

Current direct evidence is strongest for LOSO vs cross-dataset degradation/recovery. Within-subject or within-session gaps require matched within-subject/session result files.

| scope | method | direction | n | mean_acc | coverage_ge70_pct |
| --- | --- | --- | --- | --- | --- |
| CrossDataset coverage | CSP-LDA baseline | Cho->Lee | 54 | 55.15 | 5.6 |
| CrossDataset coverage | CSP-LDA baseline | Lee->Cho | 52 | 52.76 | 1.9 |
| CrossDataset coverage | DatasetEA+SubjectEA+CSP-LDA | Cho->Lee | 54 | 68.96 | 40.7 |
| CrossDataset coverage | DatasetEA+SubjectEA+CSP-LDA | Lee->Cho | 52 | 65.1 | 30.8 |
| CrossDataset coverage | SessionEA+CSP-LDA | Cho->Lee | 54 | 70.6 | 42.6 |
| CrossDataset coverage | SessionEA+CSP-LDA | Lee->Cho | 52 | 65.4 | 36.5 |
| CrossDataset coverage | DSA+SEA+SessionEA+SourceWeight+CSPNet | Cho->Lee | 54 | 73.24 | 59.3 |
| CrossDataset coverage | DSA+SEA+SessionEA+SourceWeight+CSPNet | Lee->Cho | 52 | 69.21 | 42.3 |

## 6. Calibration / Risk-Coverage

Not comparable from the current aggregate result files because trial-level confidence, probability, or logits are not stored. To compute ECE, Brier score, reliability diagrams, and risk-coverage curves, rerun inference while saving per-trial predicted probability and correctness.
