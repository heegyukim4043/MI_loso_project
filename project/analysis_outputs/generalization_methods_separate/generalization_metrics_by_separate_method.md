# Generalization Metrics by Separate Method

Methods are treated as final methods: `Original`, `EA`, `TENT`, `AdaBN`, `Snapshot`.
`TENT`, `AdaBN`, and `Snapshot` are not merged into EA in the tables below.

## Subject Robustness / Coverage

Backbone-averaged LOSO summary across EEGNet, CSPNet, and Conformer.

| method | n_backbones | mean_acc | sd_acc | median_acc | p10_acc | q1_acc | min_acc | coverage_ge_60_pct | coverage_ge_70_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original | 3 | 67.39 | 12.61 | 64.83 | 52.67 | 57.83 | 47.17 | 68.57 | 35.2 |
| EA | 3 | 70.52 | 12.34 | 68.83 | 54.83 | 62.33 | 46.83 | 80.2 | 47.13 |
| TENT | 3 | 70.13 | 12.69 | 68.17 | 54.0 | 60.33 | 47.83 | 75.47 | 48.43 |
| AdaBN | 3 | 71.27 | 12.75 | 70.17 | 54.83 | 61.33 | 47.5 | 80.83 | 51.9 |
| Snapshot | 3 | 70.75 | 12.64 | 69.5 | 54.83 | 61.0 | 47.0 | 78.63 | 50.33 |

## Transfer Benefit vs Original

Every method is compared directly against its own Original/NoEA backbone baseline for the same dataset and subject.

| reference | method | n_subject_backbone | mean_delta | median_delta | p10_delta | min_delta | max_delta | responder_rate_pct | large_benefit_ge5pp_pct | harm_rate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original | EA | 318 | 3.14 | 2.5 | -3.0 | -11.0 | 33.0 | 67.3 | 34.0 | 27.7 |
| Original | TENT | 318 | 2.75 | 2.5 | -4.5 | -16.0 | 32.5 | 65.7 | 30.8 | 28.9 |
| Original | AdaBN | 318 | 3.89 | 3.0 | -2.5 | -15.0 | 32.0 | 73.0 | 36.2 | 26.1 |
| Original | Snapshot | 318 | 3.36 | 2.5 | -3.5 | -22.5 | 29.5 | 68.6 | 33.0 | 27.7 |

## Incremental Benefit vs EA

Only TENT, AdaBN, and Snapshot are shown here. This asks whether the method improves beyond EA.

| reference | method | n_subject_backbone | mean_delta | median_delta | p10_delta | min_delta | max_delta | responder_rate_pct | large_benefit_ge5pp_pct | harm_rate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EA | TENT | 318 | -0.39 | 0.0 | -4.5 | -16.0 | 10.5 | 44.3 | 6.0 | 49.4 |
| EA | AdaBN | 318 | 0.75 | 0.5 | -3.5 | -11.5 | 14.0 | 50.9 | 15.1 | 42.8 |
| EA | Snapshot | 318 | 0.22 | 0.0 | -4.65 | -12.0 | 16.5 | 47.2 | 11.3 | 48.1 |

## Riemannian Benefit Predictor

Strongest subject-level Spearman predictor per method and dataset.

| method | dataset | baseline | candidate | feature | n | rho | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EA | cho2017 | cspnet_noea | ea_cspnet | source_pool_knn10_dist | 52 | 0.419 | 0.002 |
| EA | lee2019 | cspnet_noea | ea_cspnet | baseline_acc | 54 | -0.347 | 0.01 |
| TENT | cho2017 | ea_eegnet | ea_tent_eegnet | source_pool_knn3_dist | 52 | 0.18 | 0.2023 |
| TENT | lee2019 | ea_eegnet | ea_tent_eegnet | baseline_acc | 54 | 0.249 | 0.0694 |
| AdaBN | cho2017 | ea_conformer | ea_adabn_conformer | source_pool_mean_dist | 52 | 0.308 | 0.0265 |
| AdaBN | lee2019 | ea_conformer | ea_adabn_conformer | source_pool_knn10_dist | 54 | -0.196 | 0.1558 |
| Snapshot | cho2017 | ea_cspnet | ea_snapshot_adabn_cspnet | baseline_acc | 52 | -0.368 | 0.0073 |
| Snapshot | lee2019 | ea_cspnet | ea_snapshot_adabn_cspnet | baseline_acc | 54 | -0.353 | 0.0089 |

## Notes

- `Original` has no transfer-benefit delta because it is the reference baseline.
- Calibration and risk-coverage still require trial-level probability/logit outputs; current aggregate files cannot compute them.
- Generalization gap against cross-dataset remains a separate table because the available cross-dataset methods do not map one-to-one to all five LOSO methods.
