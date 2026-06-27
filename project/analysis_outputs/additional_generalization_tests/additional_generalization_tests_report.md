# Additional Generalization Tests

## 1. Bootstrap CI for Cross >= LOSO

Subject pairing: each cross-dataset test subject is compared with the same dataset/subject LOSO best result.

| scope | n | mean_loso_best | mean_cross_best | mean_diff_cross_minus_loso | ci95_low | ci95_high | p_boot_cross_ge_loso | cross_ge_loso_subject_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 106 | 72.542 | 72.069 | -0.474 | -1.244 | 0.287 | 0.112 | 50.000 |
| cho2017->lee2019 | 54 | 73.420 | 74.250 | 0.830 | -0.046 | 1.722 | 0.969 | 63.000 |
| lee2019->cho2017 | 52 | 71.631 | 69.804 | -1.828 | -2.980 | -0.681 | 0.001 | 36.500 |

Interpretation: if the 95% CI for `cross - LOSO` includes 0, cross-dataset best is statistically comparable to LOSO best under this bootstrap. If the CI is entirely below 0, cross remains lower than LOSO.

## 2. McNemar Test: LOSO Best vs Cross-Dataset Best

Binary outcome: accuracy >= 70%.

| scope | n | both_pass | neither_pass | cross_only_pass | loso_only_pass | mcnemar_p | loso_coverage_pct | cross_coverage_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 106 | 53 | 41 | 4 | 8 | 0.388 | 57.500 | 53.800 |
| cho2017->lee2019 | 54 | 31 | 21 | 1 | 1 | 1.000 | 59.300 | 59.300 |
| lee2019->cho2017 | 52 | 22 | 20 | 3 | 7 | 0.344 | 55.800 | 48.100 |

## 3. Riemannian Distance Pre/Post DatasetEA

Common Cho/Lee channels are used. DatasetEA is simulated by whitening each dataset with its own dataset mean covariance.

| metric | n_common_channels | pre_datasetea | post_datasetea | reduction_pct |
| --- | --- | --- | --- | --- |
| dataset_centroid_distance | 48 | 5.480 | 0.009 | 99.837 |
| all_cross_subject_pair_distance_mean | 48 | 11.425 | 10.319 | 9.678 |
| all_cross_subject_pair_distance_median | 48 | 11.081 | 9.895 | 10.708 |

## 4. Source Pool Scaling

| family | source_pool_n | train_dataset | test_dataset | direction | n_subjects | mean_acc | coverage_ge70_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CSP-LDA DatasetEA+SubjectEA all-source | all | cho2017 | lee2019 | cho2017->lee2019 | 54 | 68.102 | 42.593 |
| CSP-LDA baseline all-source | all | cho2017 | lee2019 | cho2017->lee2019 | 54 | 55.148 | 5.556 |
| CSP-LDA source_select | 10 | cho2017 | lee2019 | cho2017->lee2019 | 54 | 68.287 | 38.889 |
| CSP-LDA source_select | 20 | cho2017 | lee2019 | cho2017->lee2019 | 54 | 69.407 | 46.296 |
| CSP-LDA source_select | 30 | cho2017 | lee2019 | cho2017->lee2019 | 54 | 69.574 | 42.593 |
| CSP-LDA source_select | 40 | cho2017 | lee2019 | cho2017->lee2019 | 54 | 69.000 | 40.741 |
| CSPNet DSA+SEA+SessionEA+SourceWeight all-source | all | cho2017 | lee2019 | cho2017->lee2019 | 54 | 73.241 | 59.259 |
| CSP-LDA DatasetEA+SubjectEA all-source | all | lee2019 | cho2017 | lee2019->cho2017 | 52 | 63.948 | 21.154 |
| CSP-LDA baseline all-source | all | lee2019 | cho2017 | lee2019->cho2017 | 52 | 52.756 | 1.923 |
| CSP-LDA source_select | 10 | lee2019 | cho2017 | lee2019->cho2017 | 52 | 64.391 | 30.769 |
| CSP-LDA source_select | 20 | lee2019 | cho2017 | lee2019->cho2017 | 52 | 64.561 | 26.923 |
| CSP-LDA source_select | 30 | lee2019 | cho2017 | lee2019->cho2017 | 52 | 65.138 | 25.000 |
| CSP-LDA source_select | 40 | lee2019 | cho2017 | lee2019->cho2017 | 52 | 64.825 | 26.923 |
| CSPNet DSA+SEA+SessionEA+SourceWeight all-source | all | lee2019 | cho2017 | lee2019->cho2017 | 52 | 69.208 | 42.308 |

## 5. ECE / Reliability

Not computed here. Current aggregate CSVs do not store trial-level predicted probabilities/logits. Re-inference with per-trial confidence output is required.
