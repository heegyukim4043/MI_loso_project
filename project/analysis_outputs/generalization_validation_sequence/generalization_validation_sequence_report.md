# Generalization Validation Sequence

## 1. Separability Tertile별 Method Benefit

| dataset | feature | tertile | n | mean_original | mean_best_adaptive | persistent_rate_pct | recovery_rate_among_original_lt70_pct | EA_delta_original | TENT_delta_original | AdaBN_delta_original | Snapshot_delta_original |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cho2017 | csp_centroid_dist | low | 18 | 58.870 | 63.360 | 77.800 | 17.600 | 3.450 | 2.480 | 3.440 | 2.550 |
| cho2017 | csp_centroid_dist | mid | 16 | 64.150 | 71.860 | 25.000 | 66.700 | 5.530 | 5.650 | 6.800 | 5.570 |
| cho2017 | csp_centroid_dist | high | 18 | 72.490 | 79.410 | 27.800 | 44.400 | 5.820 | 4.150 | 5.700 | 5.430 |
| cho2017 | csp_fisher_ratio | low | 17 | 60.920 | 66.080 | 52.900 | 35.700 | 4.130 | 3.350 | 4.370 | 2.990 |
| cho2017 | csp_fisher_ratio | mid | 17 | 64.010 | 69.540 | 52.900 | 30.800 | 3.440 | 3.050 | 4.240 | 3.620 |
| cho2017 | csp_fisher_ratio | high | 18 | 70.390 | 78.560 | 27.800 | 54.500 | 7.030 | 5.610 | 7.050 | 6.690 |
| lee2019 | csp_centroid_dist | low | 18 | 59.650 | 63.050 | 77.800 | 17.600 | 1.230 | 1.310 | 2.140 | 2.060 |
| lee2019 | csp_centroid_dist | mid | 18 | 68.190 | 72.810 | 33.300 | 40.000 | 2.510 | 2.260 | 3.160 | 3.370 |
| lee2019 | csp_centroid_dist | high | 18 | 80.600 | 84.050 | 11.100 | 33.300 | 0.560 | 0.950 | 2.400 | 1.440 |
| lee2019 | csp_fisher_ratio | low | 18 | 59.280 | 62.610 | 83.300 | 6.200 | 1.230 | 1.360 | 2.190 | 2.080 |
| lee2019 | csp_fisher_ratio | mid | 18 | 69.530 | 73.690 | 33.300 | 40.000 | 2.800 | 2.210 | 2.940 | 2.960 |
| lee2019 | csp_fisher_ratio | high | 18 | 79.640 | 83.600 | 5.600 | 75.000 | 0.270 | 0.950 | 2.560 | 1.820 |

## 2. Recoverable vs Persistent Classifier

Target: among `Original < 70`, predict `recovered = best(EA,TENT,AdaBN,Snapshot) >= 70`.

| evaluation | n | positive_recovered | auroc |
| --- | --- | --- | --- |
| pooled_5fold_cv | 68 | 23 | 0.900 |
| leave_dataset_out_test_cho2017 | 38 | 15 | 0.690 |
| leave_dataset_out_test_lee2019 | 30 | 8 | 0.977 |

Univariate predictor AUROC:

| feature | n | auroc_raw | auroc_oriented |
| --- | --- | --- | --- |
| Original | 68 | 0.854 | 0.854 |
| csp_fisher_ratio | 68 | 0.715 | 0.715 |
| csp_centroid_cosine | 68 | 0.298 | 0.702 |
| csp_centroid_dist | 68 | 0.695 | 0.695 |
| cov_condition_num | 68 | 0.676 | 0.676 |
| source_pool_knn10_dist | 68 | 0.651 | 0.651 |
| source_pool_sim_weight | 68 | 0.414 | 0.586 |
| source_pool_mean_dist | 68 | 0.575 | 0.575 |
| class_cov_riemann_dist | 68 | 0.511 | 0.511 |

## 3. Oracle Gap / Method Selection

| method | mean_acc | coverage_ge70_pct | mean_regret_to_oracle | subjects_where_method_is_best_pct |
| --- | --- | --- | --- | --- |
| EA | 70.520 | 47.200 | 1.910 | 27.400 |
| TENT | 70.130 | 47.200 | 2.300 | 13.200 |
| AdaBN | 71.270 | 50.000 | 1.160 | 34.900 |
| Snapshot | 70.750 | 49.100 | 1.690 | 24.500 |
| OracleBest(EA/TENT/AdaBN/Snapshot) | 72.430 | 56.600 | 0.000 | 100.000 |

Best method counts:

| best_method | n_subjects | pct |
| --- | --- | --- |
| AdaBN | 37 | 34.900 |
| EA | 29 | 27.400 |
| Snapshot | 26 | 24.500 |
| TENT | 14 | 13.200 |

## 4. Harm Analysis

Harm is defined as `delta <= -5%p`. Top feature differences between harm and non-harm groups:

| reference | method | feature | n_harm | n_nonharm | mean_harm | mean_nonharm | cliff_delta_harm_minus_non | p | q_fdr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ea | TENT | csp_fisher_ratio | 3 | 103 | 0.958 | 0.592 | 0.650 | 0.055 | 0.273 |
| ea | TENT | Original | 3 | 103 | 71.500 | 67.265 | 0.220 | 0.523 | 0.702 |
| ea | TENT | csp_centroid_dist | 3 | 103 | 0.979 | 0.983 | 0.359 | 0.310 | 0.702 |
| ea | TENT | cov_condition_num | 3 | 103 | 1.283 | 1.312 | 0.210 | 0.562 | 0.702 |
| ea | TENT | source_pool_mean_dist | 3 | 103 | 0.181 | 0.171 | -0.036 | 0.929 | 0.929 |

## 5. Statistical Tests

Paired accuracy uses Wilcoxon signed-rank. Coverage@70 uses McNemar exact binomial test. FDR is Benjamini-Hochberg over this test family.

| comparison | n | mean_delta | wilcoxon_p | coverage_gain_count | coverage_loss_count | mcnemar_p | wilcoxon_q_fdr | mcnemar_q_fdr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EA vs Original | 106 | 3.137 | 0.000 | 14 | 2 | 0.004 | 0.000 | 0.010 |
| TENT vs Original | 106 | 2.748 | 0.000 | 16 | 4 | 0.012 | 0.000 | 0.021 |
| AdaBN vs Original | 106 | 3.885 | 0.000 | 17 | 2 | 0.001 | 0.000 | 0.005 |
| Snapshot vs Original | 106 | 3.362 | 0.000 | 16 | 2 | 0.001 | 0.000 | 0.005 |
| TENT vs EA | 106 | -0.389 | 0.047 | 5 | 5 | 1.000 | 0.055 | 1.000 |
| AdaBN vs EA | 106 | 0.748 | 0.016 | 5 | 2 | 0.453 | 0.022 | 0.634 |
| Snapshot vs EA | 106 | 0.225 | 0.782 | 8 | 6 | 0.791 | 0.782 | 0.922 |

## Main Takeaways

- Class separability stratifies generalization performance: low-separability subjects remain hardest even after adaptation.
- Recoverability can be predicted above chance from pre-adaptation accuracy, separability, and covariance geometry, but leave-dataset-out performance should be treated as the conservative estimate.
- OracleBest shows the ceiling for subject-adaptive method selection and quantifies how much accuracy is lost by choosing one global method.
- Harm analysis is necessary because methods that improve mean accuracy still create negative transfer for a nontrivial subset.
