# Class Feature Similarity Analysis

Features are computed from streamed MOABB NPZ trials.

Metrics:
- `class_cov_riemann_dist`: Riemannian distance between class mean covariance matrices. Larger means class covariance patterns are less similar.
- `csp_centroid_dist`: Euclidean distance between left/right class centroids in subject-level CSP log-variance feature space. Larger means better separation.
- `csp_centroid_cosine`: cosine similarity between class centroids. Larger means more similar.
- `csp_fisher_ratio`: between-class centroid distance normalized by within-class variance. Larger means better separability.

## Dataset Summary

| dataset | n_subjects | mean_cov_dist | mean_csp_centroid_dist | mean_csp_cosine | mean_csp_fisher |
| --- | --- | --- | --- | --- | --- |
| cho2017 | 52 | 1.715 | 0.875 | 0.980 | 0.646 |
| lee2019 | 54 | 2.096 | 1.087 | 0.970 | 0.559 |

## Illiteracy/Persistent Failure Groups

| dataset | original_illiterate | persistent_fail | n | mean_original_acc | mean_best_generalization_acc | mean_cov_dist | mean_csp_centroid_dist | mean_csp_fisher |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cho2017 | False | False | 14 | 80.185 | 85.554 | 1.887 | 1.120 | 0.919 |
| cho2017 | True | False | 15 | 63.289 | 74.122 | 1.685 | 0.825 | 0.625 |
| cho2017 | True | True | 23 | 57.344 | 61.305 | 1.630 | 0.757 | 0.495 |
| lee2019 | False | False | 23 | 81.659 | 85.362 | 2.218 | 1.302 | 0.862 |
| lee2019 | False | True | 1 | 72.000 | 66.833 | 8.922 | 5.593 | 2.006 |
| lee2019 | True | False | 8 | 66.208 | 72.917 | 1.594 | 0.830 | 0.483 |
| lee2019 | True | True | 22 | 57.826 | 61.129 | 1.841 | 0.752 | 0.205 |

## Strongest Spearman Correlations

| dataset | feature | target | n | rho | p |
| --- | --- | --- | --- | --- | --- |
| lee2019 | csp_centroid_dist | best_generalization_acc | 54 | 0.712 | 0.000 |
| lee2019 | csp_centroid_cosine | best_generalization_acc | 54 | -0.706 | 0.000 |
| lee2019 | csp_centroid_dist | original_acc | 54 | 0.706 | 0.000 |
| lee2019 | csp_centroid_cosine | original_acc | 54 | -0.703 | 0.000 |
| lee2019 | csp_fisher_ratio | best_generalization_acc | 54 | 0.683 | 0.000 |
| lee2019 | csp_fisher_ratio | original_acc | 54 | 0.674 | 0.000 |
| cho2017 | csp_centroid_dist | best_generalization_acc | 52 | 0.621 | 0.000 |
| cho2017 | csp_centroid_cosine | best_generalization_acc | 52 | -0.610 | 0.000 |
| cho2017 | csp_centroid_dist | original_acc | 52 | 0.553 | 0.000 |
| cho2017 | csp_centroid_cosine | original_acc | 52 | -0.539 | 0.000 |
| cho2017 | csp_fisher_ratio | best_generalization_acc | 52 | 0.426 | 0.002 |
| lee2019 | class_cov_riemann_dist | original_acc | 54 | 0.410 | 0.002 |

## Interpretation Guide

- If class separability metrics correlate positively with `original_acc`, low-performing subjects likely have intrinsically less separable MI features.
- If separability is low in persistent failures, that supports a persistent/physiological component beyond alignment.
- If separability is reasonable but accuracy improves mainly after alignment, that supports recoverable processing/alignment failure.
