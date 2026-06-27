# BCI Illiteracy vs Generalization Consistency

Operational threshold: accuracy < 70% is treated as BCI-illiteracy / below practical criterion.

## Subject-Backbone Level

Each row is one dataset-subject-backbone case. This is the strictest view because a subject can be illiterate for one backbone but not another.

| group | n_subject_backbone | mean_original | mean_best_adaptive | mean_best_delta | any_method_reaches_70 | persistent_fail_all_methods | mean_n_methods_pass70 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original<70 | 206 | 59.57 | 66.71 | 7.14 | 84/206 (40.8%) | 122/206 (59.2%) | 1.04 |
| Original>=70 | 112 | 81.76 | 85.61 | 3.85 | 110/112 (98.2%) | 2/112 (1.8%) | 3.7 |

## Method Effects Within Original<70 Group

| baseline_group | method | n | mean_acc | mean_delta | pass70_rate_pct | responder_rate_pct | harm_rate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original<70 | EA | 206 | 63.96 | 4.39 | 24.3 | 74.8 | 21.4 |
| Original<70 | TENT | 206 | 63.44 | 3.87 | 25.7 | 70.4 | 25.7 |
| Original<70 | AdaBN | 206 | 64.41 | 4.84 | 27.7 | 75.7 | 23.3 |
| Original<70 | Snapshot | 206 | 64.06 | 4.49 | 26.7 | 73.8 | 23.8 |

## Subject-Mean Level

Each subject is averaged across EEGNet, CSPNet, and Conformer before thresholding.

| group | n_subjects | mean_original | mean_best_adaptive | mean_best_delta | any_method_reaches_70 | persistent_fail_all_methods | mean_n_methods_pass70 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original<70 subject-mean | 68 | 59.85 | 65.44 | 5.59 | 23/68 (33.8%) | 45/68 (66.2%) | 0.93 |
| Original>=70 subject-mean | 38 | 80.86 | 84.95 | 4.08 | 37/38 (97.4%) | 1/38 (2.6%) | 3.74 |

## Persistent Illiteracy Subjects

Subject-mean Original<70 and all EA/TENT/AdaBN/Snapshot means still <70.

| dataset | subject | Original | EA | TENT | AdaBN | Snapshot | best_adaptive_acc | best_adaptive_method | best_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cho2017 | 17 | 48.83 | 54.67 | 53.0 | 54.17 | 53.67 | 54.67 | EA | 5.83 |
| lee2019 | 34 | 49.17 | 49.5 | 50.0 | 50.33 | 49.17 | 50.33 | AdaBN | 1.17 |
| cho2017 | 27 | 49.83 | 56.0 | 54.33 | 55.0 | 57.5 | 57.5 | Snapshot | 7.67 |
| cho2017 | 40 | 50.67 | 54.0 | 51.17 | 52.33 | 53.83 | 54.0 | EA | 3.33 |
| cho2017 | 34 | 51.0 | 55.0 | 54.5 | 52.33 | 52.0 | 55.0 | EA | 4.0 |
| cho2017 | 37 | 51.17 | 51.17 | 52.0 | 50.33 | 49.33 | 52.0 | TENT | 0.83 |
| cho2017 | 28 | 51.67 | 49.33 | 49.33 | 49.67 | 47.5 | 49.67 | AdaBN | -2.0 |
| lee2019 | 35 | 51.83 | 52.83 | 53.33 | 53.0 | 57.0 | 57.0 | Snapshot | 5.17 |
| lee2019 | 48 | 52.0 | 57.33 | 56.33 | 56.5 | 57.5 | 57.5 | Snapshot | 5.5 |
| cho2017 | 13 | 52.33 | 52.0 | 50.83 | 54.83 | 53.67 | 54.83 | AdaBN | 2.5 |
| lee2019 | 50 | 52.5 | 53.33 | 57.0 | 55.17 | 54.67 | 57.0 | TENT | 4.5 |
| lee2019 | 15 | 52.83 | 56.17 | 56.5 | 60.17 | 56.17 | 60.17 | AdaBN | 7.33 |
| lee2019 | 27 | 53.67 | 54.17 | 53.83 | 53.17 | 57.67 | 57.67 | Snapshot | 4.0 |
| cho2017 | 32 | 54.17 | 54.5 | 52.0 | 52.67 | 54.0 | 54.5 | EA | 0.33 |
| lee2019 | 41 | 54.83 | 58.17 | 55.83 | 58.17 | 57.67 | 58.17 | EA | 3.33 |
| cho2017 | 2 | 54.83 | 56.67 | 56.0 | 57.67 | 60.33 | 60.33 | Snapshot | 5.5 |
| lee2019 | 24 | 55.0 | 57.83 | 57.0 | 57.17 | 56.5 | 57.83 | EA | 2.83 |
| cho2017 | 35 | 56.17 | 60.5 | 58.5 | 56.83 | 60.17 | 60.5 | EA | 4.33 |
| lee2019 | 13 | 56.17 | 62.83 | 61.83 | 63.17 | 59.17 | 63.17 | AdaBN | 7.0 |
| lee2019 | 11 | 56.5 | 53.17 | 55.5 | 53.17 | 56.67 | 56.67 | Snapshot | 0.17 |
| cho2017 | 20 | 56.5 | 66.5 | 64.83 | 68.67 | 67.83 | 68.67 | AdaBN | 12.17 |
| cho2017 | 8 | 57.33 | 59.17 | 55.5 | 57.5 | 56.83 | 59.17 | EA | 1.83 |
| lee2019 | 54 | 57.33 | 63.0 | 62.33 | 64.33 | 61.17 | 64.33 | AdaBN | 7.0 |
| lee2019 | 4 | 59.0 | 57.5 | 55.83 | 59.5 | 56.83 | 59.5 | AdaBN | 0.5 |
| cho2017 | 6 | 59.0 | 66.67 | 62.33 | 62.5 | 62.83 | 66.67 | EA | 7.67 |
| cho2017 | 16 | 59.83 | 61.83 | 63.5 | 63.83 | 61.83 | 63.83 | AdaBN | 4.0 |
| cho2017 | 38 | 59.83 | 67.83 | 65.17 | 69.33 | 62.0 | 69.33 | AdaBN | 9.5 |
| lee2019 | 25 | 60.0 | 60.17 | 61.17 | 59.17 | 60.5 | 61.17 | TENT | 1.17 |
| lee2019 | 51 | 60.0 | 64.33 | 64.33 | 64.5 | 64.5 | 64.5 | AdaBN | 4.5 |
| cho2017 | 50 | 60.17 | 66.67 | 65.17 | 69.0 | 65.17 | 69.0 | AdaBN | 8.83 |

## Strongly Recovered Illiteracy Subjects

Subject-mean Original<70 but at least one generalization method reaches >=70.

| dataset | subject | Original | EA | TENT | AdaBN | Snapshot | best_adaptive_acc | best_adaptive_method | best_delta | n_methods_pass70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cho2017 | 39 | 51.33 | 75.17 | 70.83 | 76.33 | 75.33 | 76.33 | AdaBN | 25.0 | 4 |
| cho2017 | 15 | 64.83 | 79.5 | 79.67 | 82.17 | 78.17 | 82.17 | AdaBN | 17.33 | 4 |
| cho2017 | 12 | 57.67 | 70.83 | 72.0 | 72.33 | 73.17 | 73.17 | Snapshot | 15.5 | 4 |
| cho2017 | 21 | 67.83 | 78.67 | 78.67 | 81.67 | 83.17 | 83.17 | Snapshot | 15.33 | 4 |
| lee2019 | 47 | 63.5 | 74.33 | 75.33 | 75.0 | 75.67 | 75.67 | Snapshot | 12.17 | 4 |
| lee2019 | 22 | 65.33 | 67.67 | 75.33 | 76.5 | 72.5 | 76.5 | AdaBN | 11.17 | 3 |
| cho2017 | 19 | 60.0 | 71.0 | 69.17 | 70.5 | 70.17 | 71.0 | EA | 11.0 | 3 |
| cho2017 | 5 | 60.0 | 70.5 | 64.5 | 68.17 | 69.67 | 70.5 | EA | 10.5 | 1 |
| cho2017 | 11 | 61.83 | 72.33 | 68.83 | 69.5 | 69.5 | 72.33 | EA | 10.5 | 1 |
| cho2017 | 52 | 66.17 | 76.5 | 74.0 | 75.83 | 73.5 | 76.5 | EA | 10.33 | 4 |
| cho2017 | 31 | 62.5 | 66.5 | 71.33 | 70.33 | 71.67 | 71.67 | Snapshot | 9.17 | 3 |
| lee2019 | 49 | 66.33 | 67.67 | 72.0 | 74.67 | 68.83 | 74.67 | AdaBN | 8.33 | 2 |
| cho2017 | 18 | 64.67 | 70.17 | 71.33 | 72.67 | 70.0 | 72.67 | AdaBN | 8.0 | 4 |
| cho2017 | 33 | 62.83 | 67.83 | 70.33 | 66.33 | 65.0 | 70.33 | TENT | 7.5 | 1 |
| cho2017 | 47 | 67.5 | 72.83 | 73.5 | 74.67 | 72.0 | 74.67 | AdaBN | 7.17 | 4 |
| lee2019 | 17 | 63.83 | 67.33 | 65.5 | 69.0 | 71.0 | 71.0 | Snapshot | 7.17 | 1 |
| cho2017 | 49 | 64.5 | 68.33 | 68.0 | 71.5 | 70.5 | 71.5 | AdaBN | 7.0 | 2 |
| lee2019 | 52 | 65.17 | 63.83 | 65.83 | 69.5 | 72.17 | 72.17 | Snapshot | 7.0 | 1 |
| cho2017 | 26 | 68.83 | 70.0 | 70.33 | 74.67 | 69.33 | 74.67 | AdaBN | 5.83 | 3 |
| lee2019 | 42 | 67.83 | 70.17 | 72.0 | 70.67 | 68.33 | 72.0 | TENT | 4.17 | 3 |
| lee2019 | 30 | 68.67 | 65.67 | 66.83 | 71.33 | 70.33 | 71.33 | AdaBN | 2.67 | 2 |
| cho2017 | 36 | 68.83 | 71.17 | 70.5 | 70.5 | 67.17 | 71.17 | EA | 2.33 | 3 |
| lee2019 | 12 | 69.0 | 69.5 | 70.0 | 68.0 | 70.0 | 70.0 | TENT | 1.0 | 2 |

## Interpretation

- There are persistent low-performing subjects, so BCI-illiteracy is not fully solved by current generalization methods.
- But a substantial fraction of Original<70 cases crosses 70% after at least one method, meaning many apparent illiteracy cases are method/alignment dependent.
- Report both persistent-fail rate and recovered rate; using only Original accuracy overstates fixed subject inability.
