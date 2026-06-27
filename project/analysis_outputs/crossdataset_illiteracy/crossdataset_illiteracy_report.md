# Cross-Dataset Illiteracy / Recovery Analysis

Operational threshold: cross-dataset accuracy < 70%.
Baseline: `CSP-LDA baseline`.

## Pooled Summary

| group | n_subjects | mean_baseline | mean_best_combined | mean_best_delta | any_combined_reaches_70 | persistent_fail_all_combined | mean_n_combined_pass70 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline<70 | 102 | 53.16 | 71.13 | 17.96 | 53/102 (52.0%) | 49/102 (48.0%) | 1.16 |
| Baseline>=70 | 4 | 74.62 | 94.88 | 20.25 | 4/4 (100.0%) | 0/4 (0.0%) | 3.0 |

## Direction-Specific Summary

| direction | group | n_subjects | mean_baseline | mean_best_combined | mean_best_delta | any_combined_reaches_70 | persistent_fail_all_combined | mean_n_combined_pass70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cho2017->lee2019 | Baseline<70 | 51 | 54.0 | 72.95 | 18.95 | 29/51 (56.9%) | 22/51 (43.1%) | 1.35 |
| cho2017->lee2019 | Baseline>=70 | 3 | 74.67 | 95.0 | 20.33 | 3/3 (100.0%) | 0/3 (0.0%) | 3.0 |
| lee2019->cho2017 | Baseline<70 | 51 | 52.33 | 69.3 | 16.97 | 24/51 (47.1%) | 27/51 (52.9%) | 0.96 |
| lee2019->cho2017 | Baseline>=70 | 1 | 74.5 | 94.5 | 20.0 | 1/1 (100.0%) | 0/1 (0.0%) | 3.0 |

## Method Effects in Baseline<70 Group

| direction | baseline_group | method | n | mean_acc | mean_delta | pass70_rate_pct | responder_rate_pct | harm_rate_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cho2017->lee2019 | Baseline<70 | DatasetEA+SubjectEA+CSP-LDA | 51 | 66.78 | 12.78 | 39.2 | 92.2 | 5.9 |
| cho2017->lee2019 | Baseline<70 | SessionEA+CSP-LDA | 51 | 69.26 | 15.26 | 39.2 | 94.1 | 5.9 |
| cho2017->lee2019 | Baseline<70 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 51 | 72.01 | 18.01 | 56.9 | 96.1 | 3.9 |
| lee2019->cho2017 | Baseline<70 | DatasetEA+SubjectEA+CSP-LDA | 51 | 63.51 | 11.18 | 19.6 | 92.2 | 7.8 |
| lee2019->cho2017 | Baseline<70 | SessionEA+CSP-LDA | 51 | 64.9 | 12.57 | 35.3 | 88.2 | 9.8 |
| lee2019->cho2017 | Baseline<70 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 51 | 68.71 | 16.38 | 41.2 | 94.1 | 3.9 |

## Persistent Cross-Dataset Failures

| direction | subject | CSP-LDA baseline | DatasetEA+SubjectEA+CSP-LDA | SessionEA+CSP-LDA | DSA+SEA+SessionEA+SourceWeight+CSPNet | best_combined_acc | best_combined_method | best_combined_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lee2019->cho2017 | 32 | 44.5 | 47.5 | 49.0 | 49.0 | 49.0 | SessionEA+CSP-LDA | 4.5 |
| cho2017->lee2019 | 34 | 48.5 | 44.0 | 43.0 | 44.5 | 44.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | -4.0 |
| cho2017->lee2019 | 13 | 48.5 | 62.5 | 62.0 | 59.5 | 62.5 | DatasetEA+SubjectEA+CSP-LDA | 14.0 |
| lee2019->cho2017 | 7 | 48.8 | 55.4 | 53.75 | 61.3 | 61.3 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 12.5 |
| lee2019->cho2017 | 34 | 49.0 | 43.5 | 46.5 | 51.0 | 51.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 2.0 |
| lee2019->cho2017 | 29 | 49.5 | 51.0 | 54.5 | 55.5 | 55.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 6.0 |
| lee2019->cho2017 | 40 | 50.0 | 51.5 | 47.5 | 47.0 | 51.5 | DatasetEA+SubjectEA+CSP-LDA | 1.5 |
| lee2019->cho2017 | 8 | 50.0 | 51.5 | 53.5 | 53.0 | 53.5 | SessionEA+CSP-LDA | 3.5 |
| lee2019->cho2017 | 37 | 50.0 | 54.0 | 51.0 | 50.0 | 54.0 | DatasetEA+SubjectEA+CSP-LDA | 4.0 |
| lee2019->cho2017 | 17 | 50.0 | 52.0 | 53.5 | 55.0 | 55.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 5.0 |
| lee2019->cho2017 | 2 | 50.0 | 56.0 | 53.0 | 55.5 | 56.0 | DatasetEA+SubjectEA+CSP-LDA | 6.0 |
| lee2019->cho2017 | 13 | 50.0 | 48.0 | 49.5 | 57.0 | 57.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 7.0 |
| cho2017->lee2019 | 41 | 50.0 | 60.0 | 60.5 | 60.0 | 60.5 | SessionEA+CSP-LDA | 10.5 |
| cho2017->lee2019 | 24 | 50.0 | 54.5 | 56.0 | 61.5 | 61.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 11.5 |
| cho2017->lee2019 | 25 | 50.0 | 56.0 | 63.5 | 63.0 | 63.5 | SessionEA+CSP-LDA | 13.5 |
| cho2017->lee2019 | 4 | 50.0 | 59.5 | 63.0 | 65.0 | 65.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 15.0 |
| cho2017->lee2019 | 38 | 50.0 | 61.0 | 63.0 | 65.0 | 65.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 15.0 |
| lee2019->cho2017 | 38 | 50.0 | 54.0 | 59.0 | 65.0 | 65.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 15.0 |
| lee2019->cho2017 | 16 | 50.0 | 58.5 | 59.5 | 66.5 | 66.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 16.5 |
| cho2017->lee2019 | 40 | 50.0 | 54.0 | 58.5 | 68.0 | 68.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 18.0 |
| lee2019->cho2017 | 31 | 50.0 | 62.0 | 60.5 | 68.0 | 68.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 18.0 |
| cho2017->lee2019 | 31 | 50.0 | 63.0 | 69.0 | 64.5 | 69.0 | SessionEA+CSP-LDA | 19.0 |
| cho2017->lee2019 | 35 | 50.5 | 48.5 | 51.0 | 53.5 | 53.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 3.0 |
| cho2017->lee2019 | 11 | 50.5 | 51.5 | 51.0 | 54.0 | 54.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 3.5 |
| lee2019->cho2017 | 6 | 50.5 | 55.0 | 58.0 | 57.0 | 58.0 | SessionEA+CSP-LDA | 7.5 |
| cho2017->lee2019 | 50 | 50.5 | 54.0 | 58.5 | 58.0 | 58.5 | SessionEA+CSP-LDA | 8.0 |
| lee2019->cho2017 | 50 | 50.5 | 58.5 | 60.5 | 61.0 | 61.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 10.5 |
| lee2019->cho2017 | 49 | 50.5 | 64.5 | 63.5 | 63.0 | 64.5 | DatasetEA+SubjectEA+CSP-LDA | 14.0 |
| cho2017->lee2019 | 26 | 50.5 | 62.5 | 65.0 | 65.5 | 65.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 15.0 |
| lee2019->cho2017 | 18 | 50.5 | 59.5 | 60.5 | 67.5 | 67.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 17.0 |
| cho2017->lee2019 | 27 | 51.0 | 52.5 | 47.0 | 54.0 | 54.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 3.0 |
| cho2017->lee2019 | 54 | 51.0 | 58.5 | 54.0 | 64.0 | 64.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 13.0 |
| lee2019->cho2017 | 26 | 51.0 | 54.0 | 56.0 | 69.5 | 69.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 18.5 |
| lee2019->cho2017 | 27 | 51.5 | 59.0 | 62.5 | 62.5 | 62.5 | SessionEA+CSP-LDA | 11.0 |
| lee2019->cho2017 | 42 | 51.5 | 64.5 | 67.5 | 69.5 | 69.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 18.0 |
| lee2019->cho2017 | 51 | 52.0 | 59.5 | 61.0 | 63.0 | 63.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 11.0 |
| lee2019->cho2017 | 28 | 52.5 | 48.5 | 45.5 | 51.5 | 51.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | -1.0 |
| cho2017->lee2019 | 16 | 52.5 | 61.0 | 60.0 | 65.5 | 65.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 13.0 |
| cho2017->lee2019 | 15 | 53.0 | 53.5 | 58.5 | 52.0 | 58.5 | SessionEA+CSP-LDA | 5.5 |
| cho2017->lee2019 | 52 | 53.0 | 69.0 | 67.5 | 68.5 | 69.0 | DatasetEA+SubjectEA+CSP-LDA | 16.0 |

## Recovered Cross-Dataset Failures

| direction | subject | CSP-LDA baseline | DatasetEA+SubjectEA+CSP-LDA | SessionEA+CSP-LDA | DSA+SEA+SessionEA+SourceWeight+CSPNet | best_combined_acc | best_combined_method | best_combined_delta | n_combined_pass70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lee2019->cho2017 | 43 | 50.0 | 94.0 | 93.5 | 97.0 | 97.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 47.0 | 3 |
| cho2017->lee2019 | 44 | 50.0 | 84.0 | 90.0 | 93.5 | 93.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 43.5 | 3 |
| cho2017->lee2019 | 37 | 52.5 | 91.0 | 91.5 | 95.5 | 95.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 43.0 | 3 |
| lee2019->cho2017 | 41 | 54.0 | 86.5 | 88.5 | 93.0 | 93.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 39.0 | 3 |
| cho2017->lee2019 | 28 | 50.5 | 79.5 | 83.5 | 88.5 | 88.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 38.0 | 3 |
| lee2019->cho2017 | 23 | 50.0 | 76.5 | 78.0 | 88.0 | 88.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 38.0 | 3 |
| cho2017->lee2019 | 18 | 57.5 | 89.0 | 88.0 | 94.0 | 94.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 36.5 | 3 |
| cho2017->lee2019 | 5 | 54.0 | 79.0 | 85.0 | 90.5 | 90.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 36.5 | 3 |
| lee2019->cho2017 | 48 | 54.0 | 86.0 | 87.0 | 90.0 | 90.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 36.0 | 3 |
| lee2019->cho2017 | 14 | 60.0 | 95.0 | 95.0 | 92.0 | 95.0 | DatasetEA+SubjectEA+CSP-LDA | 35.0 | 3 |
| cho2017->lee2019 | 45 | 50.0 | 70.5 | 82.0 | 85.0 | 85.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 35.0 | 3 |
| cho2017->lee2019 | 33 | 60.5 | 85.5 | 86.0 | 95.5 | 95.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 35.0 | 3 |
| cho2017->lee2019 | 2 | 51.0 | 62.0 | 85.5 | 77.0 | 85.5 | SessionEA+CSP-LDA | 34.5 | 2 |
| lee2019->cho2017 | 10 | 52.0 | 76.5 | 79.5 | 85.5 | 85.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 33.5 | 3 |
| cho2017->lee2019 | 23 | 55.5 | 74.5 | 88.5 | 85.0 | 88.5 | SessionEA+CSP-LDA | 33.0 | 3 |
| cho2017->lee2019 | 9 | 52.0 | 74.5 | 77.0 | 83.0 | 83.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 31.0 | 3 |
| lee2019->cho2017 | 39 | 50.0 | 66.5 | 70.0 | 81.0 | 81.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 31.0 | 2 |
| cho2017->lee2019 | 39 | 59.0 | 84.0 | 85.0 | 88.0 | 88.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 29.0 | 3 |
| cho2017->lee2019 | 32 | 66.0 | 94.0 | 93.0 | 93.5 | 94.0 | DatasetEA+SubjectEA+CSP-LDA | 28.0 | 3 |
| cho2017->lee2019 | 8 | 52.5 | 67.5 | 69.0 | 80.0 | 80.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 27.5 | 1 |
| cho2017->lee2019 | 1 | 53.5 | 66.5 | 67.5 | 81.0 | 81.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 27.5 | 1 |
| cho2017->lee2019 | 46 | 51.0 | 72.0 | 73.0 | 76.5 | 76.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 25.5 | 3 |
| lee2019->cho2017 | 52 | 55.0 | 74.0 | 76.5 | 80.5 | 80.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 25.5 | 3 |
| lee2019->cho2017 | 24 | 51.5 | 69.5 | 73.5 | 76.5 | 76.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 25.0 | 2 |
| lee2019->cho2017 | 25 | 54.5 | 63.5 | 66.0 | 78.5 | 78.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 24.0 | 1 |
| lee2019->cho2017 | 46 | 49.6 | 67.1 | 71.67 | 73.3 | 73.3 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 23.7 | 2 |
| lee2019->cho2017 | 4 | 68.5 | 83.0 | 83.5 | 92.0 | 92.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 23.5 | 3 |
| lee2019->cho2017 | 44 | 50.0 | 72.5 | 70.0 | 73.0 | 73.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 23.0 | 3 |
| lee2019->cho2017 | 21 | 51.0 | 68.5 | 72.0 | 74.0 | 74.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 23.0 | 2 |
| cho2017->lee2019 | 12 | 54.0 | 64.0 | 67.0 | 76.5 | 76.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 22.5 | 1 |
| cho2017->lee2019 | 20 | 51.5 | 59.5 | 67.0 | 74.0 | 74.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 22.5 | 1 |
| lee2019->cho2017 | 15 | 50.0 | 63.0 | 69.0 | 72.5 | 72.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 22.5 | 1 |
| lee2019->cho2017 | 12 | 54.5 | 68.5 | 75.5 | 76.5 | 76.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 22.0 | 2 |
| lee2019->cho2017 | 22 | 50.0 | 67.5 | 72.0 | 66.0 | 72.0 | SessionEA+CSP-LDA | 22.0 | 1 |
| lee2019->cho2017 | 19 | 52.5 | 69.0 | 68.0 | 74.5 | 74.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 22.0 | 1 |
| cho2017->lee2019 | 43 | 51.5 | 73.0 | 71.5 | 71.5 | 73.0 | DatasetEA+SubjectEA+CSP-LDA | 21.5 | 3 |
| cho2017->lee2019 | 6 | 49.5 | 61.0 | 60.5 | 71.0 | 71.0 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 21.5 | 1 |
| lee2019->cho2017 | 20 | 50.5 | 64.0 | 71.0 | 71.5 | 71.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 21.0 | 2 |
| lee2019->cho2017 | 11 | 50.0 | 61.0 | 66.0 | 70.5 | 70.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 20.5 | 1 |
| cho2017->lee2019 | 30 | 51.0 | 62.0 | 67.5 | 71.5 | 71.5 | DSA+SEA+SessionEA+SourceWeight+CSPNet | 20.5 | 1 |

## Interpretation

- Cross-dataset baseline has a much larger below-70 group than LOSO.
- Combined alignment methods recover a large fraction of those below-70 target subjects.
- A persistent cross-dataset failure subgroup remains, especially when all combined methods stay below 70.
