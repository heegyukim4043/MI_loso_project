# Combined Methods: BCI Illiteracy vs Generalization Consistency

Operational threshold: accuracy < 70%.
Subject-level analysis is CSPNet-only because the available combination methods are CSPNet variants.

## Combined Method Set

- EA+Snapshot+AdaBN
- EA+SupCon
- EA+SupCon+AdaBN
- EA+SupCon+CORAL

## Summary

| group | n_subjects | mean_original | mean_ea | mean_best_combined | mean_best_combined_delta_original | mean_best_combined_delta_ea | any_combined_reaches_70 | persistent_fail_all_combined | mean_n_combined_pass70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original<70 | 66 | 59.26 | 63.99 | 66.75 | 7.49 | 2.76 | 28/66 (42.4%) | 38/66 (57.6%) | 1.17 |
| Original>=70 | 40 | 82.21 | 84.05 | 86.08 | 3.86 | 2.02 | 40/40 (100.0%) | 0/40 (0.0%) | 3.85 |

## Method Effects Within Original<70 Group

| baseline_group | method | n | mean_acc | mean_delta_vs_original | mean_delta_vs_ea | pass70_rate_pct | responder_vs_original_pct | harm_vs_original_pct | responder_vs_ea_pct | harm_vs_ea_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original<70 | EA+Snapshot+AdaBN | 66 | 64.71 | 5.45 | 0.72 | 30.3 | 81.8 | 16.7 | 57.6 | 40.9 |
| Original<70 | EA+SupCon | 66 | 63.99 | 4.72 | -0.0 | 30.3 | 74.2 | 18.2 | 48.5 | 43.9 |
| Original<70 | EA+SupCon+AdaBN | 66 | 64.46 | 5.2 | 0.47 | 31.8 | 74.2 | 24.2 | 51.5 | 45.5 |
| Original<70 | EA+SupCon+CORAL | 66 | 64.03 | 4.77 | 0.04 | 24.2 | 72.7 | 21.2 | 47.0 | 47.0 |

## Persistent Illiteracy Under All Combined Methods

| dataset | subject | Original | EA | EA+Snapshot+AdaBN | EA+SupCon | EA+SupCon+AdaBN | EA+SupCon+CORAL | best_combined_acc | best_combined_method | best_combined_delta_original | best_combined_delta_ea |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lee2019 | 34 | 46.0 | 45.0 | 50.0 | 47.0 | 45.5 | 49.0 | 50.0 | EA+Snapshot+AdaBN | 4.0 | 5.0 |
| cho2017 | 17 | 48.5 | 55.5 | 55.5 | 59.0 | 56.0 | 58.0 | 59.0 | EA+SupCon | 10.5 | 3.5 |
| cho2017 | 28 | 49.5 | 49.5 | 45.5 | 49.5 | 49.0 | 48.0 | 49.5 | EA+SupCon | 0.0 | 0.0 |
| lee2019 | 50 | 49.5 | 51.0 | 56.5 | 50.5 | 56.0 | 54.5 | 56.5 | EA+Snapshot+AdaBN | 7.0 | 5.5 |
| cho2017 | 37 | 50.5 | 57.0 | 51.0 | 49.5 | 52.5 | 55.0 | 55.0 | EA+SupCon+CORAL | 4.5 | -2.0 |
| lee2019 | 35 | 50.5 | 54.0 | 58.0 | 54.5 | 52.5 | 57.0 | 58.0 | EA+Snapshot+AdaBN | 7.5 | 4.0 |
| cho2017 | 27 | 50.5 | 58.0 | 62.5 | 57.5 | 56.0 | 60.0 | 62.5 | EA+Snapshot+AdaBN | 12.0 | 4.5 |
| lee2019 | 15 | 51.5 | 60.0 | 63.0 | 56.0 | 58.5 | 56.5 | 63.0 | EA+Snapshot+AdaBN | 11.5 | 3.0 |
| cho2017 | 40 | 52.0 | 54.0 | 53.0 | 53.0 | 51.0 | 52.0 | 53.0 | EA+Snapshot+AdaBN | 1.0 | -1.0 |
| lee2019 | 48 | 52.0 | 58.5 | 56.5 | 56.5 | 57.0 | 60.0 | 60.0 | EA+SupCon+CORAL | 8.0 | 1.5 |
| cho2017 | 34 | 53.0 | 52.0 | 54.5 | 57.0 | 52.0 | 53.5 | 57.0 | EA+SupCon | 4.0 | 5.0 |
| lee2019 | 24 | 53.5 | 54.5 | 58.0 | 56.0 | 55.0 | 57.5 | 58.0 | EA+Snapshot+AdaBN | 4.5 | 3.5 |
| cho2017 | 2 | 53.5 | 58.5 | 58.0 | 56.5 | 55.0 | 58.5 | 58.5 | EA+SupCon+CORAL | 5.0 | 0.0 |
| lee2019 | 27 | 54.5 | 55.5 | 58.5 | 51.5 | 58.0 | 56.5 | 58.5 | EA+Snapshot+AdaBN | 4.0 | 3.0 |
| lee2019 | 13 | 55.0 | 62.5 | 60.5 | 62.5 | 61.5 | 64.5 | 64.5 | EA+SupCon+CORAL | 9.5 | 2.0 |
| cho2017 | 13 | 55.5 | 53.0 | 55.0 | 52.5 | 52.0 | 56.5 | 56.5 | EA+SupCon+CORAL | 1.0 | 3.5 |
| cho2017 | 8 | 55.5 | 54.5 | 54.0 | 55.5 | 58.5 | 58.0 | 58.5 | EA+SupCon+AdaBN | 3.0 | 4.0 |
| lee2019 | 11 | 56.5 | 54.5 | 58.5 | 56.5 | 57.0 | 53.5 | 58.5 | EA+Snapshot+AdaBN | 2.0 | 4.0 |
| lee2019 | 41 | 56.5 | 59.5 | 63.0 | 56.5 | 60.5 | 56.5 | 63.0 | EA+Snapshot+AdaBN | 6.5 | 3.5 |
| lee2019 | 51 | 56.5 | 68.5 | 66.0 | 64.5 | 65.0 | 64.0 | 66.0 | EA+Snapshot+AdaBN | 9.5 | -2.5 |
| cho2017 | 32 | 57.0 | 54.0 | 55.5 | 53.5 | 53.0 | 54.0 | 55.5 | EA+Snapshot+AdaBN | -1.5 | 1.5 |
| lee2019 | 10 | 57.5 | 67.0 | 69.0 | 68.0 | 67.5 | 65.5 | 69.0 | EA+Snapshot+AdaBN | 11.5 | 2.0 |
| cho2017 | 35 | 58.0 | 57.5 | 58.0 | 56.5 | 56.0 | 55.0 | 58.0 | EA+Snapshot+AdaBN | 0.0 | 0.5 |
| lee2019 | 54 | 58.0 | 61.5 | 62.0 | 61.5 | 57.5 | 64.0 | 64.0 | EA+SupCon+CORAL | 6.0 | 2.5 |
| lee2019 | 25 | 59.0 | 56.0 | 61.0 | 60.5 | 62.0 | 59.0 | 62.0 | EA+SupCon+AdaBN | 3.0 | 6.0 |
| cho2017 | 29 | 60.0 | 56.0 | 55.5 | 54.5 | 58.5 | 53.5 | 58.5 | EA+SupCon+AdaBN | -1.5 | 2.5 |
| lee2019 | 40 | 60.0 | 69.0 | 65.0 | 61.0 | 65.5 | 61.5 | 65.5 | EA+SupCon+AdaBN | 5.5 | -3.5 |
| lee2019 | 7 | 60.0 | 64.5 | 63.5 | 66.0 | 67.0 | 61.5 | 67.0 | EA+SupCon+AdaBN | 7.0 | 2.5 |
| cho2017 | 6 | 61.0 | 67.5 | 58.5 | 66.0 | 67.0 | 67.0 | 67.0 | EA+SupCon+AdaBN | 6.0 | -0.5 |
| lee2019 | 38 | 61.5 | 57.5 | 60.5 | 61.5 | 58.5 | 58.5 | 61.5 | EA+SupCon | 0.0 | 4.0 |
| lee2019 | 4 | 61.5 | 60.0 | 62.0 | 57.5 | 59.0 | 57.5 | 62.0 | EA+Snapshot+AdaBN | 0.5 | 2.0 |
| lee2019 | 16 | 61.5 | 58.0 | 62.0 | 60.5 | 61.5 | 60.5 | 62.0 | EA+Snapshot+AdaBN | 0.5 | 4.0 |
| lee2019 | 53 | 62.0 | 64.0 | 67.0 | 60.5 | 61.0 | 58.5 | 67.0 | EA+Snapshot+AdaBN | 5.0 | 3.0 |
| cho2017 | 16 | 62.0 | 63.0 | 65.5 | 69.5 | 68.5 | 64.0 | 69.5 | EA+SupCon | 7.5 | 6.5 |
| cho2017 | 7 | 64.17 | 62.5 | 63.75 | 62.92 | 62.92 | 65.42 | 65.42 | EA+SupCon+CORAL | 1.25 | 2.92 |
| lee2019 | 26 | 66.0 | 63.5 | 63.0 | 60.0 | 63.0 | 63.0 | 63.0 | EA+Snapshot+AdaBN | -3.0 | -0.5 |
| cho2017 | 33 | 67.0 | 67.5 | 68.5 | 68.0 | 67.5 | 64.0 | 68.5 | EA+Snapshot+AdaBN | 1.5 | 1.0 |
| cho2017 | 1 | 67.5 | 68.5 | 64.0 | 68.0 | 66.0 | 68.0 | 68.0 | EA+SupCon | 0.5 | -0.5 |

## Recovered Illiteracy Under Combined Methods

| dataset | subject | Original | EA | EA+Snapshot+AdaBN | EA+SupCon | EA+SupCon+AdaBN | EA+SupCon+CORAL | best_combined_acc | best_combined_method | best_combined_delta_original | best_combined_delta_ea | n_combined_pass70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cho2017 | 39 | 50.5 | 79.5 | 78.5 | 80.0 | 75.5 | 78.0 | 80.0 | EA+SupCon | 29.5 | 0.5 | 4 |
| cho2017 | 46 | 62.5 | 81.25 | 82.5 | 75.0 | 84.58 | 79.58 | 84.58 | EA+SupCon+AdaBN | 22.08 | 3.33 | 4 |
| cho2017 | 12 | 55.0 | 71.5 | 73.0 | 72.0 | 74.0 | 72.5 | 74.0 | EA+SupCon+AdaBN | 19.0 | 2.5 | 4 |
| cho2017 | 15 | 63.5 | 80.0 | 79.0 | 82.0 | 80.5 | 81.0 | 82.0 | EA+SupCon | 18.5 | 2.0 | 4 |
| cho2017 | 38 | 52.5 | 69.5 | 70.0 | 69.0 | 67.0 | 66.5 | 70.0 | EA+Snapshot+AdaBN | 17.5 | 0.5 | 1 |
| cho2017 | 11 | 60.5 | 71.0 | 72.0 | 72.5 | 73.0 | 76.0 | 76.0 | EA+SupCon+CORAL | 15.5 | 5.0 | 4 |
| cho2017 | 19 | 58.5 | 76.0 | 74.0 | 72.5 | 72.0 | 71.0 | 74.0 | EA+Snapshot+AdaBN | 15.5 | -2.0 | 4 |
| cho2017 | 30 | 60.5 | 63.5 | 75.5 | 71.0 | 74.0 | 68.5 | 75.5 | EA+Snapshot+AdaBN | 15.0 | 12.0 | 3 |
| lee2019 | 49 | 61.0 | 68.0 | 76.0 | 69.0 | 69.5 | 70.0 | 76.0 | EA+Snapshot+AdaBN | 15.0 | 8.0 | 2 |
| cho2017 | 20 | 59.0 | 70.5 | 73.5 | 62.5 | 71.5 | 71.0 | 73.5 | EA+Snapshot+AdaBN | 14.5 | 3.0 | 3 |
| cho2017 | 5 | 61.0 | 72.0 | 69.0 | 72.0 | 71.0 | 74.5 | 74.5 | EA+SupCon+CORAL | 13.5 | 2.5 | 3 |
| cho2017 | 50 | 56.5 | 70.5 | 68.5 | 70.0 | 68.5 | 69.5 | 70.0 | EA+SupCon | 13.5 | -0.5 | 1 |
| lee2019 | 22 | 65.0 | 70.0 | 72.5 | 72.0 | 77.0 | 68.5 | 77.0 | EA+SupCon+AdaBN | 12.0 | 7.0 | 3 |
| cho2017 | 26 | 68.5 | 72.0 | 71.5 | 73.0 | 79.5 | 71.0 | 79.5 | EA+SupCon+AdaBN | 11.0 | 7.5 | 4 |
| cho2017 | 31 | 61.0 | 66.0 | 70.0 | 70.0 | 72.0 | 69.5 | 72.0 | EA+SupCon+AdaBN | 11.0 | 6.0 | 3 |
| cho2017 | 49 | 67.5 | 73.5 | 70.5 | 78.0 | 68.5 | 76.0 | 78.0 | EA+SupCon | 10.5 | 4.5 | 3 |
| lee2019 | 31 | 62.0 | 67.5 | 72.0 | 69.0 | 66.5 | 71.0 | 72.0 | EA+Snapshot+AdaBN | 10.0 | 4.5 | 2 |
| cho2017 | 47 | 66.0 | 74.0 | 70.5 | 74.0 | 76.0 | 75.0 | 76.0 | EA+SupCon+AdaBN | 10.0 | 2.0 | 4 |
| cho2017 | 18 | 66.0 | 68.5 | 69.0 | 66.5 | 71.0 | 74.0 | 74.0 | EA+SupCon+CORAL | 8.0 | 5.5 | 2 |
| lee2019 | 17 | 67.0 | 68.0 | 72.0 | 73.0 | 73.5 | 73.5 | 73.5 | EA+SupCon+AdaBN | 6.5 | 5.5 | 4 |
| lee2019 | 29 | 66.0 | 71.5 | 65.5 | 72.0 | 64.5 | 63.5 | 72.0 | EA+SupCon | 6.0 | 0.5 | 1 |
| cho2017 | 45 | 65.0 | 68.5 | 65.5 | 71.0 | 68.0 | 68.5 | 71.0 | EA+SupCon | 6.0 | 2.5 | 1 |
| cho2017 | 51 | 64.0 | 68.0 | 66.0 | 67.0 | 70.0 | 67.0 | 70.0 | EA+SupCon+AdaBN | 6.0 | 2.0 | 1 |
| lee2019 | 52 | 66.5 | 65.5 | 72.5 | 70.5 | 70.0 | 65.5 | 72.5 | EA+Snapshot+AdaBN | 6.0 | 7.0 | 3 |
| lee2019 | 12 | 66.0 | 67.5 | 68.0 | 68.5 | 71.5 | 66.0 | 71.5 | EA+SupCon+AdaBN | 5.5 | 4.0 | 1 |
| cho2017 | 36 | 67.5 | 71.5 | 68.5 | 72.5 | 72.5 | 69.5 | 72.5 | EA+SupCon | 5.0 | 1.0 | 2 |
| cho2017 | 9 | 69.58 | 74.58 | 72.92 | 74.17 | 71.67 | 72.92 | 74.17 | EA+SupCon | 4.58 | -0.42 | 4 |
| lee2019 | 42 | 68.5 | 70.5 | 71.0 | 66.5 | 71.0 | 67.0 | 71.0 | EA+Snapshot+AdaBN | 2.5 | 0.5 | 2 |

## Interpretation

- Combined methods can recover some Original<70 subjects, but a persistent low-performing subgroup remains.
- Compare `best_combined_delta_ea` to judge whether combinations add benefit beyond plain EA.
- If `mean_best_combined_delta_ea` is small, the main effect is still EA/alignment rather than the extra combined component.
