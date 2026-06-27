# LOSO Subject-Level Robustness Metrics

Generated from loso/results/*.csv on 2026-06-24.

Definitions:

- P10: 10th percentile subject accuracy. Higher means better lower-tail robustness.
- Coverage >=60/70: percentage of subjects whose accuracy is at least 60% or 70%.
- NTR: negative transfer rate in paired comparisons, i.e. percentage of subjects where the candidate is worse than the baseline.
- Paired comparisons use the same dataset and subject id.

## Selected 3x5 LOSO Matrix Robustness

| method | backbone | gen_method | n | mean_acc | median_acc | p10_acc | iqr_acc | coverage_ge_60_pct | coverage_ge_70_pct | min_acc | max_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conformer_noea | Conformer | No EA | 106 | 66.49 | 64 | 53 | 17.5 | 66 | 32.1 | 47.5 | 98 |
| cspnet_noea | CSPNet | No EA | 106 | 67.92 | 66 | 52 | 19 | 70.8 | 37.7 | 46 | 96.5 |
| ea_adabn_conformer | Conformer | EA+AdaBN | 106 | 71.4 | 70.5 | 56.5 | 19 | 84 | 53.8 | 50 | 97.5 |
| ea_adabn_cspnet | CSPNet | EA+AdaBN | 106 | 71.64 | 70.5 | 54 | 20.5 | 77.4 | 53.8 | 48 | 97 |
| ea_adabn_eegnet | EEGNet | EA+AdaBN | 106 | 70.77 | 69.5 | 54 | 18 | 81.1 | 48.1 | 44.5 | 96.5 |
| ea_conformer | Conformer | EA | 106 | 69.54 | 67.5 | 55 | 14 | 80.2 | 44.3 | 47.5 | 97.5 |
| ea_cspnet | CSPNet | EA | 106 | 71.56 | 70.5 | 54.5 | 17.5 | 78.3 | 52.8 | 45 | 97 |
| ea_eegnet | EEGNet | EA | 106 | 70.47 | 68.5 | 55 | 14.58 | 82.1 | 44.3 | 48 | 97 |
| ea_snapshot_adabn_cspnet | CSPNet | EA+Snapshot | 106 | 72.09 | 71.5 | 56.5 | 17 | 82.1 | 55.7 | 45.5 | 96.5 |
| ea_snapshot_conformer | Conformer | EA+Snapshot | 106 | 70.07 | 68.5 | 54.5 | 18 | 77.4 | 48.1 | 49 | 98 |
| ea_snapshot_eegnet | EEGNet | EA+Snapshot | 106 | 70.08 | 68.5 | 53.5 | 19.2 | 76.4 | 47.2 | 46.5 | 97.5 |
| ea_tent_conformer | Conformer | EA+TENT | 106 | 67.84 | 65 | 53 | 18 | 69.8 | 43.4 | 48.5 | 97.5 |
| ea_tent_cspnet | CSPNet | EA+TENT | 106 | 71.99 | 70.5 | 55 | 17.5 | 77.4 | 54.7 | 50 | 97.5 |
| ea_tent_eegnet | EEGNet | EA+TENT | 106 | 70.57 | 69 | 54 | 16 | 79.2 | 47.2 | 45 | 97.5 |
| eegnet_noea | EEGNet | No EA | 106 | 67.75 | 64.5 | 53 | 16 | 68.9 | 35.8 | 48 | 97 |

## Top Methods By Overall Mean Accuracy

| method | backbone | gen_method | mean_acc | median_acc | p10_acc | coverage_ge_60_pct | coverage_ge_70_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ea_snapshot_adabn_cspnet | CSPNet | EA+Snapshot | 72.09 | 71.5 | 56.5 | 82.1 | 55.7 |
| ea_supcon_adabn_cspnet | CSPNet | EA+SupCon+AdaBN | 72.08 | 71.5 | 56 | 78.3 | 57.5 |
| ea_tent_cspnet | CSPNet | EA+TENT | 71.99 | 70.5 | 55 | 77.4 | 54.7 |
| ea_subjclust_tau5_cspnet | CSPNet | EA+SubjectClust | 71.85 | 70.5 | 54 | 80.2 | 57.5 |
| ea_adabn_cspnet | CSPNet | EA+AdaBN | 71.64 | 70.5 | 54 | 77.4 | 53.8 |
| ea_cspnet | CSPNet | EA | 71.56 | 70.5 | 54.5 | 78.3 | 52.8 |
| ea_snapshot_adabn_x4_cspnet | CSPNet | EA+Snapshot | 71.52 | 69.5 | 56.5 | 84 | 49.1 |
| ea_snapshot_adabn_x3_cspnet | CSPNet | EA+Snapshot | 71.44 | 70.5 | 55.5 | 79.2 | 52.8 |
| ea_adabn_conformer | Conformer | EA+AdaBN | 71.4 | 70.5 | 56.5 | 84 | 53.8 |
| ea_supcon_cspnet | CSPNet | EA+SupCon | 71.35 | 70.5 | 55.5 | 79.2 | 53.8 |
| ea_supcon_coral_cspnet | CSPNet | EA+SupCon+CORAL | 71.34 | 70 | 56.5 | 78.3 | 50.9 |
| ea_subjclust_tau1_cspnet | CSPNet | EA+SubjectClust | 71.22 | 70.5 | 55 | 80.2 | 50.9 |

## Paired Delta Metrics

Positive mean_delta means the candidate improves over the baseline. NTR is the subject-level failure rate relative to the baseline.

| group | baseline | candidate | n | mean_delta | median_delta | p10_delta | wins | ties | losses | win_rate_pct | ntr_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EA_to_Contrastive | ea_cspnet | ea_supcon_cspnet | 106 | -0.21 | 0 | -3.5 | 43 | 11 | 52 | 40.6 | 49.1 |
| EA_to_Contrastive_Align | ea_cspnet | ea_supcon_coral_cspnet | 106 | -0.22 | 0 | -3.5 | 46 | 8 | 52 | 43.4 | 49.1 |
| EA_to_Contrastive_TTA | ea_cspnet | ea_supcon_adabn_cspnet | 106 | 0.52 | 0.42 | -3 | 54 | 5 | 47 | 50.9 | 44.3 |
| EA_to_Snapshot | ea_cspnet | ea_snapshot_adabn_cspnet | 106 | 0.53 | 0.5 | -3.5 | 56 | 5 | 45 | 52.8 | 42.5 |
| EA_to_Snapshot | ea_conformer | ea_snapshot_conformer | 106 | 0.53 | 0.5 | -6 | 54 | 4 | 48 | 50.9 | 45.3 |
| EA_to_Snapshot | ea_eegnet | ea_snapshot_eegnet | 106 | -0.39 | -1 | -6 | 40 | 6 | 60 | 37.7 | 56.6 |
| EA_to_TTA | ea_conformer | ea_adabn_conformer | 106 | 1.86 | 1.5 | -3.5 | 65 | 5 | 36 | 61.3 | 34 |
| EA_to_TTA | ea_cspnet | ea_adabn_cspnet | 106 | 0.08 | -0.5 | -3.5 | 47 | 6 | 53 | 44.3 | 50 |
| EA_to_TTA | ea_eegnet | ea_adabn_eegnet | 106 | 0.3 | 0 | -4.5 | 50 | 9 | 47 | 47.2 | 44.3 |
| EA_to_TTA | ea_conformer | ea_tent_conformer | 106 | -1.69 | -0.5 | -9 | 39 | 6 | 61 | 36.8 | 57.5 |
| EA_to_TTA | ea_cspnet | ea_tent_cspnet | 106 | 0.43 | 0.5 | -3.5 | 57 | 8 | 41 | 53.8 | 38.7 |
| EA_to_TTA | ea_eegnet | ea_tent_eegnet | 106 | 0.09 | -0.5 | -3 | 45 | 6 | 55 | 42.5 | 51.9 |
| NoEA_to_EA | conformer_noea | ea_conformer | 106 | 3.05 | 1.5 | -3.5 | 66 | 5 | 35 | 62.3 | 33 |
| NoEA_to_EA | cspnet_noea | ea_cspnet | 106 | 3.64 | 3 | -3 | 74 | 4 | 28 | 69.8 | 26.4 |
| NoEA_to_EA | eegnet_noea | ea_eegnet | 106 | 2.72 | 3 | -3.5 | 74 | 7 | 25 | 69.8 | 23.6 |

## NoEA -> EA By Dataset

| baseline | candidate | dataset | n | mean_delta | median_delta | wins | losses | win_rate_pct | ntr_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conformer_noea | ea_conformer | cho2017 | 52 | 5.45 | 4 | 38 | 12 | 73.1 | 23.1 |
| conformer_noea | ea_conformer | lee2019 | 54 | 0.74 | 0.5 | 28 | 23 | 51.9 | 42.6 |
| cspnet_noea | ea_cspnet | cho2017 | 52 | 5.66 | 5 | 42 | 9 | 80.8 | 17.3 |
| cspnet_noea | ea_cspnet | lee2019 | 54 | 1.69 | 1.5 | 32 | 19 | 59.3 | 35.2 |
| eegnet_noea | ea_eegnet | cho2017 | 52 | 3.62 | 3 | 37 | 12 | 71.2 | 23.1 |
| eegnet_noea | ea_eegnet | lee2019 | 54 | 1.86 | 1.5 | 37 | 13 | 68.5 | 24.1 |

## Main Readout

1. EA improves all three backbones in paired subject-level comparison: EEGNet +2.72, CSPNet +3.64, Conformer +3.05 percentage points overall.
2. The best overall robustness cluster is CSPNet with EA plus Snapshot/AdaBN, SupCon/AdaBN, or TENT.
3. Combination methods are not uniformly safe. For example, Conformer+TENT has negative mean delta and high NTR, while Conformer+AdaBN has the strongest positive TTA delta.
4. Lower-tail robustness remains an issue: even top methods have P10 around 55-56.5%, so mean accuracy alone is not enough.
