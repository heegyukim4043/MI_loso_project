# EEG Marker-Based Generalization Analysis

These are task-window EEG markers computed from the streamed NPZ trials. Because no pre-cue baseline is stored, `contra_desync_index` is a desynchronization-like lateralization index, not strict baseline-corrected ERD.

## Marker Definitions

- `*_hemi_asym_absdiff`: absolute left/right class difference in hemispheric asymmetry `log(right sensorimotor power) - log(left sensorimotor power)`.
- `*_contra_desync_index`: task-window ipsilateral minus contralateral log bandpower. Positive values are consistent with lower contralateral power.
- `*_sensorimotor_vs_all`: sensorimotor channel logpower relative to all-channel logpower.
- `*_class_sensorimotor_absdiff`: class difference in sensorimotor log bandpower.

## Dataset Means

| dataset | mu_hemi_asym_absdiff | beta_hemi_asym_absdiff | mu_beta_hemi_asym_absdiff | mu_contra_desync_index | beta_contra_desync_index | mu_beta_class_sensorimotor_absdiff |
| --- | --- | --- | --- | --- | --- | --- |
| cho2017 | 0.119 | 0.077 | 0.086 | 0.035 | 0.021 | 0.102 |
| lee2019 | 0.225 | 0.138 | 0.185 | 0.105 | 0.063 | 0.060 |

## Illiteracy / Persistent Failure Groups

| dataset | original_illiterate | persistent_fail | n | original_acc | best_generalization_acc | mu_hemi_asym_absdiff | beta_hemi_asym_absdiff | mu_contra_desync_index | beta_contra_desync_index | mu_beta_class_sensorimotor_absdiff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cho2017 | False | False | 14 | 80.184 | 85.554 | 0.217 | 0.127 | 0.102 | 0.057 | 0.092 |
| cho2017 | True | False | 15 | 63.289 | 74.122 | 0.107 | 0.067 | 0.032 | 0.031 | 0.130 |
| cho2017 | True | True | 23 | 57.344 | 61.305 | 0.067 | 0.054 | -0.004 | -0.006 | 0.089 |
| lee2019 | False | False | 23 | 81.659 | 85.362 | 0.379 | 0.198 | 0.190 | 0.099 | 0.053 |
| lee2019 | False | True | 1 | 72.000 | 66.833 | 0.432 | 0.952 | 0.216 | 0.476 | 0.230 |
| lee2019 | True | False | 8 | 66.208 | 72.917 | 0.164 | 0.077 | 0.082 | 0.039 | 0.067 |
| lee2019 | True | True | 22 | 57.826 | 61.129 | 0.076 | 0.061 | 0.021 | 0.015 | 0.057 |

## Strongest EEG Marker Correlations

| dataset | feature | target | n | rho | p |
| --- | --- | --- | --- | --- | --- |
| lee2019 | mu_beta_contra_desync_index | best_generalization_acc | 54 | 0.833 | 0.000 |
| lee2019 | mu_beta_hemi_asym_diff | best_generalization_acc | 54 | -0.833 | 0.000 |
| lee2019 | mu_beta_hemi_asym_absdiff | best_generalization_acc | 54 | 0.824 | 0.000 |
| lee2019 | mu_beta_hemi_asym_diff | original_acc | 54 | -0.816 | 0.000 |
| lee2019 | mu_beta_contra_desync_index | original_acc | 54 | 0.816 | 0.000 |
| lee2019 | mu_beta_hemi_asym_absdiff | original_acc | 54 | 0.804 | 0.000 |
| lee2019 | mu_hemi_asym_diff | best_generalization_acc | 54 | -0.770 | 0.000 |
| lee2019 | mu_contra_desync_index | best_generalization_acc | 54 | 0.770 | 0.000 |
| lee2019 | mu_hemi_asym_absdiff | best_generalization_acc | 54 | 0.762 | 0.000 |
| lee2019 | mu_contra_desync_index | original_acc | 54 | 0.743 | 0.000 |
| lee2019 | mu_hemi_asym_diff | original_acc | 54 | -0.743 | 0.000 |
| lee2019 | mu_hemi_asym_absdiff | original_acc | 54 | 0.732 | 0.000 |
| lee2019 | beta_hemi_asym_diff | original_acc | 54 | -0.694 | 0.000 |
| lee2019 | beta_contra_desync_index | original_acc | 54 | 0.694 | 0.000 |
| lee2019 | beta_contra_desync_index | best_generalization_acc | 54 | 0.666 | 0.000 |
| lee2019 | beta_hemi_asym_diff | best_generalization_acc | 54 | -0.666 | 0.000 |
| lee2019 | beta_hemi_asym_absdiff | original_acc | 54 | 0.617 | 0.000 |
| lee2019 | beta_hemi_asym_absdiff | best_generalization_acc | 54 | 0.594 | 0.000 |
| lee2019 | mu_hemi_asym_right | original_acc | 54 | 0.585 | 0.000 |
| lee2019 | mu_beta_hemi_asym_right | original_acc | 54 | 0.580 | 0.000 |

## Interpretation

- EEG markers are useful if they separate recovered vs persistent low-performing subjects beyond accuracy alone.
- Lateralized mu/beta asymmetry is the most physiology-aligned marker for left/right MI.
- Use these as construct-validity features alongside CSP separability and covariance geometry.
