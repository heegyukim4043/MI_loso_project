# LOSO Methods Applied To Cross-Dataset Classification

This table maps methods used in LOSO validation to their cross-dataset counterparts and reports cross-dataset classification performance.

- Direction 1: Cho2017 train -> Lee2019 test
- Direction 2: Lee2019 train -> Cho2017 test
- Accuracy, Precision, and F1 are percentages; kappa is Cohen kappa.
- Some LOSO methods do not have a direct cross-dataset run; those are listed separately.

- CSV: `/home/hkim/MI_test/loso_methods_applied_cross_dataset_performance.csv`

## Cross-Dataset Counterparts

| LOSO method | Cross-dataset counterpart | Relation | Cho->Lee Acc | Lee->Cho Acc | Avg Acc | Cho->Lee Kappa | Lee->Cho Kappa | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| CSP-LDA | StdMI+CSP-LDA | direct | 55.15 | 52.76 | 53.95 | 0.103 | 0.055 | Classical baseline with standard MI channel cross-dataset setup. |
| EA+CSP-LDA | StdMI+EA+CSP-LDA | direct | 52.87 | 59.23 | 56.05 | 0.057 | 0.185 | Classical EA baseline. Subject-level EA only. |
| CSPNet baseline | CSPNet baseline | direct | 50.02 | 52.70 | 51.36 | 0.000 | 0.054 | Neural CSPNet baseline without EA. |
| EA+CSPNet | EA+CSPNet | direct | 53.69 | 52.60 | 53.14 | 0.074 | 0.052 | Simple EA+CSPNet cross-dataset baseline. |
| EA+TENT | EA+TENT+TENT_eval | direct | 59.80 | 56.09 | 57.94 | 0.196 | 0.122 | TENT-evaluated cross-dataset counterpart. |
| EA+AdaBN+Con | EA+AdaBN+Con+AdaBN_eval | direct | 59.80 | 57.19 | 58.49 | 0.196 | 0.144 | AdaBN-evaluated contrastive counterpart. |
| EA+PSDNorm+CSPNet | EA+PSDNorm+cspnet | cross-extension | 62.32 | 59.12 | 60.72 | 0.246 | 0.182 | Power-normalized EA+CSPNet cross-dataset variant. |
| DatasetEA+SubjectEA+CSPNet | DatasetEA+SubjectEA+cspnet | cross-extension | 62.35 | 60.19 | 61.27 | 0.247 | 0.204 | Cross-dataset alignment extension of EA+CSPNet. |
| DatasetEA+SubjectEA+AdaBN+Con | DatasetEA+SubjectEA+adabn+con+AdaBN_eval | cross-extension | 62.69 | 59.80 | 61.25 | 0.254 | 0.196 | Cross-dataset alignment extension of EA+AdaBN+Con. |
| RawUnified+DatasetEA+SubjectEA+CSP-LDA | RawUnified+DatasetEA+SubjectEA+CSP-LDA | cross-extension | 68.96 | 65.10 | 67.03 | 0.379 | 0.302 | Raw preprocessing unified final classical baseline. |
| SessionEA+CSP-LDA | SessionEA+CSP-LDA | cross-extension | 70.74 | 65.14 | 67.94 | 0.415 | 0.303 | Session-level alignment extension; Lee2019 session structure used. |
| SessionEA+Feature-CORAL+CSP-LDA | SessionEA+Feature-CORAL+CSP-LDA | cross-extension | 70.20 | 65.66 | 67.93 | 0.404 | 0.313 | Main final cross-dataset candidate. |
| SessionEA+Feature-CORAL+MMD+CSP-LDA | SessionEA+Feature-CORAL+MMD-resample+CSP-LDA | cross-extension | 70.29 | 65.66 | 67.97 | 0.406 | 0.313 | MMD-added variant; marginally highest mean, extra complexity. |
| SubjClust / subject weighting | StdMI+DatasetEA+SubjectEA+SourceWeightTau0.1+cspnet | related | 62.89 | 59.37 | 61.13 | 0.258 | 0.188 | Closest cross-dataset analogue: source subject weighting. |
| Source subject selection | SourceSelect-k30+CSP-LDA | related | 69.57 | 65.14 | 67.36 | 0.391 | 0.303 | Closest selection-style cross-dataset result. |

## LOSO Methods Without Cross-Dataset Counterpart

| LOSO method | Cross-dataset status | Note |
|---|---|---|
| EA+Snapshot | not run cross-dataset | Snapshot ensemble was LOSO-only. |
| EA+Snap(x3/x4) | not run cross-dataset | Snapshot variants were LOSO-only. |
| EEGNet (no EA) | not run cross-dataset | Backbone ablation was LOSO-only for Cho/Lee. |
| EEGNet+EA+TENT | not run cross-dataset | Backbone+TENT ablation was LOSO-only. |
| Conformer (no EA) | not run cross-dataset | Backbone ablation was LOSO-only for Cho/Lee. |
| Conformer+EA+TENT | not run cross-dataset | Backbone+TENT ablation was LOSO-only. |
| KMM-TrAdaBoost-LOSO | not run cross-dataset | Current KMM implementation is LOSO validation only. |
| DANN-v1/v2 | not run cross-dataset | Current DANN results are LOSO validation only. |
| SSCL-CSD-style | not run cross-dataset | Current SSCL-CSD-style run is LOSO validation only. |

## Interpretation

- Directly transferred deep LOSO methods such as `EA+CSPNet`, `EA+TENT`, and `EA+AdaBN+Con` remain around the high-50s to low-60s in cross-dataset transfer.
- The strongest cross-dataset results come from classical CSP-LDA with cross-dataset-specific alignment: `SessionEA`, `Feature-CORAL`, and DatasetEA/SubjectEA variants.
- This supports the claim that methods successful in LOSO do not necessarily transfer under stronger dataset shift, and that explicit dataset/session alignment is needed for Cho2017 <-> Lee2019.
- Snapshot, DANN, KMM-TrAdaBoost, SSCL-CSD-style, EEGNet, and Conformer have LOSO results but no completed cross-dataset counterpart in the current result set.
