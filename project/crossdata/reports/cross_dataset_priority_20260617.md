# Cross-Dataset Priority Queue (2026-06-17)

Last updated: 2026-06-17 10:15 KST

- Input: /home/hkim/MI_test/preprocessed_sfreq100, MI_N_TIMES=201
- DSA+SEA = --dataset_ea --ea.
- AdaBN+Con uses SupCon contrastive wrappers for EEGNet/Conformer.
- DANN uses binary source-vs-target domain adversarial training with CSPNet.

| Priority | Group | Method | Status | Cho->Lee | Lee->Cho | Avg |
|---:|---|---|---|---:|---:|---:|
| 5 | DSA+SEA AdaBN+Con | EEGNet + DSA+SEA+AdaBN+Con | completed | 70.72% / k=0.414 (n=54) | 68.90% / k=0.378 (n=52) | 69.81% |
| 5 | DSA+SEA AdaBN+Con | Conformer + DSA+SEA+AdaBN+Con | pending | n=0 | n=0 |  |
| 4 | DANN | CSPNet + DSA+SEA+DANN | pending | n=0 | n=0 |  |

## Queue Status

- Last completed EEGNet + DSA+SEA+AdaBN+Con exit=0
