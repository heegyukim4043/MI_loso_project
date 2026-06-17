# Cross-Dataset Final Matrix Queue (2026-06-16)

Last updated: `2026-06-16 23:39` KST

- Input: `/home/hkim/MI_test/preprocessed_sfreq100`, `MI_N_TIMES=201`
- SubjEA = `--ea`; DSA+SEA = `--dataset_ea --ea`; SessionEA = `--session_ea`.
- TTA metric columns: AdaBN=`adabn_acc`, TENT=`tent_acc`, Snapshot=`snap_acc`.

| Priority | Group | Method | Status | Cho->Lee | Lee->Cho | Avg |
|---:|---|---|---|---:|---:|---:|
| 0 | Reference | CSPNet + DSA+SEA+SessionEA+SourceWeight | completed | 73.24% / k=0.465 (n=54) | 69.21% / k=0.384 (n=52) | 71.22% |
| 0 | Reference | CSPNet + DSA+SEA+SourceWeight | completed | 71.93% / k=0.439 (n=54) | 69.12% / k=0.382 (n=52) | 70.52% |
| 0 | Reference | EEGNet + DSA+SEA | completed | 70.81% / k=0.416 (n=54) | 68.56% / k=0.371 (n=52) | 69.68% |
| 0 | Reference | Conformer + DSA+SEA | completed | 71.62% / k=0.432 (n=54) | 67.97% / k=0.359 (n=52) | 69.79% |
| 1 | SubjEA TTA/Snapshot | EEGNet + SubjEA+AdaBN | completed | 63.95% / k=0.279 (n=54) | 64.74% / k=0.295 (n=52) | 64.35% |
| 1 | SubjEA TTA/Snapshot | EEGNet + SubjEA+Snapshotx6 | completed | 62.70% / k=0.254 (n=54) | 63.04% / k=0.261 (n=52) | 62.87% |
| 1 | SubjEA TTA/Snapshot | Conformer + SubjEA+AdaBN | completed | 65.76% / k=0.315 (n=54) | 64.54% / k=0.291 (n=52) | 65.15% |
| 1 | SubjEA TTA/Snapshot | Conformer + SubjEA+Snapshotx6 | completed | 60.68% / k=0.214 (n=54) | 63.43% / k=0.269 (n=52) | 62.06% |
| 2 | DSA+SEA TTA | CSPNet + DSA+SEA+AdaBN | completed | 72.41% / k=0.448 (n=54) | 69.90% / k=0.398 (n=52) | 71.15% |
| 2 | DSA+SEA TTA | CSPNet + DSA+SEA+TENT | completed | 72.22% / k=0.444 (n=54) | 68.54% / k=0.371 (n=52) | 70.38% |
| 2 | DSA+SEA TTA | EEGNet + DSA+SEA+AdaBN | completed | 70.72% / k=0.414 (n=54) | 68.90% / k=0.378 (n=52) | 69.81% |
| 2 | DSA+SEA TTA | EEGNet + DSA+SEA+TENT | completed | 70.17% / k=0.403 (n=54) | 68.48% / k=0.370 (n=52) | 69.32% |
| 2 | DSA+SEA TTA | Conformer + DSA+SEA+AdaBN | completed | 71.35% / k=0.427 (n=54) | 68.46% / k=0.369 (n=52) | 69.91% |
| 2 | DSA+SEA TTA | Conformer + DSA+SEA+TENT | completed | 70.04% / k=0.401 (n=54) | 62.82% / k=0.256 (n=52) | 66.43% |
| 3 | Best method backbone extension | EEGNet + DSA+SEA+SessionEA+SourceWeight | completed | 71.38% / k=0.428 (n=54) | 68.77% / k=0.375 (n=52) | 70.07% |
| 3 | Best method backbone extension | Conformer + DSA+SEA+SessionEA+SourceWeight | completed | 71.98% / k=0.440 (n=54) | 69.40% / k=0.388 (n=52) | 70.69% |

## Queue Status

- Queue completed.
