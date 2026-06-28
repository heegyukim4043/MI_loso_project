# Colab DSA+SEA+Snapshot Cross-Dataset Queue

Last updated: `2026-06-28 15:46`

- Method: DatasetEA + SubjectEA + Snapshot ensemble
- Snapshot: `T0=50`; `cross_dataset.py` stores both `snap_acc` and `snap_adabn_acc`.
- DSA+SEA args: `--dataset_ea --ea`

| Model | Cho->Lee | Lee->Cho | Avg Snap | Avg Snap+AdaBN |
|---|---:|---:|---:|---:|
| cspnet | Snap=71.50% / Snap+AdaBN=71.69% (n=54) | Snap=68.60% / Snap+AdaBN=68.80% (n=52) | 70.05% | 70.24% |
| eegnet | Snap=70.31% / Snap+AdaBN=70.24% (n=54) | Snap=68.76% / Snap+AdaBN=68.85% (n=52) | 69.53% | 69.55% |
| conformer | Snap=70.16% / Snap+AdaBN=70.38% (n=54) | Snap=67.42% / Snap+AdaBN=67.42% (n=52) | 68.79% | 68.90% |

## Queue Status

- Queue completed.
