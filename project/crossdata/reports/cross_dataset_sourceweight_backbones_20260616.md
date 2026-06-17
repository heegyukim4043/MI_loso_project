# Cross-Dataset SourceWeight Backbone Check (2026-06-16)

Last updated: `2026-06-16 08:27` KST

- Input: `/home/hkim/MI_test/preprocessed_sfreq100`, `MI_N_TIMES=201`
- DSA+SEA = `--dataset_ea --ea`; SourceWeight = covariance-similarity weighted source sampler with `tau=5.0`.

| Method | Status | Cho->Lee | Lee->Cho | Avg Acc |
|---|---|---:|---:|---:|
| CSPNet + DSA+SEA+SourceWeight tau=5 | completed | 71.93% / k=0.439 (n=54) | 69.12% / k=0.382 (n=52) | 70.52% |
| CSPNet + DSA+SEA+SessionEA+SourceWeight tau=5 | completed | 73.24% / k=0.465 (n=54) | 69.21% / k=0.384 (n=52) | 71.22% |
| EEGNet + DSA+SEA+SourceWeight tau=5 | completed | 70.02% / k=0.400 (n=54) | 67.75% / k=0.355 (n=52) | 68.88% |
| Conformer + DSA+SEA+SourceWeight tau=5 | completed | 70.35% / k=0.407 (n=54) | 69.07% / k=0.381 (n=52) | 69.71% |

## Queue Status

- Queue completed.
