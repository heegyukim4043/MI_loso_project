# Cross-Dataset srate-100Hz Performance Summary

Last updated: `2026-06-12 23:09` KST

- Preprocessed dir: `/home/hkim/MI_test/preprocessed_sfreq100`
- Cho2017: resampled from 128Hz/T=257 to 100Hz/T=201
- Lee2019: native/current 100Hz/T=201 copy
- Evaluation script: `/home/hkim/MI_test/MI_loso_project/cross_dataset.py`
- Environment: `MI_PREPROCESSED_DIR=/home/hkim/MI_test/preprocessed_sfreq100`, `MI_N_TIMES=201`

| Priority | Method | Status | Cho->Lee | Lee->Cho | Note |
|---:|---|---|---:|---:|---|
| 1 | EA+CSPNet | completed | 59.24% / k=0.185 (n=54) | 61.42% / k=0.228 (n=52) |  |
| 2 | EA+AdaBN+Con | completed | 66.93% / k=0.339 (n=54) | 64.24% / k=0.285 (n=52) |  |
| 3 | EA+Snapx6 | not run |  |  | Not executed: cross_dataset.py has no snapshot ensemble evaluation path yet. |
| 4 | EA+AdaBN | completed | 66.93% / k=0.339 (n=54) | 64.24% / k=0.285 (n=52) |  |
| 5 | EA+TENT | completed | 66.59% / k=0.332 (n=54) | 64.13% / k=0.283 (n=52) |  |
| 6 | SubjClust tau=5 analogue | completed | 71.93% / k=0.439 (n=54) | 69.12% / k=0.382 (n=52) |  |

## Queue Status

- Queue completed.
