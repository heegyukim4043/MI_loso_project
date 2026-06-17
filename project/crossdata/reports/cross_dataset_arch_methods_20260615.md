# Cross-Dataset Architecture Method Cells (2026-06-15)

Last updated: `2026-06-15 10:15` KST

- Input: `/home/hkim/MI_test/preprocessed_sfreq100`, `MI_N_TIMES=201`
- DSA+SEA means `--dataset_ea --ea` in `cross_dataset.py`.
- TENT results use `tent_acc/tent_kappa`; Snapshot uses `snap_acc/snap_kappa`.

| Method | Status | Cho->Lee | Lee->Cho |
|---|---|---:|---:|
| CSPNet + EA+Snapshot | completed | 63.81% / k=0.276 (n=54) | 61.36% / k=0.227 (n=52) |
| CSPNet + EA+TENT | completed | 66.59% / k=0.332 (n=54) | 64.13% / k=0.283 (n=52) |
| EEGNet + DSA+SEA | completed | 70.81% / k=0.416 (n=54) | 68.56% / k=0.371 (n=52) |
| EEGNet + EA+TENT | completed | 63.62% / k=0.272 (n=54) | 64.08% / k=0.282 (n=52) |
| Conformer + DSA+SEA | completed | 71.62% / k=0.432 (n=54) | 67.97% / k=0.359 (n=52) |
| Conformer + EA+TENT | completed | 65.43% / k=0.309 (n=54) | 60.06% / k=0.201 (n=52) |

## Queue Status

- Queue completed.
