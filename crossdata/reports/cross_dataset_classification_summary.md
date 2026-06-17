# Cross-Dataset Classification Summary

## Files

- Full long table: `/home/hkim/MI_test/cross_dataset_classification_performance_table.csv`
- Full markdown table: `/home/hkim/MI_test/cross_dataset_classification_performance_table.md`
- Wide table for plotting: `/home/hkim/MI_test/cross_dataset_classification_performance_wide.csv`
- Top-20 comparison plot: `/home/hkim/MI_test/cross_dataset_classification_comparison_top20.png`
- Core method comparison plot: `/home/hkim/MI_test/cross_dataset_classification_comparison_core_methods.png`
- Direction trade-off scatter: `/home/hkim/MI_test/cross_dataset_classification_direction_scatter.png`

## Top Methods By Mean Accuracy Across Both Directions

| Rank | Method | Cho->Lee | Lee->Cho | Mean |
|---:|---|---:|---:|---:|
| 1 | SessionEA+Feature-CORAL+MMD-resample+CSP-LDA | 70.29 | 65.66 | 67.97 |
| 2 | SessionEA+CSP-LDA | 70.74 | 65.14 | 67.94 |
| 3 | SessionEA+Feature-CORAL+CSP-LDA | 70.20 | 65.66 | 67.93 |
| 4 | SessionEA+MMD-resample+CSP-LDA | 70.68 | 65.16 | 67.92 |
| 5 | Feature-CORAL+CSP-LDA | 69.61 | 65.49 | 67.55 |
| 6 | SourceSelect-k30+CSP-LDA | 69.57 | 65.14 | 67.36 |
| 7 | RawUnified+DatasetEA+SubjectEA+CSP-LDA | 68.96 | 65.10 | 67.03 |
| 8 | raw+unified | 68.96 | 65.10 | 67.03 |

## Interpretation

- Best balanced candidate remains `SessionEA+Feature-CORAL+CSP-LDA` for Lee->Cho and near-best for Cho->Lee.
- Direction-specific best for Cho->Lee is `SessionEA+CSP-LDA`.
- MMD and source selection are close but do not clearly dominate the SessionEA+Feature-CORAL candidate.
- Gaussian OT and PhysioNet pivot underperform the EA/CORAL family in these runs.
