# Analysis Outputs

This directory contains post-hoc analyses for LOSO and cross-dataset MI generalization.

## Key Reports

- `generalization_validation_sequence/generalization_validation_sequence_report.md`
  - Separability tertiles, recoverable-vs-persistent classifier, oracle gap, harm analysis, and statistical tests.
- `additional_generalization_tests/additional_generalization_tests_report.md`
  - Bootstrap CI for Cross-vs-LOSO best, McNemar coverage tests, DatasetEA Riemannian pre/post gap, and source-pool scaling.
- `eeg_markers_generalization/eeg_marker_generalization_report.md`
  - Mu/beta task-window lateralization markers and their relation to generalization outcomes.
- `class_feature_similarity/class_feature_similarity_report.md`
  - Class covariance and CSP feature separability analyses.
- `illiteracy_generalization/illiteracy_generalization_consistency_report.md`
  - LOSO recoverable vs persistent BCI-illiteracy analysis.
- `combined_method_illiteracy/combined_method_illiteracy_report.md`
  - CSPNet combined-method recovery analysis.
- `crossdataset_illiteracy/crossdataset_illiteracy_report.md`
  - Cross-dataset recovery and persistent failure analysis.
- `transfer_benefit/transfer_benefit_predictor_report.md`
  - Covariance-geometry transfer benefit predictors.
- `generalization_methods_separate/generalization_metrics_by_separate_method.md`
  - Original, EA, TENT, AdaBN, and Snapshot compared as separate final methods.

## Notes

- Raw MOABB NPZ files, downloaded dataset caches, virtual environments, and generated topomap images are intentionally excluded.
- CSV files in each subdirectory contain the underlying subject-level tables used by the reports.
- Scripts in this directory reproduce the post-hoc aggregations from the saved result CSVs and streamed NPZ preprocessing outputs.
