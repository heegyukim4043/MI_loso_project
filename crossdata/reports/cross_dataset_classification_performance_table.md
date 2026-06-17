# Cross-Dataset Classification Performance

Metrics are subject-level means. Accuracy, Precision, and F1 are percentages; kappa is Cohen kappa.

- Source summary: `/home/hkim/MI_test/results/subject_level/cross_dataset_subject_performance_all_methods_summary.csv`
- Full CSV: `/home/hkim/MI_test/cross_dataset_classification_performance_table.csv`

## Ranked Summary

| Rank | Method | Direction | N | Acc | Precision | F1 | Kappa | Avg Acc |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | SessionEA+Feature-CORAL+MMD-resample+CSP-LDA | Cho->Lee | 54 | 70.29 +/- 12.27 | 70.29 +/- 12.27 | 70.29 +/- 12.27 | 0.406 +/- 0.245 | 67.97 |
| 1 | SessionEA+Feature-CORAL+MMD-resample+CSP-LDA | Lee->Cho | 52 | 65.66 +/- 11.34 | 65.66 +/- 11.34 | 65.66 +/- 11.34 | 0.313 +/- 0.227 | 67.97 |
| 2 | SessionEA+CSP-LDA | Cho->Lee | 54 | 70.74 +/- 13.05 | 71.22 +/- 13.28 | 69.87 +/- 14.28 | 0.415 +/- 0.261 | 67.94 |
| 2 | SessionEA+CSP-LDA | Lee->Cho | 52 | 65.14 +/- 12.28 | 65.51 +/- 12.41 | 64.91 +/- 12.35 | 0.303 +/- 0.246 | 67.94 |
| 3 | SessionEA+Feature-CORAL+CSP-LDA | Cho->Lee | 54 | 70.20 +/- 12.33 | 70.23 +/- 12.35 | 70.19 +/- 12.34 | 0.404 +/- 0.247 | 67.93 |
| 3 | SessionEA+Feature-CORAL+CSP-LDA | Lee->Cho | 52 | 65.66 +/- 11.41 | 65.70 +/- 11.42 | 65.64 +/- 11.42 | 0.313 +/- 0.228 | 67.93 |
| 4 | SessionEA+MMD-resample+CSP-LDA | Cho->Lee | 54 | 70.68 +/- 12.84 | 70.68 +/- 12.84 | 70.68 +/- 12.84 | 0.414 +/- 0.257 | 67.92 |
| 4 | SessionEA+MMD-resample+CSP-LDA | Lee->Cho | 52 | 65.16 +/- 12.36 | 65.16 +/- 12.36 | 65.16 +/- 12.36 | 0.303 +/- 0.247 | 67.92 |
| 5 | Feature-CORAL+CSP-LDA | Cho->Lee | 54 | 69.61 +/- 11.69 | 69.61 +/- 11.69 | 69.61 +/- 11.69 | 0.392 +/- 0.234 | 67.55 |
| 5 | Feature-CORAL+CSP-LDA | Lee->Cho | 52 | 65.49 +/- 11.54 | 65.49 +/- 11.54 | 65.49 +/- 11.54 | 0.310 +/- 0.231 | 67.55 |
| 6 | SourceSelect-k30+CSP-LDA | Cho->Lee | 54 | 69.57 +/- 12.79 | 69.57 +/- 12.79 | 69.57 +/- 12.79 | 0.391 +/- 0.256 | 67.36 |
| 6 | SourceSelect-k30+CSP-LDA | Lee->Cho | 52 | 65.14 +/- 12.68 | 65.14 +/- 12.68 | 65.14 +/- 12.68 | 0.303 +/- 0.254 | 67.36 |
| 7 | RawUnified+DatasetEA+SubjectEA+CSP-LDA | Cho->Lee | 54 | 68.96 +/- 12.37 | 69.27 +/- 12.43 | 68.71 +/- 12.60 | 0.379 +/- 0.247 | 67.03 |
| 7 | RawUnified+DatasetEA+SubjectEA+CSP-LDA | Lee->Cho | 52 | 65.10 +/- 12.27 | 65.43 +/- 12.35 | 64.85 +/- 12.41 | 0.302 +/- 0.245 | 67.03 |
| 8 | raw+unified | Cho->Lee | 54 | 68.96 +/- 12.37 | 68.96 +/- 12.37 | 68.96 +/- 12.37 | 0.379 +/- 0.247 | 67.03 |
| 8 | raw+unified | Lee->Cho | 52 | 65.10 +/- 12.27 | 65.10 +/- 12.27 | 65.10 +/- 12.27 | 0.302 +/- 0.245 | 67.03 |
| 9 | MMD-resample+CSP-LDA | Cho->Lee | 54 | 69.00 +/- 12.28 | 69.00 +/- 12.28 | 69.00 +/- 12.28 | 0.380 +/- 0.246 | 67.00 |
| 9 | MMD-resample+CSP-LDA | Lee->Cho | 52 | 65.00 +/- 12.45 | 65.00 +/- 12.45 | 65.00 +/- 12.45 | 0.300 +/- 0.249 | 67.00 |
| 10 | SourceSelect-k20+CSP-LDA | Cho->Lee | 54 | 69.41 +/- 12.70 | 69.41 +/- 12.70 | 69.41 +/- 12.70 | 0.388 +/- 0.254 | 66.98 |
| 10 | SourceSelect-k20+CSP-LDA | Lee->Cho | 52 | 64.56 +/- 12.88 | 64.56 +/- 12.88 | 64.56 +/- 12.88 | 0.291 +/- 0.258 | 66.98 |
| 11 | ZscoreBeforeEA+DatasetEA+SubjectEA+CSP-LDA | Cho->Lee | 54 | 68.81 +/- 12.33 | 68.81 +/- 12.33 | 68.81 +/- 12.33 | 0.376 +/- 0.247 | 66.94 |
| 11 | ZscoreBeforeEA+DatasetEA+SubjectEA+CSP-LDA | Lee->Cho | 52 | 65.06 +/- 12.21 | 65.06 +/- 12.21 | 65.06 +/- 12.21 | 0.301 +/- 0.244 | 66.94 |
| 12 | SourceSelect-k40+CSP-LDA | Cho->Lee | 54 | 69.00 +/- 12.87 | 69.00 +/- 12.87 | 69.00 +/- 12.87 | 0.380 +/- 0.257 | 66.91 |
| 12 | SourceSelect-k40+CSP-LDA | Lee->Cho | 52 | 64.83 +/- 12.59 | 64.83 +/- 12.59 | 64.83 +/- 12.59 | 0.297 +/- 0.252 | 66.91 |
| 13 | WeightedDatasetEA+SubjectEA+CSP-LDA | Cho->Lee | 54 | 68.57 +/- 12.18 | 68.57 +/- 12.18 | 68.57 +/- 12.18 | 0.371 +/- 0.244 | 66.78 |
| 13 | WeightedDatasetEA+SubjectEA+CSP-LDA | Lee->Cho | 52 | 64.98 +/- 12.24 | 64.98 +/- 12.24 | 64.98 +/- 12.24 | 0.300 +/- 0.245 | 66.78 |
| 14 | DatasetEA+Zscore+SubjectEA+CSP-LDA | Cho->Lee | 54 | 68.10 +/- 12.32 | 68.10 +/- 12.32 | 68.10 +/- 12.32 | 0.362 +/- 0.246 | 66.53 |
| 14 | DatasetEA+Zscore+SubjectEA+CSP-LDA | Lee->Cho | 52 | 64.95 +/- 12.32 | 64.95 +/- 12.32 | 64.95 +/- 12.32 | 0.299 +/- 0.246 | 66.53 |
| 15 | SourceSelect-k10+CSP-LDA | Cho->Lee | 54 | 68.29 +/- 12.52 | 68.29 +/- 12.52 | 68.29 +/- 12.52 | 0.366 +/- 0.250 | 66.34 |
| 15 | SourceSelect-k10+CSP-LDA | Lee->Cho | 52 | 64.39 +/- 13.09 | 64.39 +/- 13.09 | 64.39 +/- 13.09 | 0.288 +/- 0.262 | 66.34 |
| 16 | StdMI+DatasetEA+SubjectEA+CSP-LDA | Cho->Lee | 54 | 68.10 +/- 12.38 | 68.10 +/- 12.38 | 68.10 +/- 12.38 | 0.362 +/- 0.248 | 66.03 |
| 16 | StdMI+DatasetEA+SubjectEA+CSP-LDA | Lee->Cho | 52 | 63.95 +/- 11.80 | 63.95 +/- 11.80 | 63.95 +/- 11.80 | 0.279 +/- 0.236 | 66.03 |
| 17 | FrequencySpecificEA+CSP-LDA | Cho->Lee | 54 | 67.19 +/- 11.97 | 67.19 +/- 11.97 | 67.19 +/- 11.97 | 0.344 +/- 0.239 | 64.67 |
| 17 | FrequencySpecificEA+CSP-LDA | Lee->Cho | 52 | 62.16 +/- 10.33 | 62.16 +/- 10.33 | 62.16 +/- 10.33 | 0.243 +/- 0.207 | 64.67 |
| 18 | RiemannianDatasetEA+SubjectEA+CSP-LDA | Cho->Lee | 54 | 64.25 +/- 11.91 | 64.25 +/- 11.91 | 64.25 +/- 11.91 | 0.285 +/- 0.238 | 61.57 |
| 18 | RiemannianDatasetEA+SubjectEA+CSP-LDA | Lee->Cho | 52 | 58.89 +/- 10.39 | 58.89 +/- 10.39 | 58.89 +/- 10.39 | 0.178 +/- 0.208 | 61.57 |
| 19 | psd+DatasetEA+SubjectEA+adabn+con+AdaBN_eval | Cho->Lee | 54 | 62.29 +/- 10.02 | 62.29 +/- 10.02 | 62.29 +/- 10.02 | 0.246 +/- 0.200 | 61.41 |
| 19 | psd+DatasetEA+SubjectEA+adabn+con+AdaBN_eval | Lee->Cho | 52 | 60.53 +/- 8.99 | 60.53 +/- 8.99 | 60.53 +/- 8.99 | 0.211 +/- 0.180 | 61.41 |
| 20 | DatasetEA+SubjectEA+cspnet | Cho->Lee | 54 | 62.35 +/- 9.51 | 62.35 +/- 9.51 | 62.35 +/- 9.51 | 0.247 +/- 0.190 | 61.27 |
| 20 | DatasetEA+SubjectEA+cspnet | Lee->Cho | 52 | 60.19 +/- 8.61 | 60.19 +/- 8.61 | 60.19 +/- 8.61 | 0.204 +/- 0.172 | 61.27 |
| 21 | DatasetEA+SubjectEA+adabn+con+AdaBN_eval | Cho->Lee | 54 | 62.69 +/- 9.99 | 62.69 +/- 9.99 | 62.69 +/- 9.99 | 0.254 +/- 0.200 | 61.25 |
| 21 | DatasetEA+SubjectEA+adabn+con+AdaBN_eval | Lee->Cho | 52 | 59.80 +/- 8.21 | 59.80 +/- 8.21 | 59.80 +/- 8.21 | 0.196 +/- 0.164 | 61.25 |
| 22 | StdMI+DatasetEA+SubjectEA+SourceWeightTau0.1+cspnet | Cho->Lee | 54 | 62.89 +/- 8.99 | 62.89 +/- 8.99 | 62.89 +/- 8.99 | 0.258 +/- 0.180 | 61.13 |
| 22 | StdMI+DatasetEA+SubjectEA+SourceWeightTau0.1+cspnet | Lee->Cho | 52 | 59.37 +/- 9.14 | 59.37 +/- 9.14 | 59.37 +/- 9.14 | 0.188 +/- 0.183 | 61.13 |
| 23 | psd+DatasetEA+SubjectEA+adabn+con | Cho->Lee | 54 | 62.02 +/- 9.87 | 62.02 +/- 9.87 | 62.02 +/- 9.87 | 0.240 +/- 0.197 | 61.05 |
| 23 | psd+DatasetEA+SubjectEA+adabn+con | Lee->Cho | 52 | 60.08 +/- 8.95 | 60.08 +/- 8.95 | 60.08 +/- 8.95 | 0.202 +/- 0.179 | 61.05 |
| 24 | DatasetEA+SubjectEA+adabn+con | Cho->Lee | 54 | 62.35 +/- 9.51 | 62.35 +/- 9.51 | 62.35 +/- 9.51 | 0.247 +/- 0.190 | 61.03 |
| 24 | DatasetEA+SubjectEA+adabn+con | Lee->Cho | 52 | 59.72 +/- 8.35 | 59.72 +/- 8.35 | 59.72 +/- 8.35 | 0.194 +/- 0.167 | 61.03 |
| 25 | psd+DatasetEA+SubjectEA+tent+TENT_eval | Cho->Lee | 54 | 62.29 +/- 9.98 | 62.29 +/- 9.98 | 62.29 +/- 9.98 | 0.246 +/- 0.200 | 60.90 |
| 25 | psd+DatasetEA+SubjectEA+tent+TENT_eval | Lee->Cho | 52 | 59.51 +/- 9.06 | 59.51 +/- 9.06 | 59.51 +/- 9.06 | 0.190 +/- 0.181 | 60.90 |
| 26 | psd+DatasetEA+SubjectEA+cspnet | Cho->Lee | 54 | 62.02 +/- 9.87 | 62.02 +/- 9.87 | 62.02 +/- 9.87 | 0.240 +/- 0.197 | 60.81 |
| 26 | psd+DatasetEA+SubjectEA+cspnet | Lee->Cho | 52 | 59.61 +/- 9.08 | 59.61 +/- 9.08 | 59.61 +/- 9.08 | 0.192 +/- 0.182 | 60.81 |
| 27 | StdMI+DatasetEA+SubjectEA+PseudoLabel0.80+cspnet | Cho->Lee | 54 | 60.95 +/- 9.01 | 60.95 +/- 9.01 | 60.95 +/- 9.01 | 0.219 +/- 0.180 | 60.75 |
| 27 | StdMI+DatasetEA+SubjectEA+PseudoLabel0.80+cspnet | Lee->Cho | 52 | 60.54 +/- 9.09 | 60.54 +/- 9.09 | 60.54 +/- 9.09 | 0.211 +/- 0.182 | 60.75 |
| 28 | EA+PSDNorm+cspnet | Cho->Lee | 54 | 62.32 +/- 9.46 | 62.32 +/- 9.46 | 62.32 +/- 9.46 | 0.246 +/- 0.189 | 60.72 |
| 28 | EA+PSDNorm+cspnet | Lee->Cho | 52 | 59.12 +/- 8.17 | 59.12 +/- 8.17 | 59.12 +/- 8.17 | 0.182 +/- 0.163 | 60.72 |
| 29 | StdMI+DatasetEA+SubjectEA+cspnet | Cho->Lee | 54 | 62.14 +/- 8.88 | 62.14 +/- 8.88 | 62.14 +/- 8.88 | 0.243 +/- 0.178 | 60.67 |
| 29 | StdMI+DatasetEA+SubjectEA+cspnet | Lee->Cho | 52 | 59.20 +/- 8.90 | 59.20 +/- 8.90 | 59.20 +/- 8.90 | 0.184 +/- 0.178 | 60.67 |
| 30 | psd+DatasetEA+SubjectEA+tent | Cho->Lee | 54 | 62.02 +/- 9.87 | 62.02 +/- 9.87 | 62.02 +/- 9.87 | 0.240 +/- 0.197 | 60.08 |
| 30 | psd+DatasetEA+SubjectEA+tent | Lee->Cho | 52 | 58.14 +/- 8.00 | 58.14 +/- 8.00 | 58.14 +/- 8.00 | 0.163 +/- 0.160 | 60.08 |
| 31 | StdMI+DatasetEA+SubjectEA+DSBN+con | Cho->Lee | 54 | 62.60 +/- 9.69 | 62.60 +/- 9.69 | 62.60 +/- 9.69 | 0.252 +/- 0.194 | 60.02 |
| 31 | StdMI+DatasetEA+SubjectEA+DSBN+con | Lee->Cho | 52 | 57.43 +/- 7.65 | 57.43 +/- 7.65 | 57.43 +/- 7.65 | 0.149 +/- 0.153 | 60.02 |
| 32 | StdMI+DatasetEA+SubjectEA+DSBN+cspnet | Cho->Lee | 54 | 62.60 +/- 9.69 | 62.60 +/- 9.69 | 62.60 +/- 9.69 | 0.252 +/- 0.194 | 60.00 |
| 32 | StdMI+DatasetEA+SubjectEA+DSBN+cspnet | Lee->Cho | 52 | 57.41 +/- 7.51 | 57.41 +/- 7.51 | 57.41 +/- 7.51 | 0.148 +/- 0.150 | 60.00 |
| 33 | SubjectEA+DatasetEA+cspnet | Cho->Lee | 54 | 60.81 +/- 9.74 | 60.81 +/- 9.74 | 60.81 +/- 9.74 | 0.216 +/- 0.195 | 59.66 |
| 33 | SubjectEA+DatasetEA+cspnet | Lee->Cho | 52 | 58.52 +/- 8.47 | 58.52 +/- 8.47 | 58.52 +/- 8.47 | 0.170 +/- 0.169 | 59.66 |
| 34 | StdMI+DatasetEA+SubjectEA+SourceWeightTau0.5+cspnet | Cho->Lee | 54 | 60.82 +/- 8.82 | 60.82 +/- 8.82 | 60.82 +/- 8.82 | 0.216 +/- 0.176 | 59.61 |
| 34 | StdMI+DatasetEA+SubjectEA+SourceWeightTau0.5+cspnet | Lee->Cho | 52 | 58.40 +/- 8.50 | 58.40 +/- 8.50 | 58.40 +/- 8.50 | 0.168 +/- 0.170 | 59.61 |
| 35 | StdMI+DatasetEA+SubjectEA+SourceWeightTau1.0+cspnet | Cho->Lee | 54 | 60.95 +/- 8.69 | 60.95 +/- 8.69 | 60.95 +/- 8.69 | 0.219 +/- 0.174 | 59.58 |
| 35 | StdMI+DatasetEA+SubjectEA+SourceWeightTau1.0+cspnet | Lee->Cho | 52 | 58.21 +/- 8.93 | 58.21 +/- 8.93 | 58.21 +/- 8.93 | 0.164 +/- 0.179 | 59.58 |
| 36 | StdMI+DatasetEA+SubjectEA+adabn+con+PseudoLabel0.85 | Cho->Lee | 54 | 61.85 +/- 9.34 | 61.85 +/- 9.34 | 61.85 +/- 9.34 | 0.237 +/- 0.187 | 59.38 |
| 36 | StdMI+DatasetEA+SubjectEA+adabn+con+PseudoLabel0.85 | Lee->Cho | 52 | 56.91 +/- 7.11 | 56.91 +/- 7.11 | 56.91 +/- 7.11 | 0.138 +/- 0.142 | 59.38 |
| 37 | StdMI+DatasetEA+SubjectEA+adabn+con+PseudoLabel0.85+AdaBN_eval | Cho->Lee | 54 | 62.08 +/- 9.30 | 62.08 +/- 9.30 | 62.08 +/- 9.30 | 0.242 +/- 0.186 | 59.29 |
| 37 | StdMI+DatasetEA+SubjectEA+adabn+con+PseudoLabel0.85+AdaBN_eval | Lee->Cho | 52 | 56.50 +/- 7.20 | 56.50 +/- 7.20 | 56.50 +/- 7.20 | 0.130 +/- 0.144 | 59.29 |
| 38 | StdMI+DatasetEA+SubjectEA+PseudoLabel0.85+cspnet | Cho->Lee | 54 | 61.85 +/- 9.34 | 61.85 +/- 9.34 | 61.85 +/- 9.34 | 0.237 +/- 0.187 | 59.09 |
| 38 | StdMI+DatasetEA+SubjectEA+PseudoLabel0.85+cspnet | Lee->Cho | 52 | 56.32 +/- 7.29 | 56.32 +/- 7.29 | 56.32 +/- 7.29 | 0.126 +/- 0.146 | 59.09 |
| 39 | EA+AdaBN+Con+AdaBN_eval | Cho->Lee | 54 | 59.80 +/- 8.37 | 59.80 +/- 8.37 | 59.80 +/- 8.37 | 0.196 +/- 0.167 | 58.49 |
| 39 | EA+AdaBN+Con+AdaBN_eval | Lee->Cho | 52 | 57.19 +/- 7.58 | 57.19 +/- 7.58 | 57.19 +/- 7.58 | 0.144 +/- 0.152 | 58.49 |
| 40 | EA+TENT+TENT_eval | Cho->Lee | 54 | 59.80 +/- 8.14 | 59.80 +/- 8.14 | 59.80 +/- 8.14 | 0.196 +/- 0.163 | 57.94 |
| 40 | EA+TENT+TENT_eval | Lee->Cho | 52 | 56.09 +/- 6.66 | 56.09 +/- 6.66 | 56.09 +/- 6.66 | 0.122 +/- 0.133 | 57.94 |
| 41 | GaussianOT+SubjectEA+CSP-LDA | Cho->Lee | 54 | 54.71 +/- 7.03 | 54.71 +/- 7.03 | 54.71 +/- 7.03 | 0.094 +/- 0.141 | 57.71 |
| 41 | GaussianOT+SubjectEA+CSP-LDA | Lee->Cho | 52 | 60.70 +/- 10.25 | 60.70 +/- 10.25 | 60.70 +/- 10.25 | 0.214 +/- 0.205 | 57.71 |
| 42 | PhysioNetPivot+CSP-LDA | Cho->Lee | 54 | 55.03 +/- 7.36 | 55.03 +/- 7.36 | 55.03 +/- 7.36 | 0.101 +/- 0.147 | 56.91 |
| 42 | PhysioNetPivot+CSP-LDA | Lee->Cho | 52 | 58.79 +/- 10.19 | 58.79 +/- 10.19 | 58.79 +/- 10.19 | 0.176 +/- 0.204 | 56.91 |
| 43 | GaussianOT+CSP-LDA | Cho->Lee | 54 | 54.85 +/- 7.33 | 54.85 +/- 7.33 | 54.85 +/- 7.33 | 0.097 +/- 0.147 | 56.73 |
| 43 | GaussianOT+CSP-LDA | Lee->Cho | 52 | 58.61 +/- 10.06 | 58.61 +/- 10.06 | 58.61 +/- 10.06 | 0.172 +/- 0.201 | 56.73 |
| 44 | StdMI+DatasetEA+SubjectEA+PseudoLabel0.90+cspnet | Cho->Lee | 54 | 61.53 +/- 9.47 | 61.53 +/- 9.47 | 61.53 +/- 9.47 | 0.231 +/- 0.189 | 56.51 |
| 44 | StdMI+DatasetEA+SubjectEA+PseudoLabel0.90+cspnet | Lee->Cho | 52 | 51.49 +/- 2.90 | 51.49 +/- 2.90 | 51.49 +/- 2.90 | 0.030 +/- 0.058 | 56.51 |
| 45 | StdMI+EA+CSP-LDA | Cho->Lee | 54 | 52.87 +/- 5.13 | 52.87 +/- 5.13 | 52.87 +/- 5.13 | 0.057 +/- 0.103 | 56.05 |
| 45 | StdMI+EA+CSP-LDA | Lee->Cho | 52 | 59.23 +/- 10.11 | 59.23 +/- 10.11 | 59.23 +/- 10.11 | 0.185 +/- 0.202 | 56.05 |
| 46 | EA+AdaBN+Con | Cho->Lee | 54 | 53.69 +/- 6.13 | 53.69 +/- 6.13 | 53.69 +/- 6.13 | 0.074 +/- 0.123 | 53.97 |
| 46 | EA+AdaBN+Con | Lee->Cho | 52 | 54.26 +/- 5.82 | 54.26 +/- 5.82 | 54.26 +/- 5.82 | 0.085 +/- 0.116 | 53.97 |
| 47 | StdMI+CSP-LDA | Cho->Lee | 54 | 55.15 +/- 6.73 | 55.15 +/- 6.73 | 55.15 +/- 6.73 | 0.103 +/- 0.135 | 53.95 |
| 47 | StdMI+CSP-LDA | Lee->Cho | 52 | 52.76 +/- 4.92 | 52.76 +/- 4.92 | 52.76 +/- 4.92 | 0.055 +/- 0.098 | 53.95 |
| 48 | EA+TENT | Cho->Lee | 54 | 53.69 +/- 6.13 | 53.69 +/- 6.13 | 53.69 +/- 6.13 | 0.074 +/- 0.123 | 53.39 |
| 48 | EA+TENT | Lee->Cho | 52 | 53.09 +/- 4.04 | 53.09 +/- 4.04 | 53.09 +/- 4.04 | 0.062 +/- 0.081 | 53.39 |
| 49 | EA+CSPNet | Cho->Lee | 54 | 53.69 +/- 6.13 | 53.69 +/- 6.13 | 53.69 +/- 6.13 | 0.074 +/- 0.123 | 53.14 |
| 49 | EA+CSPNet | Lee->Cho | 52 | 52.60 +/- 4.08 | 52.60 +/- 4.08 | 52.60 +/- 4.08 | 0.052 +/- 0.081 | 53.14 |
| 50 | CSPNet baseline | Cho->Lee | 54 | 50.02 +/- 0.14 | 50.02 +/- 0.14 | 50.02 +/- 0.14 | 0.000 +/- 0.003 | 51.36 |
| 50 | CSPNet baseline | Lee->Cho | 52 | 52.70 +/- 4.25 | 52.70 +/- 4.25 | 52.70 +/- 4.25 | 0.054 +/- 0.085 | 51.36 |
| 51 | contrastive | Cho->Lee | 54 | 50.02 +/- 0.14 | 50.02 +/- 0.14 | 50.02 +/- 0.14 | 0.000 +/- 0.003 | 51.36 |
| 51 | contrastive | Lee->Cho | 52 | 52.70 +/- 4.25 | 52.70 +/- 4.25 | 52.70 +/- 4.25 | 0.054 +/- 0.085 | 51.36 |
| 52 | cspnet | Cho->Lee | 54 | 50.02 +/- 0.14 | 50.02 +/- 0.14 | 50.02 +/- 0.14 | 0.000 +/- 0.003 | 51.36 |
| 52 | cspnet | Lee->Cho | 52 | 52.70 +/- 4.25 | 52.70 +/- 4.25 | 52.70 +/- 4.25 | 0.054 +/- 0.085 | 51.36 |
