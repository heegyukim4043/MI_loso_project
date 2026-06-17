# MI Test Progress

Last updated: `2026-06-05 09:59` KST

## Current Status

- Active training/cross-dataset processes: `none`
- GPU0/1/2: `idle` (`1 MiB`, `0%` utilization each)
- Stale monitor process: PID `22016` only; this is not a training job.
- Requested baselines completed: `CSP-LDA`, `EA-CSP-LDA`, `DatasetEA + SubjectEA + CSP-LDA`
- Cross-dataset correction queue completed through pseudo-label and source-weighting sweeps.
- Raw preprocessing note: current cross-dataset runs still use existing preprocessed `.npz`; raw-level resample/bandpass/channel-order regeneration remains unresolved.

## Final Method Selection

- Main cross-dataset method: `StdMI DatasetEA + SubjectEA + CSP-LDA`
- Reason: current best in both directions.
- Cho2017 -> Lee2019: `68.10%`, kappa `0.362`
- Lee2019 -> Cho2017: `63.95%`, kappa `0.279`
- Interpretation: the biggest cross-dataset gain is alignment plus classical CSP features, not the deep CSPNet head.
- Neural-only representative: `DatasetEA + SubjectEA + CSPNet` (`62.35% / 60.19%`) as the simple aligned neural baseline.

## Recommended Main Cross-Dataset Table

| Method | Cho2017 -> Lee2019 | Lee2019 -> Cho2017 | Purpose |
|---|---:|---:|---|
| CSP-LDA | 55.15 / k=0.103 | 52.76 / k=0.055 | classical baseline |
| EA-CSP-LDA | 52.87 / k=0.057 | 59.23 / k=0.185 | EA classical baseline |
| DatasetEA + SubjectEA + CSP-LDA | **68.10 / k=0.362** | **63.95 / k=0.279** | current main method |
| CSPNet baseline | 50.02 / k=0.000 | 52.70 / k=0.054 | neural baseline |
| EA + CSPNet | 53.69 / k=0.074 | 52.60 / k=0.052 | simple EA effect |
| EA + PSDNorm + CSPNet | 62.32 / k=0.246 | 59.12 / k=0.182 | power normalization effect |
| DatasetEA + SubjectEA + CSPNet | 62.35 / k=0.247 | 60.19 / k=0.204 | simple neural alignment baseline |
| DatasetEA + SubjectEA + AdaBN + Con | 62.69 / k=0.254 | 59.80 / k=0.196 | Cho->Lee neural best |
| PSDNorm + DatasetEA + SubjectEA + AdaBN + Con | 62.29 / k=0.246 | 60.53 / k=0.211 | Lee->Cho neural best |
| SubjectEA + DatasetEA + CSPNet | 60.81 / k=0.216 | 58.52 / k=0.170 | EA order ablation |

## Additional Correction Results

| Method | Cho2017 -> Lee2019 | Lee2019 -> Cho2017 | Note |
|---|---:|---:|---|
| StdMI DatasetEA + SubjectEA + CSPNet | 62.14 / k=0.243 | 59.20 / k=0.184 | standard MI channel neural baseline |
| StdMI DatasetEA + SubjectEA + DSBN + CSPNet | 62.60 / k=0.252 | 57.41 / k=0.148 | DSBN did not help Lee->Cho |
| StdMI DatasetEA + SubjectEA + DSBN + Con | 62.60 / k=0.252 | 57.43 / k=0.149 | DSBN + contrastive also not useful |
| StdMI DatasetEA + SubjectEA + CSPNet + PseudoLabel(0.80) | 60.95 / k=0.219 | 60.54 / k=0.211 | best pseudo for Lee->Cho, still below CSP-LDA |
| StdMI DatasetEA + SubjectEA + CSPNet + PseudoLabel(0.85) | 61.85 / k=0.237 | 56.32 / k=0.126 | pseudo threshold sweep |
| StdMI DatasetEA + SubjectEA + CSPNet + PseudoLabel(0.90) | 61.53 / k=0.231 | 51.49 / k=0.030 | too strict/noisy outcome |
| StdMI DatasetEA + SubjectEA + AdaBN + Con + PseudoLabel(0.85) | 62.08 / k=0.242 | 56.50 / k=0.130 | did not beat neural baseline |
| StdMI DatasetEA + SubjectEA + SourceWeight(tau=0.1) | 62.89 / k=0.258 | 59.37 / k=0.188 | best source weighting |
| StdMI DatasetEA + SubjectEA + SourceWeight(tau=0.5) | 60.82 / k=0.216 | 58.40 / k=0.168 | source weighting sweep |
| StdMI DatasetEA + SubjectEA + SourceWeight(tau=1.0) | 60.95 / k=0.219 | 58.21 / k=0.164 | source weighting sweep |

## Required Cross-Dataset Checklist

| Priority | Required item | Reason | Status |
|---|---|---|---|
| Very high | CSP-LDA cross-dataset baseline | Classical baseline needed so CSPNet is not the only reference | completed |
| Very high | EA-CSP-LDA cross-dataset baseline | Direct comparison to classical EA pipeline | completed |
| Very high | DatasetEA + SubjectEA + CSP-LDA | Tests whether alignment gain is model-independent | completed, current best |
| Very high | Final representative method selection | Direction-specific bests differed before; method must be fixed | selected: `DatasetEA+SubjectEA+CSP-LDA` |
| High | Target subject distribution analysis | Needed to show which target subjects improve, not only mean accuracy | completed CSV |
| High | EA order ablation table | Supports `DatasetEA -> SubjectEA` design | completed, needs paper table |
| High | Common channel vs standard MI channel comparison | Explains sensitivity to channel subset/order | partially completed |
| Medium | Pseudo-label results | Additional improvement check | completed, not better than aligned CSP-LDA |
| Medium | Source weighting results | Source subject selection effect | completed, not better than aligned CSP-LDA |
| Medium | Raw preprocessing unification | Reviewer may question preprocessing mismatch | needed |
| Low | SSCL-CSD full reproduction | Numeric competition, high implementation cost | optional |

## Target Subject Distribution Analysis

Generated files:

- `/home/hkim/MI_test/results/cross_dataset_subject_analysis_20260604.csv`
- `/home/hkim/MI_test/results/cross_dataset_subject_analysis_summary_20260604.csv`

Key summary:

| Direction | Method | Mean Acc | Subjects >=60% | Subjects >=70% |
|---|---|---:|---:|---:|
| Cho2017 -> Lee2019 | CSP-LDA | 55.15 | 10/54 | 3/54 |
| Cho2017 -> Lee2019 | DatasetEA+SubjectEA+CSP-LDA | 68.10 | 40/54 | 23/54 |
| Cho2017 -> Lee2019 | DatasetEA+SubjectEA+CSPNet | 62.35 | 31/54 | 12/54 |
| Lee2019 -> Cho2017 | CSP-LDA | 52.76 | 4/52 | 1/52 |
| Lee2019 -> Cho2017 | DatasetEA+SubjectEA+CSP-LDA | 63.95 | 32/52 | 11/52 |
| Lee2019 -> Cho2017 | DatasetEA+SubjectEA+CSPNet | 60.19 | 25/52 | 6/52 |

Improvement counts:

- Cho2017 -> Lee2019: `DatasetEA+SubjectEA+CSP-LDA` improves over `CSP-LDA` on `50/54` target subjects, mean delta `+12.95%p`.
- Cho2017 -> Lee2019: it improves over `DatasetEA+SubjectEA+CSPNet` on `45/54` subjects, mean delta `+5.75%p`.
- Lee2019 -> Cho2017: it improves over `CSP-LDA` on `48/52` target subjects, mean delta `+11.19%p`.
- Lee2019 -> Cho2017: it improves over `DatasetEA+SubjectEA+CSPNet` on `37/52` subjects, mean delta `+3.76%p`.

## Next Open Work

1. Raw preprocessing unification: raw data loading -> resample target Hz -> standard MI channel order -> 8-30 Hz bandpass -> epoch extraction -> rerun selected methods.
2. Paper table cleanup: main table, EA order ablation, subject-level distribution table.
3. Optional: SSCL-CSD reproduction only if numeric competition is needed.


## Raw-Unified Cross-Dataset Verification Started (2026-06-05 10:11)

- Status: `running`
- Service: `mi-raw-unified-verification-20260605b.service`
- Manager PID observed: `329331`
- Preprocess PID observed: `329332`
- Goal: regenerate raw-level unified `.npz` with resample `128 Hz`, fixed standard MI channel order, bandpass `8-30 Hz`, epoch `[0.5, 2.5] s`, then rerun `DatasetEA + SubjectEA + CSP-LDA` both directions.
- Preprocess log: `/home/hkim/MI_test/results/runs/raw_unified_verification_20260605_raw_unified_preprocess.log`
- Verification log after preprocessing: `/home/hkim/MI_test/results/runs/raw_unified_verification_20260605_raw_unified_csp_lda.log`


## Raw-Unified Cross-Dataset Verification (2026-06-05 15:40)

- Status: `completed`
- Preprocessed dir: `/home/hkim/MI_test/preprocessed_raw_unified`
- Pipeline: raw load -> fixed standard MI channel order -> ICA -> resample `128 Hz` -> bandpass `8-30 Hz` -> epoch `[0.5, 2.5] s`
- Evaluation: `DatasetEA + SubjectEA + CSP-LDA`, `MI_N_TIMES=257`
- Preprocess log: `/home/hkim/MI_test/results/runs/raw_unified_verification_20260605_raw_unified_preprocess.log`
- Verification log: `/home/hkim/MI_test/results/runs/raw_unified_verification_20260605_raw_unified_csp_lda.log`
- Cho2017 -> Lee2019: `68.96%`, kappa `0.379`, subjects `54`
- Lee2019 -> Cho2017: `65.10%`, kappa `0.302`, subjects `52`
- CSV Cho->Lee: `/home/hkim/MI_test/results/loso_results_20260605_raw_unified_cross_cho2017_to_lee2019_csp_lda.csv`
- CSV Lee->Cho: `/home/hkim/MI_test/results/loso_results_20260605_raw_unified_cross_lee2019_to_cho2017_csp_lda.csv`


## Cross-Dataset Statistical/Figure Artifacts

- Status: `completed`
- Statistical tests: `/home/hkim/MI_test/results/tables/cross_dataset_and_loso_stat_tests.csv`
- Statistical test table: `/home/hkim/MI_test/results/tables/statistical_tests.md`
- EA order ablation table: `/home/hkim/MI_test/results/tables/ea_order_ablation_summary.md`
- Subject-level figure: `/home/hkim/MI_test/results/figures/cross_subject_distribution_cho2017_to_lee2019.png`
- Subject-level figure: `/home/hkim/MI_test/results/figures/cross_subject_distribution_lee2019_to_cho2017.png`
- Subject-level figure: `/home/hkim/MI_test/results/figures/loso_subject_distribution_cho2017.png`
- Subject-level figure: `/home/hkim/MI_test/results/figures/loso_subject_distribution_lee2019.png`
- t-SNE figure: `/home/hkim/MI_test/results/figures/tsne_raw_unified_ea_before_after.png`


## Cross-Dataset Alignment Variant Sweep (2026-06-08 09:11)

- Status: `completed`
- Summary CSV: `/home/hkim/MI_test/results/cross_dataset_alignment_variants_summary_20260608_alignment_variants.csv`

| Variant | Cho2017 -> Lee2019 | Lee2019 -> Cho2017 |
|---|---:|---:|
| riemannian_dataset_subject | 64.25% / k=0.285 | 58.89% / k=0.178 |


## Additional Cross-Dataset Artifacts And Variant Checks (2026-06-08 09:20)

- Status: `completed`
- Statistical tests added for cross-dataset and LOSO comparisons.
- Subject-level distribution figures generated for LOSO and cross-dataset.
- EA order ablation table generated.
- t-SNE before/after EA visualization generated.
- BCI IV 2a EA generalization check completed without KMM comparison.

### Generated Tables/Figures

- Statistical tests CSV: `/home/hkim/MI_test/results/tables/cross_dataset_and_loso_stat_tests.csv`
- Statistical tests table: `/home/hkim/MI_test/results/tables/statistical_tests.md`
- EA order ablation table: `/home/hkim/MI_test/results/tables/ea_order_ablation_summary.md`
- Cross subject distribution Cho->Lee: `/home/hkim/MI_test/results/figures/cross_subject_distribution_cho2017_to_lee2019.png`
- Cross subject distribution Lee->Cho: `/home/hkim/MI_test/results/figures/cross_subject_distribution_lee2019_to_cho2017.png`
- LOSO distribution Cho2017: `/home/hkim/MI_test/results/figures/loso_subject_distribution_cho2017.png`
- LOSO distribution Lee2019: `/home/hkim/MI_test/results/figures/loso_subject_distribution_lee2019.png`
- t-SNE EA before/after: `/home/hkim/MI_test/results/figures/tsne_raw_unified_ea_before_after.png`

### Cross-Dataset Alignment Variant Sweep

Current raw-unified best remains `DatasetEA + SubjectEA + CSP-LDA`:

- Cho2017 -> Lee2019: `68.96%`, kappa `0.379`
- Lee2019 -> Cho2017: `65.10%`, kappa `0.302`

| Variant | Cho2017 -> Lee2019 | Lee2019 -> Cho2017 | Decision |
|---|---:|---:|---|
| Z-score -> DatasetEA -> SubjectEA | 68.81 / k=0.376 | 65.06 / k=0.301 | close, not better |
| DatasetEA -> Z-score -> SubjectEA | 68.10 / k=0.362 | 64.95 / k=0.299 | not better |
| Weighted DatasetEA -> SubjectEA | 68.57 / k=0.371 | 64.98 / k=0.300 | not better |
| Log-Euclidean Riemannian Dataset/Subject Alignment | 64.25 / k=0.285 | 58.89 / k=0.178 | worse; exclude from main |

Summary CSV: `/home/hkim/MI_test/results/cross_dataset_alignment_variants_summary_20260608_alignment_variants_all.csv`

### BCI IV 2a EA Generalization

- Dataset: `BNCI2014-001 / BCI Competition IV 2a`, binary left/right MI
- Method: `EA + CSPNet`, LOSO
- Subjects: `9`
- Accuracy: `73.96% ± 10.25%`
- Kappa: `0.479`
- Result CSV: `/home/hkim/MI_test/results/loso_results_ea_cspnet_bciciv2a_cspnet.csv`

### Interpretation

- Cross-dataset improvement over CSP-LDA is statistically strong in both directions.
- Raw-unified `DatasetEA + SubjectEA + CSP-LDA` remains the final main method.
- Z-score and Weighted DatasetEA are useful negative/near-miss ablations, but do not improve the final result.
- Riemannian/log-Euclidean alignment under this CSP-LDA setup hurts performance and should not be main.
- OT alignment still requires the `POT` package (`ot`) and was not run in this pass.

## Cross-Dataset Extra Variant Sweep (2026-06-08 09:40)

- Status: `completed`
- Summary CSV: `/home/hkim/MI_test/results/cross_dataset_extra_variants_summary_20260608_extra_variants.csv`

| Variant | Cho2017 -> Lee2019 | Lee2019 -> Cho2017 |
|---|---:|---:|
| gaussian_ot_source_to_target | 54.85% / k=0.097 | 58.61% / k=0.172 |
| gaussian_ot_then_subject_ea | 54.71% / k=0.094 | 60.70% / k=0.214 |
| frequency_specific_ea | 67.19% / k=0.344 | 62.16% / k=0.243 |
| physionet_pivot_dataset | 55.03% / k=0.101 | 58.79% / k=0.176 |

## Cross-Dataset SessionEA/CORAL/MMD Sweep (2026-06-08 10:07)

- Status: `completed`
- Summary CSV: `/home/hkim/MI_test/results/cross_dataset_session_coral_mmd_summary_20260608_session_coral_mmd.csv`

| Variant | Cho2017 -> Lee2019 | Lee2019 -> Cho2017 |
|---|---:|---:|
| session_ea | 70.74% / k=0.415 | 65.13% / k=0.303 |
| feature_coral | 69.61% / k=0.392 | 65.49% / k=0.310 |
| mmd_resample | 69.00% / k=0.380 | 65.00% / k=0.300 |
| session_ea_feature_coral | 70.20% / k=0.404 | 65.66% / k=0.313 |

## Cross-Dataset SessionEA/CORAL/MMD Sweep (2026-06-08 10:37)

- Status: `completed`
- Summary CSV: `/home/hkim/MI_test/results/cross_dataset_session_coral_mmd_summary_20260608_session_mmd_sourceselect.csv`

| Variant | Cho2017 -> Lee2019 | Lee2019 -> Cho2017 |
|---|---:|---:|
| session_ea_mmd_resample | 70.68% / k=0.414 | 65.16% / k=0.303 |
| session_ea_feature_coral_mmd_resample | 70.29% / k=0.406 | 65.66% / k=0.313 |
| source_select_k10 | 68.29% / k=0.366 | 64.39% / k=0.288 |
| source_select_k20 | 69.41% / k=0.388 | 64.56% / k=0.291 |
| source_select_k30 | 69.57% / k=0.391 | 65.14% / k=0.303 |
| source_select_k40 | 69.00% / k=0.380 | 64.83% / k=0.297 |

## Cross-Dataset SessionEA/CORAL/MMD Sweep (2026-06-08 14:16)

- Status: `completed`
- Summary CSV: `/home/hkim/MI_test/results/cross_dataset_session_coral_mmd_summary_20260608_metrics_final.csv`

| Variant | Cho2017 -> Lee2019 | Lee2019 -> Cho2017 |
|---|---:|---:|
| base | 68.96% / k=0.379 | 65.10% / k=0.302 |
| session_ea | 70.74% / k=0.415 | 65.13% / k=0.303 |
| session_ea_feature_coral | 70.20% / k=0.404 | 65.66% / k=0.313 |


## LOSO KMM-TrAdaBoost Started (2026-06-10)

- Status: `running`
- Unit: `mi-loso-kmm-tradaboost-20260610b.service`
- Log: `/home/hkim/MI_test/results/runs/loso_kmm_tradaboost_20260610.log`
- Script: `/home/hkim/MI_test/MI_loso_project/loso_kmm_tradaboost.py`
- Run id: `20260610_kmm_tradaboost`
- Scope: Cho2017 and Lee2019 LOSO, both `KMM-TrAdaBoost-LOSO` and `EA+KMM-TrAdaBoost-LOSO`
- Metrics saved: `acc`, `precision`, `f1`, `bac`, `kappa`
- Output pattern: `/home/hkim/MI_test/results/loso_results_*kmm_tradaboost*20260610_kmm_tradaboost.csv`
- Note: leakage-free LOSO adaptation; target labels are not used. KMM uses unlabeled target CSP features for source weighting, then AdaBoost is trained on weighted source labels.
- Last updated: `2026-06-10 19:52` KST

## LOSO Validation Queue Started (2026-06-11)

- Status: `running`
- Existing subject-level results added to `/home/hkim/MI_test/subject_performance.md`:
  - `DANN-v1 / CSPNetDANN` from `/home/hkim/MI_test/results/archive/loso_results_20260423_cspnetdann_cspnetdann.csv`
  - `DANN-v2 / CSPNetDANN` from `/home/hkim/MI_test/results/archive/loso_results_dann_v2_cspnetdann.csv`
  - `EA+CSPNet+TENT` from `/home/hkim/MI_test/results/loso_results_ea_tent_cspnet.csv`
  - `KMM-TrAdaBoost` was already appended after completion.
- New run 1: `DANN validation`
  - Unit: `mi-loso-dann-validation-20260611c.service`
  - GPU: `CUDA_VISIBLE_DEVICES=0`
  - Command model: `cspnetdann`
  - Result CSV: `/home/hkim/MI_test/results/loso_results_dann_validation_20260611_cspnetdann.csv`
  - Log: `/home/hkim/MI_test/results/runs/loso_dann_validation_20260611.log`
- New run 2: `SSCL-CSD-style validation`
  - Unit: `mi-loso-sscl-csd-style-20260611c.service`
  - GPU: `CUDA_VISIBLE_DEVICES=1`
  - Command model: `cspnetcontrastive --ea --augment --lambda_con 0.5 --con_temperature 0.07`
  - Result CSV: `/home/hkim/MI_test/results/loso_results_sscl_csd_style_20260611_cspnetcontrastive_aug.csv`
  - Log: `/home/hkim/MI_test/results/runs/loso_sscl_csd_style_20260611.log`
  - Note: current implementation is SSCL-CSD-style using existing EA+SupCon+CSPNetContrastive/augmentation, not a line-by-line reproduction of the original unpublished code.
- Existing run still active: `conformer_noea` on GPU 2.
- Last updated: `2026-06-11 17:47` KST

## Cross-Dataset srate-100Hz Queue Started (2026-06-12)

- Status: `running`
- Unit: `mi-cross-dataset-sfreq100-20260612.service`
- GPU: `CUDA_VISIBLE_DEVICES=2`
- Preprocessed dir: `/home/hkim/MI_test/preprocessed_sfreq100`
- Summary: `/home/hkim/MI_test/cross_dataset_sfreq100_results_summary.md`
- Queue log: `/home/hkim/MI_test/results/runs/cross_dataset_sfreq100_queue_20260612.log`
- Current first run: `EA+CSPNet` (`20260612_sfreq100_ea_cspnet`)
- Priority order: `EA+CSPNet`, `EA+AdaBN+Con`, `EA+Snapx6` (not executable yet), `EA+AdaBN`, `EA+TENT`, `SubjClust tau=5 analogue`
- Note: Cho2017 was downsampled to 100Hz/T=201; Lee2019 is 100Hz/T=201. `EA+Snapx6` requires adding snapshot ensemble inference to `cross_dataset.py`.
- Last updated: `2026-06-12 21:22` KST

## Cross-Dataset srate-100Hz Follow-up Completed (2026-06-13)

- Status: `completed`
- Completed prerequisite: `DatasetEA+SubjectEA+SourceWeight tau=5` = Cho->Lee 71.93% / k=0.439, Lee->Cho 69.12% / k=0.382
- EA+AdaBN+Snapx6 sfreq100:
  - Base: Cho->Lee 59.91% / k=0.198, Lee->Cho 61.23% / k=0.225
  - AdaBN eval: Cho->Lee 66.60% / k=0.332, Lee->Cho 64.74% / k=0.295
  - Snapshot eval: Cho->Lee 63.81% / k=0.276, Lee->Cho 61.36% / k=0.227
  - Snapshot+AdaBN eval: Cho->Lee 66.41% / k=0.328, Lee->Cho 63.48% / k=0.270
  - CSVs: `/home/hkim/MI_test/results/loso_results_20260613_sfreq100_ea_adabn_snapx6_cross_*_cspnet.csv`
- SessionEA/CORAL/MMD sfreq100 CSP-LDA:
  - session_ea: Cho->Lee 70.60% / k=0.412, Lee->Cho 65.40% / k=0.308
  - session_ea_feature_coral: Cho->Lee 70.02% / k=0.400, Lee->Cho 65.56% / k=0.311
  - session_ea_feature_coral_mmd_resample: Cho->Lee 70.14% / k=0.403, Lee->Cho 65.60% / k=0.312
  - Summary CSV: `/home/hkim/MI_test/results/cross_dataset_session_coral_mmd_summary_20260613_sfreq100_sessionea.csv`
- Current interpretation: sfreq100 helps DL strongly only when combined with DatasetEA+SubjectEA+SourceWeight; plain EA+Snapx6 does not beat sfreq100 SourceWeight or CSP-LDA SessionEA.
- Last updated: `2026-06-15` KST

## Cross-Dataset srate-100Hz SessionEA+SourceWeight Started (2026-06-15)

- Status: `running`
- Unit: `mi-cross-dataset-sfreq100-sessionea-sourceweight-20260615.service`
- GPU: `CUDA_VISIBLE_DEVICES=2`
- Command: `cross_dataset.py --both --model cspnet --dataset_ea --ea --session_ea --source_weighting --source_weight_tau 5.0`
- Run id: `20260615_sfreq100_datasetea_subjectea_sessionea_sourceweight_tau5`
- Log: `/home/hkim/MI_test/results/runs/cross_dataset_sfreq100_sessionea_sourceweight_tau5_20260615.log`
- Baseline to beat: `sfreq100 DatasetEA+SubjectEA+SourceWeight tau=5` = Cho->Lee 71.93% / k=0.439, Lee->Cho 69.12% / k=0.382
- Last updated: `2026-06-15` KST

## Cross-Dataset Architecture Baselines Started (2026-06-15)

- Status: `running`
- Purpose: fill missing EEGNet and Conformer cross-dataset architecture cells.
- Unit: `mi-cross-dataset-arch-baselines-20260615.service`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- Input: `/home/hkim/MI_test/preprocessed_sfreq100`, `MI_N_TIMES=201`
- Run 1: `cross_dataset.py --both --model eegnet --run_id 20260615_sfreq100_eegnet_cross_archbaseline`
- Run 2: `cross_dataset.py --both --model conformer --run_id 20260615_sfreq100_conformer_cross_archbaseline`
- Logs: `/home/hkim/MI_test/results/runs/cross_dataset_sfreq100_eegnet_archbaseline_20260615.log`, `/home/hkim/MI_test/results/runs/cross_dataset_sfreq100_conformer_archbaseline_20260615.log`
- Last updated: `2026-06-15` KST

## Cross-Dataset Architecture Method Cells Queue Started (2026-06-15)

- Status: `waiting/running`
- Summary: `/home/hkim/MI_test/cross_dataset_arch_methods_20260615.md`
- Already completed from existing CSVs:
  - `CSPNet + EA+Snapshot` = Cho->Lee 63.81% / k=0.276, Lee->Cho 61.36% / k=0.227
  - `CSPNet + EA+TENT` = Cho->Lee 66.59% / k=0.332, Lee->Cho 64.13% / k=0.283
- Missing cells queued after current GPU0 architecture baseline:
  - `EEGNet + DSA+SEA`
  - `EEGNet + EA+TENT`
  - `Conformer + DSA+SEA`
  - `Conformer + EA+TENT`
- Unit: `mi-cross-dataset-arch-methods-20260615.service`
- Queue log: `/home/hkim/MI_test/results/runs/cross_dataset_arch_methods_queue_20260615.log`
- Last updated: `2026-06-15 08:20` KST

## Current Compute Status (2026-06-16)

- Cross-dataset status: `completed`
  - Architecture method cells completed: `/home/hkim/MI_test/cross_dataset_arch_methods_20260615.md`
  - Best current cross-dataset DL: `sfreq100 + DatasetEA + SubjectEA + SessionEA + SourceWeight tau=5`
    - Cho->Lee 73.24% / k=0.465
    - Lee->Cho 69.21% / k=0.384
  - Previous SourceWeight baseline: Cho->Lee 71.93% / k=0.439, Lee->Cho 69.12% / k=0.382
  - SessionEA improved Cho->Lee by +1.31%p and Lee->Cho by +0.09%p.
- Remaining running jobs are LOSO only:
  - `EEGNet EA+Snapshot` on GPU1: `python train_loso.py --model eegnet --ea --snapshot_ensemble --snapshot_T0 50 --dataset both --run_id eegnet_ea_snapshot`
  - `Conformer EA+AdaBN` on GPU2: `python train_loso.py --model conformer --ea --adabn --dataset both --run_id conformer_ea_adabn`
  - Conformer EA+Snapshot is queued by shell command after Conformer EA+AdaBN finishes.
- GPU status at check: GPU0 idle, GPU1/2 active.
- Last updated: `2026-06-16` KST

## Cross-Dataset SourceWeight Backbone Queue Started (2026-06-16)

- Status: `running`
- Unit: `mi-cross-dataset-sourceweight-backbones-20260616.service`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- Summary: `/home/hkim/MI_test/cross_dataset_sourceweight_backbones_20260616.md`
- Queue log: `/home/hkim/MI_test/results/runs/cross_dataset_sourceweight_backbones_queue_20260616.log`
- Running first: `EEGNet + DSA+SEA+SourceWeight tau=5`
- Next: `Conformer + DSA+SEA+SourceWeight tau=5`
- Baselines included in summary:
  - `CSPNet + DSA+SEA+SourceWeight tau=5` = Cho->Lee 71.93% / k=0.439, Lee->Cho 69.12% / k=0.382
  - `CSPNet + DSA+SEA+SessionEA+SourceWeight tau=5` = Cho->Lee 73.24% / k=0.465, Lee->Cho 69.21% / k=0.384
- Last updated: `2026-06-16 07:44` KST

## Cross-Dataset Final Matrix + forgit Refresh Started (2026-06-16)

- Status: `running`
- Unit: `mi-cross-dataset-final-matrix-forgit-20260616.service`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- Queue: `/home/hkim/MI_test/MI_loso_project/manage_cross_dataset_final_matrix_20260616.py`
- Summary: `/home/hkim/MI_test/cross_dataset_final_matrix_20260616.md`
- Log: `/home/hkim/MI_test/results/runs/cross_dataset_final_matrix_queue_20260616.log`
- After compute completes, these run automatically:
  - `refresh_forgit_crossdata.py`
  - `refresh_forgit_performance_summary.py`
- First running experiment: `EEGNet + SubjEA+AdaBN`
- Last updated: `2026-06-16 19:33` KST
