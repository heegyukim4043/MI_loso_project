# Priority Claim Analysis

Generated: 2026-06-24

This note ranks the proposed NS/NE-style claims by what can be supported immediately from the current repository outputs.

## Priority 1: BCI Inefficiency / Practical Coverage

Status: directly supported from subject-level LOSO and cross-dataset CSVs.

Key LOSO coverage results:

| Setting | Dataset | Method | Mean | >=70% subjects |
|---|---|---|---:|---:|
| LOSO | Cho2017 | CSP-LDA | 60.66 | 9/52 = 17.3% |
| LOSO | Cho2017 | EA+CSPNet | 71.43 | 28/52 = 53.8% |
| LOSO | Cho2017 | EA+AdaBN+CSPNet | 71.04 | 30/52 = 57.7% |
| LOSO | Cho2017 | EA+SupCon+AdaBN+CSPNet | 71.81 | 31/52 = 59.6% |
| LOSO | All | CSP-LDA | 61.74 | 22/106 = 20.8% |
| LOSO | All | EA+CSPNet | 71.56 | 56/106 = 52.8% |
| LOSO | All | EA+Snapshot+AdaBN+CSPNet | 72.09 | 59/106 = 55.7% |
| LOSO | All | EA+SupCon+AdaBN+CSPNet | 72.08 | 61/106 = 57.5% |

Key cross-dataset coverage results:

| Direction | Method | Mean | >=70% subjects |
|---|---|---:|---:|
| Cho->Lee | CSP-LDA baseline | 55.15 | 3/54 = 5.6% |
| Cho->Lee | DatasetEA+SubjectEA+CSP-LDA | 68.96 | 22/54 = 40.7% |
| Cho->Lee | DSA+SEA+SessionEA+SourceWeight+CSPNet | 73.24 | 32/54 = 59.3% |
| Lee->Cho | CSP-LDA baseline | 52.76 | 1/52 = 1.9% |
| Lee->Cho | DatasetEA+SubjectEA+CSP-LDA | 65.10 | 16/52 = 30.8% |
| Lee->Cho | DSA+SEA+SessionEA+SourceWeight+CSPNet | 69.21 | 22/52 = 42.3% |

Safe claim:

> The apparent fraction of BCI-inefficient users is strongly method-dependent. In Cho2017 LOSO, the fraction of subjects above a practical 70% threshold increases from 17.3% with CSP-LDA to 53.8-59.6% with aligned CSPNet variants.

Use this as the strongest immediate subject-level claim.

## Priority 2: Hierarchical Alignment Impact

Status: directly supported as an alignment-impact/performance-bottleneck analysis. Avoid calling this a direct measurement of distribution distance unless covariance distances are computed.

Classical cross-dataset hierarchy:

| Direction | Stage | Accuracy | Delta vs baseline | Delta vs previous |
|---|---|---:|---:|---:|
| Cho->Lee | CSP-LDA baseline | 55.15 | 0.00 | 0.00 |
| Cho->Lee | DatasetEA+SubjectEA+CSP-LDA | 68.96 | +13.81 | +13.81 |
| Cho->Lee | SessionEA+CSP-LDA | 70.74 | +15.59 | +1.78 |
| Lee->Cho | CSP-LDA baseline | 52.76 | 0.00 | 0.00 |
| Lee->Cho | DatasetEA+SubjectEA+CSP-LDA | 65.10 | +12.34 | +12.34 |
| Lee->Cho | SessionEA+CSP-LDA | 65.14 | +12.38 | +0.04 |

Deep/source-weighted hierarchy:

| Direction | Stage | Accuracy | Delta vs previous |
|---|---|---:|---:|
| Cho->Lee | DSA+SEA+SourceWeight+CSPNet | 71.93 | 0.00 |
| Cho->Lee | DSA+SEA+SessionEA+SourceWeight+CSPNet | 73.24 | +1.31 |
| Lee->Cho | DSA+SEA+SourceWeight+CSPNet | 69.12 | 0.00 |
| Lee->Cho | DSA+SEA+SessionEA+SourceWeight+CSPNet | 69.21 | +0.09 |

Safe claim:

> Cross-dataset performance is dominated by dataset/subject-level alignment effects. DatasetEA+SubjectEA adds +13.81 percentage points in Cho->Lee and +12.34 percentage points in Lee->Cho over CSP-LDA, while SessionEA adds a smaller incremental gain.

Stronger claim requiring more analysis:

> Dataset shift is larger than subject/session shift.

To make this stronger, compute direct covariance-distance or domain-classifier evidence between datasets, subjects, and sessions.

## Priority 3: Calibration-Free Deployment

Status: directly supported if framed as threshold feasibility, not clinical readiness.

Strong but safe wording:

> Under a zero-target-label protocol, aligned CSPNet variants move roughly half of the LOSO cohort above the 70% accuracy threshold, and the best cross-dataset method reaches 59.3% above-threshold coverage in Cho->Lee.

Avoid:

> Clinically deployable BCI for 54-57% of users.

Reason: clinical/deployment readiness needs online validation, latency, stability, and task-specific utility beyond offline accuracy.

## Priority 4: Equipment As Hidden Variable

Status: plausible discussion claim, not fully proven from current CSVs alone.

Current support:

- Cross-dataset baseline performance is very low: 55.15% for Cho->Lee and 52.76% for Lee->Cho.
- DatasetEA+SubjectEA recovers large accuracy: +13.81 and +12.34 percentage points.
- This is consistent with a strong recording-system/domain variable.

Safe claim:

> The large gain from DatasetEA suggests that dataset-level recording differences act as a major hidden domain variable in cross-dataset MI decoding.

Needs additional evidence for a stronger equipment-specific claim:

- Explicit mapping of Cho2017 and Lee2019 hardware/recording systems.
- Covariance-distance comparison: inter-dataset vs inter-subject vs inter-session.
- Optional domain classifier: predict dataset/equipment from covariance features before and after DatasetEA.

## Priority 5: CSP Filter Topographic Analysis

Status: high impact, but not directly executable from the current packaged files.

Current blocker:

- No `.pt`, `.pth`, `.ckpt`, `.npy`, `.npz`, or saved CSP spatial-filter artifact is present in this workspace.
- The repository contains scripts and result CSVs, but not trained model weights or saved CSP filters.

What is needed:

- Saved CSP filters/spatial weights from CSP-LDA or CSPNet, or
- Access to preprocessed EEG arrays so CSP filters can be recomputed, or
- Rerun script that saves filters per fold/dataset.

Recommended figure:

| Panel | Comparison |
|---|---|
| A | CSP-LDA no alignment topomap |
| B | EA+CSP-LDA or EA+CSPNet topomap |
| C | DatasetEA+SubjectEA cross-dataset topomap |
| D | Best DSA+SEA+SessionEA+SourceWeight topomap if filters are available |

Claim to test:

> Alignment improves recovery of neuroanatomically plausible sensorimotor CSP patterns around C3/C4.

## Recommended Order

1. Use BCI inefficiency/coverage as the strongest immediate subject-level claim.
2. Use hierarchical alignment impact as the core methodological result.
3. Use calibration-free threshold feasibility as the engineering implication.
4. Keep equipment hidden variable as a discussion claim until covariance/hardware analysis is added.
5. Add CSP topomap only after filters or preprocessed EEG are available.

## Generated Supporting Files

| File | Purpose |
|---|---|
| `analysis_outputs/coverage_threshold_metrics.csv` | LOSO and cross-dataset threshold coverage at 60/70/75%. |
| `analysis_outputs/hierarchical_alignment_impact.csv` | Dataset/subject/session alignment impact table. |
| `analysis_outputs/loso_subject_robustness_metrics.csv` | LOSO method-level robustness metrics. |
| `analysis_outputs/loso_paired_delta_metrics.csv` | Paired subject-level deltas and NTR. |
| `analysis_outputs/loso_paired_subject_deltas.csv` | Full subject-level paired deltas. |
