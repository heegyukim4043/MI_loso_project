# Transfer Benefit Predictor / Source Similarity 요약

## 산출 파일

- `subject_covariance_features_loso.csv`: LOSO subject별 covariance 안정성/거리 feature.
- `transfer_benefit_loso_features.csv`: LOSO subject별 `delta = candidate_acc - baseline_acc` + feature 결합 테이블.
- `transfer_benefit_loso_spearman.csv`: LOSO Spearman correlation 결과.
- `subject_source_similarity_crossdataset.csv`: cross-dataset source pool similarity feature. Cho/Lee 공통 48채널 사용.
- `transfer_benefit_cross_features.csv`: cross-dataset subject별 transfer benefit + source similarity feature.
- `transfer_benefit_cross_spearman.csv`: cross-dataset Spearman correlation 결과.
- `transfer_benefit_predictor_report.md`: 전체 자동 요약 보고서.

## D. Transfer Benefit Predictor

질문: 어떤 subject가 generalization/alignment에 더 취약하거나 더 큰 이득을 보는가?

현재 결과에서 가장 강한 신호는 CSPNet의 `NoEA -> EA`입니다.

| 비교 | Dataset | Feature | rho | p | 해석 |
|---|---:|---|---:|---:|---|
| CSPNet NoEA -> EA | Cho2017 | source_pool_knn10_dist | 0.419 | 0.002 | source pool과 더 먼 subject일수록 EA 이득이 큼 |
| CSPNet NoEA -> EA | Cho2017 | source_pool_sim_weight | -0.413 | 0.002 | 가까운 source subject가 많을수록 EA 이득은 작아짐 |
| CSPNet NoEA -> EA | Cho2017 | cov_condition_num | 0.333 | 0.016 | covariance가 불안정한 subject일수록 EA 이득이 큼 |
| CSPNet NoEA -> EA | Lee2019 | baseline_acc | -0.347 | 0.010 | baseline이 낮은 subject일수록 EA 이득이 큼 |
| Conformer NoEA -> EA | Cho2017 | baseline_acc | -0.321 | 0.020 | baseline 취약 subject가 EA 이득을 더 봄 |

CSPNet NoEA -> EA의 평균 이득:

| Dataset | n | mean delta | median delta |
|---|---:|---:|---:|
| Cho2017 | 52 | +5.66%p | +5.00%p |
| Lee2019 | 54 | +1.69%p | +1.50%p |

주장 가능 문장:

- EA는 평균 성능만 올리는 것이 아니라, baseline이 낮거나 covariance/source-pool 관점에서 outlier에 가까운 subject의 취약성을 줄이는 방향으로 작동한다.
- 특히 CSPNet에서는 source pool과 먼 subject일수록 EA benefit이 커져, EA를 "cross-subject covariance mismatch 보정기"로 해석할 수 있다.
- 이 결과는 "누가 generalization에 취약한가"를 baseline accuracy와 covariance geometry로 사전에 추정할 수 있다는 예비 근거다.

주의:

- 이 분석은 Spearman correlation 기반이므로 causal predictor라고 쓰면 안 된다.
- backbone/method 전체에서 동일하게 반복되는 신호는 아니고, CSPNet NoEA -> EA에서 가장 일관적이다.

## E. Source-Subject Similarity vs Benefit

질문: test subject와 유사한 source subject가 많을수록 generalization benefit이 커지는가?

cross-dataset CSP-LDA 결과에서는 단순한 "유사 source가 많을수록 좋다" 가설이 방향별로 일관되지는 않았다.

| 비교 | Train -> Test | Feature | rho | p | 해석 |
|---|---|---|---:|---:|---|
| CrossCSP -> SubjectEA | Cho -> Lee | baseline_acc | -0.688 | <0.001 | baseline이 낮은 Lee subject가 SubjectEA 이득을 크게 봄 |
| CrossCSP -> DatasetEA+SubjectEA | Lee -> Cho | source_dataset_mean_dist | 0.501 | <0.001 | Lee source pool과 더 먼 Cho subject가 DSA+SEA 이득을 크게 봄 |
| CrossCSP -> DatasetEA+SubjectEA | Lee -> Cho | source_dataset_sim_weight | -0.501 | <0.001 | 가까운 source가 많을수록 추가 이득은 작음 |
| CrossCSP -> SubjectEA | Lee -> Cho | source_dataset_mean_dist | 0.315 | 0.023 | 더 먼 Cho subject가 SubjectEA 이득을 더 봄 |

Cross-dataset 평균 이득:

| 비교 | Train -> Test | n | mean delta | median delta |
|---|---|---:|---:|---:|
| CrossCSP -> DatasetEA+SubjectEA | Cho -> Lee | 54 | +12.95%p | +11.75%p |
| CrossCSP -> DatasetEA+SubjectEA | Lee -> Cho | 52 | +11.19%p | +9.00%p |
| CrossCSP -> SubjectEA | Cho -> Lee | 54 | -2.28%p | -0.50%p |
| CrossCSP -> SubjectEA | Lee -> Cho | 52 | +6.48%p | +5.05%p |

주장 가능 문장:

- Cross-dataset에서는 DatasetEA+SubjectEA가 양방향 모두 큰 평균 이득을 보인다.
- source similarity는 "유사 source가 많으면 항상 좋다"가 아니라, alignment가 source pool에서 멀리 떨어진 target subject에게 더 큰 보정 이득을 제공한다는 방향으로 해석하는 편이 현재 결과와 맞다.
- 따라서 E의 가설은 원문 그대로 쓰기보다 "source-subject covariance geometry predicts who benefits from alignment"로 바꾸는 것이 더 안전하다.

논문 방향:

- D는 LOSO의 subject vulnerability 분석으로 바로 사용 가능.
- E는 "source similarity가 benefit을 예측한다"까지는 가능하지만, "유사 source가 많을수록 좋다"는 주장은 현재 결과와 반대 방향이 섞여 있으므로 피하는 것이 좋다.
