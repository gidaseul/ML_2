# Representation Learning

루트 [README](../README.md)에서 `방법 2. 표현학습`에 해당하는 보조 문서입니다.  
이 프로젝트는 전처리와 특징 조합만으로는 설명하기 어려운 SportsBall 데이터의 복잡성을, `latent representation` 관점에서 다시 해석하려는 시도였습니다.

## PDF Archive

표지를 클릭하면 실제 PDF가 열립니다.

| 보고서 PDF | 발표자료 PDF |
| --- | --- |
| [![ML_2 report cover](../assets/images/ml2-report-cover.png)](<./ML프로젝트2_02팀/ML_2(Image Classification  with Representation Learning).pdf>) | [![ML_2 presentation cover](../assets/images/ml2-ppt-cover.png)](<./ML프로젝트2_02팀/ML_2(ppt).pdf>) |

## Focus

| 항목 | 내용 |
| --- | --- |
| 목적 | 모델이 데이터를 어떤 공간에 어떻게 표현하는지 이해하는 것 |
| 데이터 | `MNIST`, `SportsBall` |
| 핵심 기법 | `Base CNN Encoder`, `Autoencoder`, `Projection Head`, `Contrastive Learning`, `InfoNCE`, `t-SNE`, `Heatmap` |
| 핵심 포인트 | 성능 향상뿐 아니라 표현 공간이 실제로 어떻게 형성되는지 해석 |

## Why This Project Started

- `ML_1`에서 전처리와 특징 조합을 직접 탐구한 결과, SportsBall 데이터는 입력 정제만으로 해결하기 어려운 한계가 있다는 점이 분명해졌습니다.
- 그래서 다음 단계에서는 `입력을 더 잘 만드는 문제`를 넘어서, `모델 내부 표현이 클래스 구조를 얼마나 잘 분리하는가`를 보는 방향으로 전환했습니다.

## What Was Implemented

- `Base CNN Encoder`를 설계해 특징 표현 학습
- `Autoencoder`를 적용해 latent representation과 복원 성능 확인
- `Projection Head`를 추가해 contrastive learning 실험
- `InfoNCE` 기반 학습으로 같은 클래스는 가깝게, 다른 클래스는 멀어지게 만드는 구조 시도
- `t-SNE`로 train/test latent space 분포 시각화
- Heatmap으로 모델이 실제로 주목한 영역 해석

## Main Results

| 구분 | 결과 |
| --- | --- |
| MNIST | Train Accuracy `99.56%` |
| MNIST | Test Accuracy `98.00%` |
| SportsBall | Train Accuracy `70.90%` |
| SportsBall | Base model Test Accuracy `48.00%` |
| SportsBall | Autoencoder 기반 표현학습 실험에서는 테스트 정확도 `10%` 수준 구간 확인 |
| SportsBall | 최종 분석에서 성능이 좋지 않았던 contrastive learning 실험 결과 `9.00%` 비교 |

## Evidence

![Representation learning clustering and latent-space evidence](../assets/images/representation-latent-evidence.png)

- 출처: `[2024-2_ML] Project1 specifications/Project.ipynb`, `../mnist/mnist_latent_feature.ipynb`, `../sportsball/sportsball_latent_feature.ipynb`
- 구성: K-Means, GMM, 그리고 t-SNE 기반 feature-space 시각화
- 의미: MNIST는 클래스 간 분리가 비교적 선명하고, SportsBall은 배경과 객체 변화 때문에 latent space가 더 넓고 복잡하게 퍼지는 경향을 보여 줍니다.

## Why This Method Matters

- MNIST처럼 구조가 단순한 데이터는 표현학습이 안정적으로 작동했습니다.
- SportsBall처럼 배경 잡음이 많고 객체가 복잡한 데이터는, 표현학습만으로 바로 성능이 좋아지지 않았습니다.
- 이 단계의 가장 큰 의미는 "왜 성능이 낮았는가"를 latent space와 heatmap으로 설명한 점입니다.
- 결과적으로 표현학습은 항상 성능 향상으로 이어지는 것이 아니라, 데이터 구조와 학습 목표가 잘 맞아야 한다는 점을 확인했습니다.

## Related Files

- [`ML_2 최종 보고서 PDF`](<./ML프로젝트2_02팀/ML_2(Image Classification  with Representation Learning).pdf>)
- [`ML_2 발표자료 PDF`](<./ML프로젝트2_02팀/ML_2(ppt).pdf>)
- [`ML_2 코드 압축본`](<./ML프로젝트2_02팀/ML_소스코드_02팀 .zip>)

## Notes Before Reproducing

- 노트북 내부 경로가 과거 로컬 환경인 `/home/gidaseul/Documents/GitHub/ML_2/...` 로 남아 있어 현재 폴더 구조에 맞는 수정이 필요합니다.
- `contrastive_t_sne(test).ipynb`는 `contrastive_weights3.pth`를 불러오도록 되어 있지만, 현재 폴더에는 `contrastive_weights.pth`가 있어 파일명 또는 경로를 맞춰야 합니다.

## Summary

표현학습 방법은 단순히 정확도를 높이기 위한 실험이 아니라,  
`latent space가 어떻게 형성되는지`, `왜 어떤 경우에는 잘 작동하지 않는지`를 해석하기 위한 접근이었습니다.
