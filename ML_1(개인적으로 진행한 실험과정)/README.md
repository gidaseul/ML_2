# ML_1 Personal Experiment Log

이 폴더는 `ML_1` 최종 산출물을 만들기 전에 진행한 개인 탐구 기록입니다.  
학부연구생 과정에서 SportsBall 데이터의 전처리 효율성과 조합을 직접 실험했고, 그 결과가 최종 프로젝트 파이프라인으로 이어졌습니다.

## Why This Folder Exists

- 목표는 단순히 모델을 바꾸는 것이 아니라, `어떤 전처리 조합이 실제로 분류 성능에 더 큰 영향을 주는지`를 확인하는 것이었습니다.
- 특히 SportsBall은 배경 잡음, 객체 크기 변화, 비원형 클래스가 많아서 `ROI 품질`, `전처리 순서`, `특징 조합`의 영향이 매우 컸습니다.
- 이 폴더는 정제된 결과 보고서가 아니라, 그 결론에 도달하기 전의 실험 로그와 중간 산출물을 모아 둔 작업 아카이브입니다.

## What Is Inside

- `Image_processing/전처리후_분류모델선택.ipynb`
  ROI를 추출한 뒤 어떤 분류기를 붙이는 것이 적절한지 비교한 노트북
- `Image_processing/단계별코드_업그레이드_모델기법 순서 변경.ipynb`
  전처리 단계의 순서를 바꿨을 때 성능과 결과 이미지가 어떻게 달라지는지 확인한 기록
- `Image_processing/image2_roi_svm.ipynb`
  ROI 추출과 SVM 기반 분류를 연결한 실험
- `Image_processing/xgboost_65.ipynb`
  XGBoost 적용 실험과 성능 비교
- `Image_processing/randomforest_1025.ipynb`
  Random Forest 기반 대안 실험
- `다슬/HOG_hist.ipynb`
  `HOG only`와 `HOG + color histogram` 조합을 비교한 실험
- `다슬/new_ROI.ipynb`
  ROI 방식과 PCA 기반 정리 실험
- `다슬/BOVW.ipynb`
  Bag of Visual Words 기반 특징 표현 실험

## Main Insight

- SportsBall에서는 분류기 선택보다 `객체가 포함된 영역을 얼마나 안정적으로 잘라내는가`가 먼저 중요했습니다.
- 전처리 단계의 순서도 결과에 영향을 주었고, 특정 조합은 같은 모델을 써도 성능 차이를 크게 만들었습니다.
- `HOG + 색상 히스토그램` 같은 조합은 단일 특징보다 더 나은 분리 가능성을 보여 주었습니다.
- 따라서 이 실험의 핵심은 `모델 성능 개선`보다 `전처리 설계의 영향력을 실험적으로 검증한 것`에 있습니다.

## Relation To Final Project

- 이 폴더는 개인 탐구 로그입니다.
- 최종적으로 정리된 팀 프로젝트 결과는 [`../ML_1/README.md`](../ML_1/README.md)에 있습니다.
- 최종 PDF는 [`../ML_1/ML_1(Image Classification  with Machine Learning).pdf`](<../ML_1/ML_1(Image Classification  with Machine Learning).pdf>)에서 확인할 수 있습니다.

## Notes

- 노트북 이름과 파일 구성이 일정하지 않은 이유는, 실험을 반복하며 중간 파생본이 계속 생겼기 때문입니다.
- 따라서 이 폴더는 재현용 패키지보다는 `탐구 과정 자체를 보여 주는 증거 폴더`로 보는 것이 맞습니다.
