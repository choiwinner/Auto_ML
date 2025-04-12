# AutoML with PyCaret
=====================

## Overview
------------

This code uses the PyCaret library to create an automated machine learning (AutoML) pipeline. The pipeline includes data preprocessing, model selection, hyperparameter tuning, and model evaluation.

## Importing Libraries
----------------------

The code starts by importing the necessary libraries:
```python
import streamlit as st
import pandas as pd
import pycaret.classification as cls
import pycaret.regression as reg
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os


<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# PyCaret에서 사용하는 머신러닝 모델들의 알고리즘 분석

PyCaret은 Python 기반의 오픈 소스 저코드(low-code) 머신러닝 라이브러리로, 데이터 과학자들이 효율적으로 모델을 개발할 수 있도록 다양한 알고리즘과 기능을 제공합니다. 이 보고서에서는 PyCaret에서 지원하는 다양한 머신러닝 알고리즘을 상세히 분석합니다.

## PyCaret 개요

PyCaret은 R의 Caret 패키지에서 영감을 받아 개발된 Python 라이브러리로, 머신러닝 모델의 구축, 비교, 튜닝 및 배포 과정을 간소화합니다[^1]. 단 몇 줄의 코드로 다양한 머신러닝 알고리즘을 평가하고 비교할 수 있어 데이터 과학 프로젝트의 효율성을 크게 향상시킵니다.

## 분류 알고리즘 (Classification)

PyCaret은 다양한 분류 알고리즘을 지원하며, 이들은 `pycaret.classification` 모듈을 통해 접근할 수 있습니다.

### 로지스틱 회귀 (Logistic Regression)

로지스틱 회귀는 종속 변수가 범주형일 때 사용되는 통계적 방법입니다. 선형 함수에 시그모이드 함수를 적용하여 결과를 0과 1 사이의 확률로 변환합니다. PyCaret에서는 'lr' 식별자로 사용됩니다[^7][^8].

### K-최근접 이웃 (K-Nearest Neighbors)

KNN은 새로운 데이터 포인트의 클래스를 예측할 때 가장 가까운 K개의 이웃의 클래스를 기반으로 결정하는 알고리즘입니다. 거리 측정에는 주로 유클리드 거리가 사용됩니다. PyCaret에서는 'knn' 식별자로 사용됩니다.

### 나이브 베이즈 (Naive Bayes)

베이즈 정리를 기반으로 한 확률적 분류기로, 각 특성이 독립적이라고 가정합니다. 텍스트 분류에 특히 효과적입니다. PyCaret에서는 'nb' 식별자로 사용됩니다.

### 결정 트리 (Decision Tree)

결정 트리는 특성 기반의 결정 규칙을 통해 데이터를 분류합니다. 각 내부 노드는 특성에 대한 테스트를 나타내고, 각 브랜치는 테스트 결과를 나타내며, 각 리프 노드는 클래스 레이블을 나타냅니다. PyCaret에서는 'dt' 식별자로 사용됩니다[^3][^4].

### 서포트 벡터 머신 (Support Vector Machine)

SVM은 클래스 간의 최대 마진을 찾는 경계를 생성하는 알고리즘입니다. 커널 트릭을 통해 비선형 분류도 가능합니다. PyCaret에서는 'svm' 식별자로 사용됩니다.

### 랜덤 포레스트 (Random Forest)

랜덤 포레스트는 여러 결정 트리의 앙상블로, 각 트리는 데이터의 무작위 서브셋과 특성 서브셋으로 훈련됩니다. 최종 예측은 모든 트리의 예측을 집계하여 결정됩니다. PyCaret에서는 'rf' 식별자로 사용됩니다[^3][^4].

### 그래디언트 부스팅 (Gradient Boosting)

그래디언트 부스팅은 이전 모델의 오류를 수정하는 새 모델을 순차적으로 추가하는 앙상블 방법입니다. 이는 잔차(residual)에 기반한 모델을 점진적으로 구축합니다. PyCaret에서는 'gbc' 식별자로 사용됩니다.

### 선형 판별 분석 (Linear Discriminant Analysis)

LDA는 클래스 간의 차이를 최대화하고 클래스 내의 차이를 최소화하는 특성의 선형 조합을 찾는 차원 축소 기법입니다. PyCaret에서는 'lda' 식별자로 사용됩니다[^8].

## 회귀 알고리즘 (Regression)

PyCaret의 회귀 모듈은 `pycaret.regression`을 통해 접근할 수 있으며, 다양한 회귀 알고리즘을 제공합니다.

### 선형 회귀 (Linear Regression)

선형 회귀는 종속 변수와 하나 이상의 독립 변수 간의 선형 관계를 모델링합니다. 최소 제곱법을 사용하여 실제 값과 예측 값 사이의 잔차 제곱합을 최소화합니다.

### 라쏘 회귀 (Lasso Regression)

라쏘는 선형 회귀에 L1 정규화를 추가한 것으로, 일부 계수를 정확히 0으로 만들어 특성 선택 효과를 가집니다. 이는 모델의 복잡성을 줄이고 과적합을 방지하는 데 도움이 됩니다.

### 릿지 회귀 (Ridge Regression)

릿지는 선형 회귀에 L2 정규화를 추가한 것으로, 모든 계수를 0에 가깝게 만들어 과적합을 줄입니다. 다중 공선성이 있을 때 특히 효과적입니다.

### 엘라스틱넷 (Elastic Net)

엘라스틱넷은 라쏘와 릿지 정규화를 결합한 방법으로, L1과 L2 페널티의 가중치를 조정하여 두 방법의 장점을 활용합니다.

### 서포트 벡터 회귀 (Support Vector Regression)

SVR은 SVM의 회귀 버전으로, 데이터 포인트의 특정 마진 내에서 최대한 많은 데이터 포인트를 포함하는 초평면을 찾습니다.

### 결정 트리 회귀 (Decision Tree Regressor)

결정 트리 회귀는 분류와 유사하지만, 리프 노드에 클래스 레이블 대신 연속적인 값을 할당합니다.

### 랜덤 포레스트 회귀 (Random Forest Regressor)

랜덤 포레스트 회귀는 여러 결정 트리의 앙상블로, 각 트리의 예측 평균을 최종 예측으로 사용합니다.

## 클러스터링 알고리즘 (Clustering)

PyCaret의 클러스터링 모듈은 `pycaret.clustering`을 통해 접근할 수 있습니다.

### K-평균 (K-Means)

K-평균은 데이터를 K개의 클러스터로 나누는 알고리즘으로, 각 데이터 포인트를 가장 가까운 중심점(centroid)에 할당하고, 할당된 포인트들의 평균으로 중심점을 업데이트하는 과정을 반복합니다.

### 계층적 클러스터링 (Hierarchical Clustering)

계층적 클러스터링은 데이터 포인트 간의 거리를 기반으로 클러스터를 구축하는 상향식 또는 하향식 접근 방식을 사용합니다. 결과는 트리 형태의 덴드로그램으로 표현됩니다.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN은 밀도 기반 클러스터링 알고리즘으로, 고밀도 영역의 포인트들을 클러스터로 그룹화하고 저밀도 영역의 포인트들을 노이즈로 처리합니다.

## 앙상블 기법

PyCaret은 여러 모델을 결합하여 성능을 향상시키는 다양한 앙상블 기법을 제공합니다.

### 배깅 (Bagging)

배깅은 동일한 알고리즘의 여러 인스턴스를 부트스트랩 샘플에 대해 훈련시키고 그 결과를 평균화하는 방법입니다. PyCaret에서는 `ensemble_model()` 함수를 통해 구현할 수 있으며, 기본 방법은 배깅입니다[^5].

### 부스팅 (Boosting)

부스팅은 이전 모델의 오류를 보완하는 새로운 모델을 순차적으로 추가하는 방법입니다. PyCaret에서는 `ensemble_model(method='Boosting')` 형태로 구현할 수 있습니다[^5].

### 블렌딩 (Blending)

블렌딩은 여러 모델의 예측을 소프트 보팅(각 클래스에 대한 확률의 합산)으로 결합하는 방법입니다. PyCaret에서는 `blend_models()` 함수를 통해 구현할 수 있습니다[^5][^7].

### 스태킹 (Stacking)

스태킹은 여러 모델의 출력에 가중치를 적용한 가중 합을 계산하는 방법입니다. 첫 번째 단계에서는 각 서브 모델의 예측을 생성하고, 두 번째 단계에서는 각 모델에 해당하는 가중치를 학습합니다. PyCaret에서는 `stack_models()` 함수를 통해 구현할 수 있습니다[^5].

## 모델 최적화 기능

PyCaret은 모델의 성능을 최적화하기 위한 다양한 기능을 제공합니다.

### 모델 튜닝 (Model Tuning)

`tune_model()` 함수는 K-fold 교차 검증을 사용하여 모델의 하이퍼파라미터를 최적화합니다. 기본적으로 랜덤 서치를 사용하지만, 'optuna', 'scikit-optimize', 'tune-sklearn' 등 다양한 최적화 라이브러리를 선택할 수 있습니다[^1][^3][^5].

### 하이퍼파라미터 공간 정의

PyCaret은 모든 모델에 대한 기본 검색 공간을 정의하고 있지만, `custom_grid` 매개변수를 통해 사용자 정의 검색 공간도 지정할 수 있습니다[^5].

## 결론

PyCaret은 분류, 회귀, 클러스터링, 이상 탐지 등 다양한 머신러닝 작업을 위한 포괄적인 알고리즘 세트를 제공합니다. 또한 모델 비교, 하이퍼파라미터 튜닝, 앙상블 방법 등을 통해 모델 성능을 최적화할 수 있는 다양한 기능을 제공합니다. 이러한 기능들은 단 몇 줄의 코드로 구현할 수 있어, 데이터 과학자의 생산성을 크게 향상시킬 수 있습니다[^2][^4][^6].

PyCaret의 직관적인 인터페이스와 강력한 기능은 초보자부터 전문가까지 모든 수준의 사용자가 머신러닝 프로젝트를 효율적으로 진행할 수 있도록 지원합니다. 특히 어떤 알고리즘이 특정 데이터셋에 가장 적합한지 빠르게 파악하고자 할 때 매우 유용합니다[^8].

<div>⁂</div>

[^1]: https://www.machinelearningmastery.com/pycaret-for-machine-learning/

[^2]: https://minimin2.tistory.com/137

[^3]: https://abluesnake.tistory.com/137

[^4]: https://wikidocs.net/226645

[^5]: https://baechu-story.tistory.com/68

[^6]: https://jaylala.tistory.com/entry/머신러닝-with-파이썬-Pycaret이란-Pycaret을-활용한-머신러닝

[^7]: https://velog.io/@ezoo0422/Python-pycaret을-사용하여-모델-선정하기

[^8]: https://2-54.tistory.com/101

[^9]: https://pycaret.org

[^10]: https://dsbook.tistory.com/360

[^11]: https://github.com/pycaret/pycaret

[^12]: https://pycaret.gitbook.io/docs/get-started/preprocessing

[^13]: https://datasciencebeehive.tistory.com/6

[^14]: https://pycaret.gitbook.io/docs/learn-pycaret/official-blog/write-and-train-custom-ml-models-using-pycaret

[^15]: https://sthsb.tistory.com/31

[^16]: https://www.datasource.ai/uploads/624e8836466a40923b64b901b5050c0f.html

[^17]: https://42-snoopy.tistory.com/entry/ML-pycaret

[^18]: https://pycaret.readthedocs.io/en/stable/api/classification.html

[^19]: https://dsbook.tistory.com/361

[^20]: https://2-54.tistory.com/101

[^21]: https://minimin2.tistory.com/137

[^22]: https://velog.io/@hyungenie/AutoML-PyCaret

[^23]: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115\&topMenu=100\&aihubDataSe=realm\&dataSetSn=71392

[^24]: https://devocean.sk.com/blog/techBoardDetail.do?ID=165238\&boardType=techBlog

[^25]: https://brunch.co.kr/@ueber/364

[^26]: https://abluesnake.tistory.com/137

[^27]: https://velog.io/@workhard/Pycaret-사용법-및-Voting

[^28]: https://dacon.io/competitions/official/235647/codeshare/2428

[^29]: https://hyeon827.tistory.com/60
