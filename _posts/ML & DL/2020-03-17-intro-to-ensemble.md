---
layout: post
title: '앙상블 학습'
excerpt: 앙상블의 기초개념 학습
category: ML & DL
tags:
  - 앙상블
  - Voting
  - Bagging
  - Stacking
  - 사이킷런
---



이번 포스팅에서는 `앙상블(Ensemble)` 의 개념을 간단히 정리해보겠습니다.<br/>

## 앙상블 학습이란?

**앙상블(Ensemble)** 이란 여러 개의 알고리즘을 사용하여, 그 예측을 결합함으로써 보다 정확한 예측을 도출하는 기법을 말합니다. 앙상블은 집단 지성이 힘을 발휘하는 것처럼 단일의 강한 알고리즘보다 복수의 약한 알고리즘이 더 뛰어날 수 있다는 생각에 기반을 두고 있습니다.

이미지, 영상, 음성 등의 비정형 데이터의 분류는 딥러닝이 뛰어난 성능을 보이지만, 많은 정형 데이터의 분류에서는 앙상블이 뛰어난 성능을 보인다고 합니다.

앙상블 학습의 유형은 **보팅(Voting), 배깅(Bagging), 부스팅(Boosting), 스태킹(Stacking)** 등이 있습니다. 

- **보팅**은 여러 종류의 알고리즘을 사용한 각각의 결과에 대해 투표를 통해 최종 결과를 예측하는 방식입니다.
- **배깅**은 같은 알고리즘에 대해 데이터 샘플을 다르게 두고, 학습을 수행해 보팅을 수행하는 방식입니다. 이 때의  데이터 샘플은 중첩이 허용됩니다. 가령, 10000개의 데이터에 대해 10개의 알고리즘이 배깅을 사용할 때, 각 1000개의 데이터 내에는 중복된 데이터가 존재할 수 있습니다. 배깅을 사용하는 대표적인 방식이 **Random Forest** 입니다.

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-17-Ensemble-1.png?raw=true">

- **부스팅**은 여러 개의 알고리즘이 순차적으로 학습을 하되, 앞에 학습한 알고리즘의 예측이 틀린 데이터에 대해 올바르게 예측할 수 있도록, 그 다음번 알고리즘에 가중치를 부여하여 학습과 예측을 진행하는 방식입니다.
- 마지막으로 **스태킹**은 여러 가지 다른 모델의 예측 결과값을 다시 학습 데이터로 만들어 다른 모델(메타모델)로 재학습시켜 결과를 예측하는 방법입니다.



## 하드보팅(Hard Voting)과 소프트보팅(Soft Voting)

보팅은 다시 하드보팅과 소프트보팅으로 나눌 수 있습니다. 

**하드보팅**은 다수결 원칙과 비슷합니다. **소프트보팅**은 각 알고리즘이 레이블 값 결정 확률을 예측해고, 이것들을 평균하여 이들 중 확률이 가장 높은 레이블 값을 최종 예측값으로 예측합니다.

일반적으로는 소프트보팅의 성능이 더 좋아서 많이 적용된다고 합니다.

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-17-Ensemble-2.png?raw=true">



## 사이킷런의 VotingClassifier

사이킷런은 보팅방식의 앙상블을 구현한 VotingClassifier 클래스를 제공하고 있습니다.

사이킷런에서 제공되는 위스콘신 유방암 데이터 세트를 이용해 보팅방식의 앙상블을 적용해보겠습니다.

```python
# 필요한 모듈과 데이터 불러오기
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from warnings import filterwarnings
filterwarnings('ignore')

cancer = load_breast_cancer()

data_df = pd.DataFrame(cancer.data, columns = cancer.feature_names)
data_df.head(3)
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-17-Ensemble-3.png?raw=true">



로지스틱회귀와 KNN을 기반으로 소프트보팅 방식의 분류기를 만들어 보겠습니다.

```python
# 보팅 적용을 위한 개별 모델은 로지스틱회귀와 KNN 입니다.
logistic_regression = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)

# 개별 모델을 소프트보팅 기반의 앙상블 모델로 구현한 분류기
voting_model = VotingClassifier(estimators = [('Logistic Regression', logistic_regression),('KNN', knn)],
                                voting='soft')

# 데이터를 훈련셋과 테스트셋으로 나누기
X_train, X_test, y_train, y_test = train_test_split(cancer.data, 
                                                    cancer.target, 
                                                    test_size = 0.3, 
                                                    random_state = 50)
# 보팅 분류기의 학습/예측/평가
voting_model.fit(X_train,y_train)
pred = voting_model.predict(X_test)
print('보팅 분류기의 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))

# 개별 모델의 학습/예측/평가
classifiers = [logistic_regression, knn]
for classifier in classifiers:
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    class_name = classifier.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, accuracy_score(y_test, pred)))
```

```
# 출력:
보팅 분류기의 정확도: 0.9532
LogisticRegression 정확도: 0.9474
KNeighborsClassifier 정확도: 0.9357
```

위와 같이 간단한 두 알고리즘을 사용한 경우에도, 보팅 분류기의 정확도가 개별 모델의 정확도보다 조금 높게 나타났습니다. 하지만 항상 여러 알고리즘을 결합한다고 항상 성능이 향상되는 것은 아닙니다.





---------

###### Reference

- 파이썬 머신러닝 완벽가이드
- https://subinium.github.io/introduction-to-ensemble-1/
