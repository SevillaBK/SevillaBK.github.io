---
layout: post
title: '[사이킷런] 사이킷런의 기본 프레임워크'
excerpt: 사이킷런의 사용하기 위해 기본 구조를 학습해보자
category: MachineLearning
tags:
  - 사이킷런
  - MachineLearning

---



## Estimator 클래스 및 fit( ), predict( ) 메소드

사이킷런의 Estimator 클래스는 기본적으로 `Classifier`와 `Regressor`로 나뉩니다.<br/>각각의 Estimator는 내부에서 `fit( )` 과 `predict( )` 를 내부에서 구현하고 있습니다. 

*  `fit( )` : 주어진 데이터로 모델을 학습시키는 메소드입니다. 

* `predict( )` : 학습된 모델로 예측을 수행하는 메소드입니다.
* `transform( )` : 입력된 데이터의 형태에 맞추어 데이터를 변환하는 메소드 입니다.

`cross_val_score()`(평가 함수)나 `GridSearchCV( )`(하이퍼 파라미터 튜닝) 같은 클래스의 경우에는 Estimator를 인자로 받고, fit( )과 predict( ) 를 호출해서 평가하거나 튜닝을 수행합니다.



## 사이킷런의 주요 모듈

|    분류     |           모듈명           |                             설명                             |
| :---------: | :------------------------: | :----------------------------------------------------------: |
| 예제 데이터 |      sklearn.datasets      |                사이킷런 내장 예제 데이터 세트                |
|  피처처리   |   sklearn.preprocessing    | 데이터 전처리에 필요한 다양한 가공기능<br/>(인코딩, 정규화, 스케일링 등) |
|  피처처리   | sklearn.feature_selection  | 알고리즘에 큰 영향을 미치는 피처를 우선순위대로 셀렉션하는 기능 제공 |
|  피처처리   | sklearn.feature_extraction |   텍스트, 이미지 데이터의 벡터화된 피처를 추출하는데 사용    |
|  차원축소   |   sklearn.decomposition    | 차원 축소와 관련된 알고리즘을 지원<br/>(PCA, NMF, Truncated SVD 등) |
|  모델선택   |  sklearn.model_selection   |     훈련, 테스트 데이터 분리, 그리드 서치 등의 기능 제공     |
|    평가     |      sklearn.metrics       | 다양한 모델의 성능평가 측정방법 제공<br/>(Accuracy, ROC-AUC, RMSE 등 |
|  알고리즘   |      sklearn.ensemble      | 앙상블 알고리즘 제공(RandomForest, AdaBoost, Gradient Boost  등) |
|  알고리즘   |    sklearn.linear_model    | 회귀 관련 알고리즘 제공<br/>(linear, Ridge, Lasso, Logistic 등) |
|  알고리즘   |    sklearn.naive_bayes     |         나이브 베이즈 알고리즘 제공(Gaussian NB 등)          |
|  알고리즘   |     sklearn.neighbors      |              최근접이웃 알고리즘 제공(K-NN 등)               |
|  알고리즘   |        sklearn.svm         |                  서포트 벡터 머신 알고리즘                   |
|  알고리즘   |        sklearn.tree        |                    의사결정 트리 알고리즘                    |
|  알고리즘   |      sklearn.cluster       |                  비지도 클러스터링 알고리즘                  |
|  유틸리티   |      sklearn.pipeline      | 피처처리 등의 변환과 ML 알고리즘 학습, 예측 등을 함께 묶어서 실행하는 유틸리티 제공 |



## 내장된 예제 데이터 세트 살펴보기

```python
# 붓꽃 예제 데이터 세트 불러오기
from sklearn.datasets import load_iris

iris_data = load_iris()
print(type(iris_data))
```

```
# 출력:
<class 'sklearn.utils.Bunch'>
```

```python
keys = iris_data.keys()
print('붓꽃 데이터 세트의 키들: ', keys)
```

```
# 출력:
붓꽃 데이터 세트의 키들:  dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
```

```python
print('feature_names의 type: ', type(iris_data.feature_names))
print('feature_names의 shape: ', len(iris_data.feature_names))
print('feature_names: ', iris_data.feature_names)

print('\ntarget_names의 type: ', type(iris_data.target_names))
print('target_names의 shape: ', len(iris_data.target_names))
print('target_names: ', iris_data.target_names)

print('\ndata의 type: ', type(iris_data.data))
print('data의 shape: ', iris_data.data.shape)
print('data: \n', iris_data['data'])

print('\ntarget의 type: ', type(iris_data.target))
print('target의 shape: ', iris_data.target.shape)
print('target: \n', iris_data['target'])
```

```
# 출력: 
feature_names의 type:  <class 'list'>
feature_names의 shape:  4
feature_names:  ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

target_names의 type:  <class 'numpy.ndarray'>
target_names의 shape:  3
target_names:  ['setosa' 'versicolor' 'virginica']

data의 type:  <class 'numpy.ndarray'>
data의 shape:  (150, 4)
data: 
 [[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]
 [5.4 3.9 1.7 0.4]
 [4.6 3.4 1.4 0.3]
 .
 .
 (중략)
 .
 .
 [6.3 2.5 5.  1.9]
 [6.5 3.  5.2 2. ]
 [6.2 3.4 5.4 2.3]
 [5.9 3.  5.1 1.8]]

target의 type:  <class 'numpy.ndarray'>
target의 shape:  (150,)
target: 
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
```

---------

###### Reference

- 파이썬 머신러닝 완벽가이드
