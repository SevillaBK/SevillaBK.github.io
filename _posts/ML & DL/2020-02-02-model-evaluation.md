---
layout: post
title: '모델의 평가'
excerpt: 모델의 성능 평가에 사용되는 지표들을 알아보자
category: ML & DL
tags:
  - 사이킷런
  - 혼동행렬
  - 정확도
  - 재현율
  - 민감도
  - 정밀도
  - F1
  - ROC
  - AUC

---



이번 포스팅에서는 학습한 모델을 어떻게 평가를  해야하는지 알아보겠습니다.<br/>모델 평가를 위해 사용되는 지표는 여러 가지가 있지만 대표적으로 `정확도(accuracy)` 가 있고,  `민감도(재현율)(recall)`, `특이도(specificity)`,  `정밀도(precision)`, `F1 score` 등이 있습니다.

## 정확도(Accuracy)

정확도는 정답과 예측값이 얼마나 동일한지를 판단하는 지표입니다.<br/>직관적으로 모델 예측 성능을 나타내는 평가지표이지만 데이터 레이블의 구성에 따라 모델 성능을 왜곡할 수 있습니다. 가령, 캐글의 타이타닉 예제에서도 생존자 중 여성의 비중이 높기 때문에, 특별한 알고리즘 없이 여성을 모두 생존자로 분류해도 정확도를 높게 나올 수 있습니다.

```python
# 캐글 타이타닉 데이터셋에서 사망자 분포 확인
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv('Titanic/input/train.csv')

sns.countplot(data = titanic_df, x = 'Survived', hue = 'Sex')
plt.title('Number of Survived with Sex')
plt.show()
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/titanic.png?raw=true" width = "300" height = "200">

Survived 가 1이 생존자를 나타내는데, 위의 그래프에서 보는 것 처럼 생존자 중 여성의 비중이 확연히 높습니다. 이런 경우에는 단순히 성별에 따라 생존자를 분류해도 정확도가 높게 나올 수 있습니다.

사이킷런의 `BaseEstimator` 를 활용하여, 단순히 성별에 따라 생존자 여부를 예측하는 분류기를 생성해서 정확도를 보겠습니다.

※ 사이킷런의 `BaseEstimator` 는 Customized된 Estimator를 생성할 수 있는 클래스입니다.

```python
import numpy as np

# fit() 메서드는 아무 것도 수행하지 않고
# predict()는 Sex가 1이면 0, 그렇지 않으면 1로 예측하는 단순한 분류기 생성
from sklearn.base import BaseEstimator

class DummyClassifier(BaseEstimator):
  def fit(self, X, y = None):
    pass
  
  def predict(self, X):
    pred = np.zeros((X.shape[0],1))
    for i in range(X.shape[0]):
      if X['Sex'].iloc[i] == 1:
        pred[i] = 0
      else:
        pred[i] = 1
    return pred
```

```python
# 생성된 DummyClassifier를 이용해 타이타닉 예제 생존자 예측 수행
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Null 처리 함수
def fillna(df):
  df['Age'].fillna(df['Age'].mean(), inplace = True)
  df['Cabin'].fillna('N', inplace = True)
  df['Embarked'].fillna('N', inplace = True)
  df['Fare'].fillna(0, inplace = True)
  return df

# 학습에 불필요한 피처 제거
def drop_feature(df):
  df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)
  return df

# LabelEncoder 수행
def format_feature(df):
  df['Cabin'] = df['Cabin'].str[:1]
  features = ['Cabin', 'Sex', 'Embarked']
  for feature in features:
    label_encoder = LabelEncoder()
    label_encoder.fit(df[feature])
    df[feature] = label_encoder.transform(df[feature])
  return df

# 위의 함수들을 한꺼번에 실행하는 함수
def transform_features(df):
  df = fillna(df)
  df = drop_feature(df)
  df = format_features(df)
  return df
```

```python
# 타이타닉 데이터 로딩 및 학습 데이터 / 테스트 데이터 분할
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop(['Survived'], axis = 1)
X_titanic_df = transform_features(X_titanic_df)

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, 
                                                    y_titanic_df,
                                                    test_size = 0.2,
                                                    random_state = 10)

# 위에서 생성한 Dummy Classifier를 활용해서 학습/예측/평가 수행
model = DummyClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print('Dummy Classifier의 정확도: {0: .4f}'.format(accuracy_score(y_test, predictions)))
```

```
# 출력:
Dummy Classifier의 정확도:  0.8212
```

위와 같이 단순한 알고리즘으로 예측을 하더라도, 데이터 레이블의 구성으로 인해 정확도는 82.12%로 매우 높은 수치가 나왔습니다. 때문에 정확도를 평가지표로 사용할 때는 신중해야 합니다.

정확도는 특히, 불균형한 레이블 값 분포에서 모델 성능을 판단할 경우에는 적합한 평가지표가 아닙니다.<br/>가령 100개의 데이터 중 95개의 레이블이 0, 5개의 레이블이 1인 경우에 무조건 0을 반환하는 모델을 만들면 정확도가 95%가 됩니다. 하지만 이런 데이터셋에서는 대개 목표는 1을 찾아내는 것입니다. 때문에 정확도만으로는 목적에 맞는 모델평가를 수행할 수 없습니다.

이후에는 정확도 외에 또 어떤 지표를 활용하여 모델을 평가할 수 있는지 보겠습니다.



## 혼동행렬(Confusion Matrix)

`혼동행렬(Confusion Matrix)` 은 분류문제에서 예측 오류가 얼마나 되고, 어떤 유형의 오류가 발생하는지를 보여주는 행렬로 아래와 같은 배열을 가지고 있습니다.

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/confusion%20matrix.jpg?raw=true" width ="60%">

그 후, 학습셋과 검증데이터셋 전체를 다시 학습하여 테스트셋에 대해 예측을 수행합니다. 이를 통해 학습하지 않은 데이터에 대해서도 모델의 성능이 잘 나오는지 확인합니다.

* `TN(True Negative)` : 실제 값이 음성(Negative)인데 예측 값도 음성(Negative) → 맞게 예측
* `FP(False Positive)` : 실제 값이 음성(Negative)인데 예측 값은 양성(Positive) 
* `FN(False Negative)` : 실제 값이 양성(Positive)인데 예측 값은 음성(Negative)
* `TP(True Positive)` : 실제 값이 양성(Positive)인데 예측 값도 양성(Positive) → 맞게 예측

사이킷런은 혼동행렬을 구하기 위해 `confusion_matrix( )` 함수를 제공합니다.<br/>출력 결과의 배열은 위의 그림과 동일합니다.

앞에서 DummyClassifier로 예측한 결과를 이용해서 혼동행렬을 출력해보겠습니다.

```python
# DummyClassifier를 이용하여 예측한 결과와 y_test를 비교하여 혼동행렬 출력하기
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
print(cm)
print('')
print('True Negative : ' , cm[0][0], ' → 사망자(0)를 사망자(0)로 예측')
print('False Positive : ', cm[0][1], '→ 사망자(0)인데 생존자(1)로 예측')
print('False Negative : ', cm[1][0], ' → 생존자(1)인데 사망자(0)로 예측')
print('True Positive: ', cm[1][1], '→ 생존자(1)를 생존자(1)로 예측')
```

```
# 출력:
[[104  13]
 [ 19  43]]
 
True Negative :  104  → 사망자(0)를 사망자(0)로 예측
False Positive :  13 → 사망자(0)인데 생존자(1)로 예측
False Negative :  19  → 생존자(1)인데 사망자(0)로 예측
True Positive:  43 → 생존자(1)를 생존자(1)로 예측
```



## 오차행렬을 통해 알 수 있는 지표들

##### (1) 정확도(Accuracy) = ( TN + TP ) / ( TN + FP + FN + TP )

* 위에서 살펴보았던 정확도입니다. 전체 예측 중 양성을 양성이라 예측하고, 음성을 음성이라고 예측한 갯수의 비율입니다. 레이블의 분포에 따라 한쪽으로만 분류하더라도 결과가 높게 나올 수 있습니다.

##### (2) 정밀도(Precision) = TP / ( FP + TP )

* 양성으로 예측한 것 중 실제로 양성인 비율입니다.

##### (3) 재현율(Recall) = TP / ( FN + TP )

* 재현율은 전체 양성 데이터 중 양성으로 예측한 수의 비율입니다. 다른 말로 민감도라고도 하며 양성에 얼마나 민감하게 반응하느냐는 의미입니다.
* 민감도(Sensitivity), 또는 TPR(True Positive Rate) 이라고도 합니다.

##### (4) 특이도(Specificity) = TN / ( TN + FP )

* 특이도는 전체 음성 데이터 중 음성으로 예측한 수의 비율입니다. 음성을 얼마나 잘 골라내는지를 판정하는 지표입니다.
* TNR(True Negative Rate) 이라고도 합니다.

##### ※  FPR(False Positive Rate) = 1 - Specificity = FP / ( TN + FP )

* 거짓양성비율(FalsePositiveRate)은 음성 중 양성으로 잘못 판정된 것의 비율입니다.



프로젝트의 목적에 따라 특정 지표가 유용하게 사용됩니다.<br/>가령, 암 환자 판별의 경우에는 정상인을 암환자으로 분류하더라도, 암인 케이스를 반드시 판정해내야 합니다. 때문에 재현율이 중시됩니다.

사이킷런에서는 정확도, 재현율, 정밀도 계산을 위해`accuracy_score( )`, `recall_score( )` 와 `precision_score( )` 함수를 제공합니다.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 호출한 지표들을 한꺼번에 계산하는 함수 정의
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print('오차행렬')
    print(confusion)
    print('')
    print('정확도 : {:.4f}'.format(accuracy))
    print('정밀도 : {:.4f}'.format(precision))
    print('재현율 : {:.4f}'.format(recall))
```

로지스틱회귀 모델을 이용해 타이타닉 예제의 생존자를 다시 예측 후 평가를 수행해보겠습니다.

```python
from sklearn.linear_model import LogisticRegression

titanic_df = pd.read_csv('Titanic/input/train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop(['Survived'], axis = 1)
X_titanic_df = transform_features(X_titanic_df)

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df,
                                                    y_titanic_df,
                                                    test_size = 0.2,
                                                    random_state = 10)

clf = LogisticRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
get_clf_eval(y_test, pred)
```

```
# 출력:
[[104  13]
 [ 17  45]]

정확도 : 0.8324
정밀도 : 0.7759
재현율 : 0.7258
```



## 정밀도와 재현율의 trade-off 관계

정밀도와 재현율은 상호보완적인 지표로 한 쪽을 높이려고 하면, 다른 한 쪽이 떨어지기 쉽습니다.

사이킷런의 분류 알고리즘은 예측 데이터가 특정 레이블에 속하는지 판단하기 위해 개별 레이블별로 확률을 구하고, 그 확률이 큰 레이블 값으로 예측합니다.

- 이진 분류의 경우, 일반적으로는 임계값을 50%로 정하고 이보다 크면 양성(Positive), 작으면 음성(Negative)로 결정합니다.
- `predict_proba( )` 를 사용하여 개별 레이블별 예측확률을 반환할 수 있습니다.

```python
# 타이타닉 생존자 데이터에서 predict() 결과 값과 predict_proba()결과값을 비교하기
pred_proba = clf.predict_proba(X_test)
pred = clf.predict(X_test)

print('pred_proba의 shape: {}'.format(pred_proba.shape))
print('pred_proba의 array의 앞 5개만 샘플로 추출:\n', pred_proba[:5])
# 예측확률 array와 예측결과값 array를 병합하여 예측확률과 결과값을 한 번에 확인하기
pred_proba_result = np.concatenate([pred_proba, pred.reshape(-1, 1)], axis=1)
print('두 개의 레이블 중 더 큰 확률을 클래스 값으로 예측: \n', pred_proba_result[:5])
```

```
# 출력:
pred_proba의 shape: (179, 2)
pred_proba의 array의 앞 5개만 샘플로 추출:
 [[0.88292372 0.11707628]
 [0.84599254 0.15400746]
 [0.86140126 0.13859874]
 [0.08019536 0.91980464]
 [0.13863631 0.86136369]]
두 개의 레이블 중 더 큰 확률을 클래스 값으로 예측: 
 [[0.88292372 0.11707628 0.        ]
 [0.84599254 0.15400746 0.        ]
 [0.86140126 0.13859874 0.        ]
 [0.08019536 0.91980464 1.        ]
 [0.13863631 0.86136369 1.        ]]
```



#### 정밀도/재현율 trade-off 관계를 살펴보기

사이킷런의 `Binarizer` 클래스를 활용하여 임계치에 따른 정밀도/재현율의 변화를 살펴보겠습니다.

※  `Binarizer`  : fit_transform( )을 이용하여 정해진 임계치보다 같거나 작으면 0, 크면 1보다 반환

```python
from sklearn.preprocessing import Binarizer

# 예시
X = [[-1, -1, 2],
     [2, 0, 0], 
     [0, 1.1, 1.2]]

# X의 개별원소들이 threshold보다 크면 1, 작거나 같으면 0을 반환
binarizer = Binarizer(threshold=1.1)
print(binarizer.fit_transform(X))
```

```
# 출력:
[[0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]]
```

앞선 Logistic Regression 객체의 predict_proba()의 결과 값에 Binarizer클래스를 적용하여 최종 예측 값을 구하고, 최종 예측 값에 대해 평가해보겠습니다.

```python
# 임계값
thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]

# 임계치에 따른 평가지표를 조사하기 위한 함수 생성
def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold)
        custom_predict = binarizer.fit_transform(pred_proba_c1)
        print('임계값: ', custom_threshold)
        get_clf_eval(y_test, custom_predict)
        print('---------------')

pred_proba = clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)
```

```
# 출력:
임계값:  0.4
오차행렬
[[98 19]
 [14 48]]

정확도 : 0.8156
정밀도 : 0.7164
재현율 : 0.7742
---------------
임계값:  0.45
오차행렬
[[101  16]
 [ 17  45]]

정확도 : 0.8156
정밀도 : 0.7377
재현율 : 0.7258
---------------
임계값:  0.5
오차행렬
[[104  13]
 [ 17  45]]

정확도 : 0.8324
정밀도 : 0.7759
재현율 : 0.7258
---------------
임계값:  0.55
오차행렬
[[105  12]
 [ 23  39]]

정확도 : 0.8045
정밀도 : 0.7647
재현율 : 0.6290
---------------
임계값:  0.6
오차행렬
[[107  10]
 [ 25  37]]

정확도 : 0.8045
정밀도 : 0.7872
재현율 : 0.5968
---------------
```

임계값이 증가할수록 정밀도 값은 높아지지만 재현율 값이 낮아짐을 볼 수 있습니다. <br/>이를 `precision_recall_curve( )` 함수와 matplitlib을 활용하여 시각화해보겠습니다.

※ `precision_recall_curve` : 실제클래스값과 예측 확률값을 입력인자로 받아 임계값에 따른 정밀도, 재현율, 임계값을 ndarray로 반환

```python
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import precision_recall_curve

def precision_recall_curve_plot(y_test, predict_proba_c1):
    # 정밀도, 재현율, 임계값을 ndarray로 추출하기
    precisions, recalls, thresholds = precision_recall_curve(y_test, predict_proba_c1)
    
    # x축을 임계값, y축을 정밀도, 재현율로 그래프 그리기
    plt.figure(figsize = (8, 6))
    thresholds_boundary = thresholds.shape[0]
    # 정밀도와 재현율은 임계값보다 1개 적게 반환되기 때문에 슬라이싱을 해주어야 합니다.
    plt.plot(thresholds, precisions[:thresholds_boundary], 
             linestyle = '--', label='precision')
    plt.plot(thresholds, recalls[:thresholds_boundary], 
             linestyle = '-', label = 'recall')
    
    # x축의 scale을 0.1 단위로 변경
    plt.xticks(np.round(np.arange(0, 1, 0.1),2))
    plt.xlabel('thresholds')
    plt.ylabel('precision(recall)')
    
    plt.legend()
    plt.grid()
    plt.show()
    
precision_recall_curve_plot(y_test, clf.predict_proba(X_test)[:, 1])
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/precision-recall-curve.png?raw=true" width = "60%">

위 그래프를 보면 임계값을 높임에 따라 정밀도는 개선되지만, 재현율은 나빠짐을 한눈에 알 수 있습니다.



## F1 스코어

정밀도와 재현율을 결합한 지표로 정밀도와 재현율이 어느 한 족으로 치우지지 않을 때 상대적으로 높은 값을 가집니다.

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/f1 score.png?raw=true" width = "40%">

예시 ) 모델 A의 정밀도 0.9, 재현율 0.1 / 모델 B의 정밀도 0.5, 재현율 0.5<br/>          → 모델 A의 F1 = 0.18, 모델 B의 F1 = 0.5



사이킷런에서는 쉽게 f1을 계산할 수 있도록 `f1_score( )` 함수를 제공하고 있습니다.

```python
from sklearn.linear_model import LogisticRegresion

titanic_df = pd.read_csv('Titanic/input/train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop(['Survived'], axis = 1)
X_titanic_df = transform_features(X_titanic_df)

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df,
                                                    y_titanic_df,
                                                    test_size = 0.2,
                                                    random_state = 10)

clf = LogisticRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# f1 score를 계산하기
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print('정밀도(Precision) : {:.4f}'.format(precision))
print('재현율(Recall) : {:.4f}'.format(recall))
print('F1 : {:.4f}'.format(f1))
```

```
# 출력:
정밀도(Precision) : 0.7759
재현율(Recall) : 0.7258
F1 : 0.7500
```



#### 타이타닉 생존자 예측에서 임계값을 변화시키며 F1, 정밀도, 재현율 구하기

```python
# 예측결과에 따른 평가지표 계산
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print('오차행렬')
    print(confusion)
    # F1 score print 추가
    print('\n정확도: {:.4f}\n정밀도: {:.4f}\n재현율: {:.4f}\nF1: {:.4f}'.format(accuracy, precision, recall, f1))

# 임계값에 따른 평가지표 계산    
def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold)
        custom_predict = binarizer.fit_transform(pred_proba_c1)
        print('임계값: ', custom_threshold)
        get_clf_eval(y_test, custom_predict)
        print('---------------')
    
thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
pred_proba = clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1, 1), thresholds)
```

```
# 출력:
임계값:  0.4
오차행렬
[[98 19]
 [14 48]]

정확도: 0.8156
정밀도: 0.7164
재현율: 0.7742
F1: 0.7442
---------------
임계값:  0.45
오차행렬
[[101  16]
 [ 17  45]]

정확도: 0.8156
정밀도: 0.7377
재현율: 0.7258
F1: 0.7317
---------------
임계값:  0.5
오차행렬
[[104  13]
 [ 17  45]]

정확도: 0.8324
정밀도: 0.7759
재현율: 0.7258
F1: 0.7500
---------------
임계값:  0.55
오차행렬
[[105  12]
 [ 23  39]]

정확도: 0.8045
정밀도: 0.7647
재현율: 0.6290
F1: 0.6903
---------------
임계값:  0.6
오차행렬
[[107  10]
 [ 25  37]]

정확도: 0.8045
정밀도: 0.7872
재현율: 0.5968
F1: 0.6789
---------------
```

```python
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import precision_recall_curve

def precision_recall_curve_plot(y_test, predict_proba_c1):
    # 정밀도, 재현율, 임계값을 ndarray로 추출하기
    precisions, recalls, thresholds = precision_recall_curve(y_test, predict_proba_c1)
    f1 = 2 * (precisions * recalls) / (precisions + recalls)
    
    # x축을 임계값, y축을 정밀도, 재현율로 그래프 그리기
    plt.figure(figsize = (8, 6))
    thresholds_boundary = thresholds.shape[0]
    # 정밀도와 재현율은 임계값보다 1개 적게 반환되기 때문에 슬라이싱을 해주어야 합니다.
    plt.plot(thresholds, precisions[:thresholds_boundary], 
             linestyle = '--', label='precision')
    plt.plot(thresholds, recalls[:thresholds_boundary], 
             linestyle = '-', label = 'recall')
    plt.plot(thresholds, f1[:thresholds_boundary], 
             linestyle = ':',  label = 'f1')
    
    # x축의 scale을 0.1 단위로 변경
    plt.xticks(np.round(np.arange(0, 1, 0.1),2))
    plt.xlabel('thresholds')
    plt.ylabel('score')
    
    plt.legend()
    plt.grid()
    plt.show()
    
pred_proba = clf.predict_proba(X_test)
precision_recall_curve_plot(y_test, pred_proba[:, 1])
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/precision-recall-f1.png?raw=true" width = "60%">

위의 그래프를 보면, f1 score 는 민감도와 재현율이 적절히 균형을 맞춘 지점에서 가장 높게 나타남을 알 수 있습니다.



## ROC곡선과 AUC

ROC곡선은 False Positive Rate(FPR)이 변할 때 True Positive Rate(TPR)이 어떻게 변하는지를 나타내는 곡선입니다.

* False Positive Rate : 음성 중 양성으로 잘못 판정한 것의 비율
* True Positive Rate : 양성 중 양성으로 잘 판정한 것의 비율

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/ROC-curve.png?raw=true" width = "60%">

위의 그림은 ROC 곡선의 예시로 가운데 대각선은 무작위로 분류를 하는 분류기의 ROC 곡선입니다.
곡선이 가운데 대각선에 가까울수록 성능이 떨어지며 멀어질수록 성능이 뛰어난 것입니다.

사이킷런의 `roc_curve( )` 기능을 활용하면, FPR, TPR, 임계치을 반환합니다.

#### 타이타닉 생존자 예측모델의 FPR, TPR, 임계치 구하기

```python
from sklearn.metrics import roc_curve

# 레이블 값이 1일 때 예측 확률을 추출
pred_proba_class1 = clf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, pred_proba_class1)

# 반환된 임계값 중 샘플로 10건만 추출하되 임계값을 5 step으로 추출
thr_index = np.arange(1, thresholds.shape[0], 5)
print('샘플 추출된 임계값 배열의 index 10개: ', thr_index)
print('샘플용 10개의 임계값: ', np.round(thresholds[thr_index], 2))

# 샘플로 추출된 임계값에 따른 FPR, TPR 값
print('샘플 임계값별 FPR: ', np.round(fpr[thr_index], 3))
print('샘플 임계값별 TPR: ', np.round(tpr[thr_index], 3))
```

```
# 출력: 
샘플 추출된 임계값 배열의 index 10개:  [ 1  6 11 16 21 26 31 36 41 46 51]
샘플용 10개의 임계값:  [0.95 0.69 0.66 0.51 0.38 0.28 0.23 0.15 0.13 0.13 0.04]
샘플 임계값별 FPR:  [0. 0.017 0.06  0.103 0.179 0.248 0.316 0.573 0.667 0.735 1. ]
샘플 임계값별 TPR:  [0.016 0.516 0.565 0.726 0.774 0.839 0.887 0.903 0.935 0.952 1. ]
```

roc_curve( )의 결과값을 보면 임계값이 1에 가까운 값에서 점점 작아질수록 FPR이 점점 커집니다. 그리고 FPR이 조금씩 커질 때 TPR은 가파르게 커집니다. 이것을 그래프로 시각화해보겠습니다.

```python
# ROC 곡선의 시각화
def roc_curve_plot(y_test, pred_proba_c1):
    #임계값에 따른 FPR, TPR 값을반환 받음
    fprs, tprs, thresholds  = roc_curve(y_test, pred_proba_c1)
    # ROC곡선을 그래프로 그림
    plt.plot(fprs, tprs, label='ROC')
    # 가운데 대각선 직선을 그림
    plt.plot([0,1], [0,1], 'k--', label='Random')
    
    # FPR X축의 Scale을 0.1 단위로 변경, X, Y축 명 설정 등
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR(1-Sensitivity)')
    plt.ylabel('TPR(Recall)')
    plt.legend()
    
roc_curve_plot(y_test, pred_proba[:, 1])
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/ROC-Titanic.png?raw=true" width = "60%">

일반적으로 ROC 곡선 자체는 FPR과 TPR의 변화 값을 보는데 이용하며 분류의 성능지표로는 ROC 면적에 기반한 AUC 값으로 결정합니다.

* `AUC(Area Under Curve)` : 0.5와 1사이의 값을 가집니다. 곡선 밑의 면적 값으로 1에 가까울수록 좋은 수치입니다. ROC 곡선이 대각선 자체일 때는 0.5입니다. 

  → `roc_auc_score( )` 함수를 이용해서 AUC 값을 구할 수 있습니다.

```python
from sklearn.metrics import roc_auc_score

pred = clf.predict(X_test)
roc_score = roc_auc_score(y_test, pred)
print('ROC AUC 값 : {:.4f}'.format(roc_score))
```

```
# 출력:
ROC AUC 값 : 0.8073
```



---------

###### Reference

- 파이썬 머신러닝 완벽가이드
