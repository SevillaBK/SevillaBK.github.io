---
layout: post
title: '[사이킷런] RandomForest(랜덤포레스트)'
excerpt: RandomForest의 개념과 사이킷런에서의 활용법
category: ML & DL
tags:
  - 앙상블
  - RandomForest
  - Bagging
  - 사이킷런

---



이번 포스팅에서는 `랜덤포레스트(RandomForest)` 의 개념을 간단히 정리해보겠습니다.<br/>

## 배깅(Bagging)이란?

랜덤포레스트를 이야기하기 전에 앞선 포스팅에서 먼저 다루었단 배깅의 개념에 대해서 다시 정리하겠습니다.

**배깅(Bagging)**은 Bootstrap Aggregating의 약자로 보팅(Voting)과는 달리 동일한 알고리즘으로 여러 분류기를 만든 후 예측 결과를 보팅으로 최종 결정하는 알고리즘입니다.

배깅은 아래와 같은 방식으로 진행됩니다.

(1) 동일한 알고리즘을 사용하는 일정 수의 분류기 생성

(2) 각각의 분류기는 **부트스트리팽(Bootstrapping)** 방식으로 생성된 샘플 데이터를 학습

(3) 최종적으로 모든 분류기의 보팅을 통해 예측 결정

※ 부트스트래핑 샘플링은 전체 데이터에서 일부 데이터의 중첩을 허용하는 샘플링 방식입니다.

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-17-RandomForest-1.png?raw=true">



## 랜덤포레스트(RandomForest)

랜덤 포레스트는 여러 개의 결정트리(Decision Tree)를 활용한 배깅방식의 대표적인 알고리즘입니다.

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-17-RandomForest-2.png?raw=true">




#### 장점

* 결정트리의 쉽고 직관적인 장점을 그대로 가지고 있습니다.
* 앙상블 알고리즘 중 비교적 빠른 수행속도를 가지고 있습니다.
* 다양한 분야에서 좋은 성능을 나타냅니다.

#### 단점

* 하이퍼 파라미터가 많아 튜닝을 위한 많은 시간이 소요됩니다.



#### 사용자 행동 데이터셋을 이용한 RandomForest 예측

사용자 행동 데이터셋을 RandomForest에 적용하여 동작 예측모델을 만들어보겠습니다.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
```

```python
# human activity 데이터 세트에 중복된 Feature명으로 인해 판다스 0.25버전 이상에서 
# Duplicate name 에러가 발생하여 feature 이름을 수정하는 함수 설정
def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1]) 
                                                                                           if x[1] >0 else x[0] ,  axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df
```

```python
# 데이터셋을 구성하는 함수 설정
def get_human_dataset():
    
    # 각 데이터 파일들은 공백으로 분리되어 있으므로 read_csv에서 공백문자를 sep으로 할당
    feature_name_df = pd.read_csv('human_activity/features.txt', sep='\s+',
                                                     header=None, names=['column_index', 'column_name'])
    
    # 중복된 피처명을 수정하는 get_new_feature_name_df()를 이용하여 새로운 feature명 데이터프레임 생성
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    
    # 데이터프레임에 피처명을 컬럼으로 뷰여하기 위해 리스트 객체로 다시 반환
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()
    
    # 학습 피처 데이터세트와 테스트 피처 데이터를 데이터프레임으로 로딩
    # 컬럼명은 feature_name 적용
    X_train = pd.read_csv('human_activity/train/X_train.txt', sep='\s+', names=feature_name)
    X_test = pd.read_csv('human_activity/test/X_test.txt', sep='\s+', names=feature_name)
    
    # 학습 레이블과 테스트 레이블 데이터를 데이터 프레임으로 로딩, 컬럼명은 action으로 부여
    y_train = pd.read_csv('human_activity/train/y_train.txt', sep='\s+', names=['action'])
    y_test = pd.read_csv('human_activity/test/y_test.txt', sep='\s+', names=['action'])
    
    # 로드된 학습/테스트용 데이터프레임을 모두 반환
    return X_train, X_test, y_train, y_test

# 학습/테스트용 데이터 프레임 반환
X_train, X_test, y_train, y_test = get_human_dataset()
```

```python
# 랜덤 포레스트 학습 및 별도의 테스트 세트로 예측 성능 평가
rf_clf = RandomForestClassifier(random_state=221)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('랜덤 포레스트 정확도: {:.4f}'.format(accuracy))
```

```
# 출력:
랜덤 포레스트 정확도: 0.9220
```

파라미터 값을 default로 수행한 결과, 랜덤포레스트는 사용자 행동 인식 데이터셋에 92.20%의 정확도를 보입니다.



## 랜덤포레스트의 하이퍼파라미터 튜닝

랜덤포레스트는 트리 기반의 하이퍼파라미터를 사용하여 하이퍼 파라미터를 튜닝합니다. 사용되는 주요 파라미터들은 다음과 같습니다.

* **n_estimators**
  * 사용되는 Decision Tree의 갯수를 지정
  * default : 10
  *  무작정 트리 갯수를 늘리면 성능 좋아지는 것 대비 시간이 걸릴 수 있음

* **min_samples_split**

  * 노드를 분할하기 위한 최소한의 샘플 데이터수<br/>→ 과적합을 제어하는데 사용합니다. 값이 작을수록 분할노드가 많아져 과적합 가능성 증가
  * default : 2 

* **min_samples_leaf**

  * 리프노드가 되기 위한 최소한의 샘플 데이터수<br/>→ 과적합을 제어하는데 사용합니다. 값이 작을수록 과적합 가능성 증가
  * default : 1
  * 불균형 데이터의 경우 특정 클래스 데이터가 극도로 적을 수 있으므로 작은 값으로 설정 필요

* **max_features**

  * 최적의 분할을 위해 고려할 피처의 최대 갯수

  * default : auto (Decision Tree에서는 default가 None인 것과 차이)

  * int형으로 지정 → 피처 갯수

  * float형으로 지정 → 전체 갯수의 일정 비율만큼 사용

  * `sqrt` 또는 `auto` → 전체 피처 중 √(피처 개수) 만큼 선정

  * `log2` : 전체 피처 중 log2(전체 피처 개수) 만큼 선정

* **max_depth**

  * 트리의 최대 깊이
  * default : None
    → 완벽하게 클래스 값이 결정될 때까지 분할<br/>     또는 데이터 갯수가 min_samples_split보다 작아질 때까지 분할
  * 깊이가 깊어지면 과적합될 수 있으므로 적절히 제어 필요

* **max_leaf_nodes**

  * 리프노드의 최대 갯수

  * default : None

    

```python
# RandomForest의 하이퍼 파라미터 default 상태
model = RandomForestClassifier()
model
```

```
# 출력:
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
```



#### GridSearchCV를 통한 랜덤포레스트의 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

params = {'n_estimators' : [10, 100],
          'max_depth' : [6, 8, 10, 12],
          'min_samples_leaf' : [8, 12, 18],
          'min_samples_split' : [8, 16, 20]
          }

# RandomForestClassifier 객체 생성 후 GridSearchCV 수행
rf_clf = RandomForestClassifier(random_state = 221, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf,
                       param_grid = params,
                       cv = 3, 
                       n_jobs = -1)
grid_cv.fit(X_train, y_train)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
```

```
# 출력:
최적 하이퍼 파라미터:  {'max_depth': 12, 'min_samples_leaf': 12, 'min_samples_split': 8, 'n_estimators': 100}
최고 예측 정확도: 0.9193
```

max_depth : 12 , min_samples_leaf : 12 , min_samples_split : 8 , n_estimators : 100 일 때, 정확도가 91.93%로 측정되었습니다.

이번에는 이 파라미터로 RandomForestClassifier를 다시 학습시킨 뒤, 별도의 데이터셋에서 예측성능을 측정해보겠습니다.

```python
# 위의 결과로 나온 최적 하이퍼 파라미터로 다시 모델을 학습하여 
# 테스트 세트 데이터에서 예측 성능을 측정
rf_clf1 = RandomForestClassifier(n_estimators = 100, 
                                max_depth = 12,
                                min_samples_leaf = 12,
                                min_samples_split = 8,
                                random_state = 0,
                                n_jobs = -1)
rf_clf1.fit(X_train, y_train)
pred = rf_clf1.predict(X_test)
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test,pred)))
```

```
# 출력:
예측 정확도: 0.9213
```



## 랜덤포레스트의 각 피처 중요도 시각화 : feature_importances_

max_depth : 12 , min_samples_leaf : 12 , min_samples_split : 8 , n_estimators : 100 일 때,  어떤 피처가 크게 영향을 미쳤는지 보기 위해 feature importance를 시각화해보겠습니다.

```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ftr_importances_values = rf_clf1.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Top 20 Feature Importances')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-22-Human-dataset-5.png?raw=true">



---------

###### Reference

- 파이썬 머신러닝 완벽가이드
- https://www.researchgate.net/publication/322179244_Data_Mining_Accuracy_and_Error_Measures_for_Classification_and_Prediction
- https://pvsmt99345.i.lithium.com/t5/image/serverpage/image-id/33046iB8743F7094DB9C87/image-size/large?v=1.0&px=999 
