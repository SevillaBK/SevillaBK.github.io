---
layout: post
title: 'DecisionTree(결정트리) 개괄과 사이킷런에서의 활용'
excerpt: DecisionTree의 기초적인 내용을 이해하고, 사이킷런에서 실행해보기
category: ML & DL
tags:
  - 사이킷런
  - DecisionTree

---



이번 포스팅에서는 여러 알고리즘 중 `DecisionTree(결정트리)`에 대해 정리해보겠습니다.<br/>

## Decision Tree란?

데이터에 있는 규칙을 학습을 통해 자동으로 찾아내 꼬리에 꼬리를 무는 트리 형태의 규칙을 만드는 알고리즘입니다. 조금 더 쉽게 이야기하자면 if 와 else에 해당하는 조건들을 찾아내 예측을 위한 규칙을 만드는 알고리즘입니다.



## Decision Tree의 구조

Decision Tree는 기본적으로 세 가지 노드( `루트노드, 리프노드, 규칙노드` )로 이루어져있습니다.

* **루트노드(Root Node)** : 트리의 시작점

* **리프노드(Leaf Node)** : 트리의 끝단으로 알고리즘에 의해 결정된 값

* **규칙노드/내부노드(Decision Node / Internal Node)** : 데이터셋의 피처들이 결합해 만들어진 규칙 조건

  

#### 캐글 타이타닉 데이터셋에서의 Decision Tree 예시

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-10-titanic-decision-tree.png?raw=true">

위의 그림[( 출처: https://bigwhalelearning.wordpress.com/2014/11/27/77/ )](https://bigwhalelearning.wordpress.com/2014/11/27/77/)은 캐글 타이타닉 예제를 통해 Decision Tree의 구조를 간략히 나타낸 것으로 아래와 같이 노드를 분류할 수 있습니다.

* **Is Passenger Male?**  →  루트노드
* **Age < 18**, **3rd Class?**, **Embarked from Southhampton?** → 규칙노드
* **Died**, **Survived** : 리프노드

Decision Tree에서 노드들을 많이 생성하여 규칙들을 만들어내면 그만큼 알고리즘이 복잡해져 세밀한 결과를 낼 수도 있습니다. 하지만 그만큼 주어진 데이터셋에 대한 과적합(Overfitting)으로 이어지기 쉽습니다.<br/>(다른 말로, 트리의 깊이(depth)가 깊어질수록 Decision Tree는 과적합되기 쉬워 예측 성능이 저하될 수 있습니다.)

가능한 적은 규칙노드로 높은 성능을 가지려면 데이터 분류를 할 때, 최대한 많은 데이터셋이 해당 분류에 속할 수 있도록 규칙 노드의 규칙이 정해져야 합니다. 이를 위해 최대한 균일한 데이터셋이 구성되도록 분할하는 것이 필요합니다. 즉, 분할된 데이터가 특정 속성을 잘 나타내도록 만들어야 합니다.

규칙노드는 정보균일도가 높은 데이터셋으로 쪼개지도록 조건을 찾아 서브 더이터셋을 만들고, 서브 데이터셋에서 이런 작업을 반복하며 최종 값을 예측하게 됩니다.

사이킷런의 Decision Tree는 기본적으로 **지니계수**를 이용하여 데이터를 분할합니다.

※ **지니계수** : 경제학에서 불평등지수를 나타낼 때 사용하는 것으로 0일 때 완전 평등, 1일 때 완전 불평등을 의미합니다. 머신러닝에서는 데이터셋의 데이터가 다양한 값을 가질수록 평등하며, 특정 값으로 쏠릴수록 불평등해집니다. 즉, 다양성이 낮을수록 균일도가 높아 1에 가까워진다는 의미로 사이킷런에서는 지니계수가 높은 속성을 기준으로 데이터를 분할합니다. 



## Decision Tree의 장단점

### 장점

* 쉽고 직관적입니다.
* 각 피처에 대한 스케일링과 정규화같은 전처리 작업의 영향도가 크지 않습니다.

### 단점

* 규칙을 추가하여 서브트리를 만들어 나갈수록 모델이 복잡해지고, 과적합에 빠지기 쉽습니다.<br/>→ 트리의 크기를 사전에 제한하는 파라미터 튜닝이 필요합니다.



## 사이킷런 Decision Tree Classifier의 파라미터

사이킷런에서 Deicision Tree를 이용하여 분류작업을 할 때, 사용되는 주요 파라미터들은 다음과 같습니다.

* **min_samples_split**

  - 노드를 분할하기 위한 최소한의 샘플 데이터수<br/>→ 과적합을 제어하는데 사용합니다. 값이 작을수록 분할노드가 많아져 과적합 가능성 증가
  - default : 2 <br/><br/>
  
  
  
* **min_samples_leaf**

  - 리프노드가 되기 위한 최소한의 샘플 데이터수<br/>→ 과적합을 제어하는데 사용합니다. 값이 작을수록 과적합 가능성 증가
  - default : 1 



* **max_features**

  - 최적의 분할을 위해 고려할 피처의 최대 갯수
  - default : None → 데이터셋의 모든 피처를 사용합니다.
  - int형으로 지정 → 피처 갯수  
  - float형으로 지정 → 전체 갯수의 일정 비율만큼 사용
  - `sqrt` 또는 `auto` → 전체 피처 중 √(피처 개수) 만큼 선정
  - `log2` : 전체 피처 중 log2(전체 피처 개수) 만큼 선정



* **max_depth**
  - 트리의 최대 깊이
  - default : None<br/>→ 완벽하게 클래스 값이 결정될 때까지 분할<br/>    또는 데이터 갯수가 min_samples_split보다 작아질 때까지 분할
  - 깊이가 깊어지면 과적합될 수 있으므로 적절히 제어 필요



* **max_leaf_nodes** 
  - 리프노드의 최대 갯수



## Decision Tree 모델의 시각화

#### 사이킷런의 붓꽃 데이터 셋을 이용한 Decision Tree 시각화

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# DecisionTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state = 20)

# 붓꽃 데이터셋을 로딩하고, 학습과 테스트 셋으로 분리
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, 
                                                    test_size = 0.2,
                                                    random_state = 20)
# DecisionTreeClassifier 학습
dt_clf.fit(X_train, y_train)
```

```
# 출력: 
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=20, splitter='best')
```

```python
from sklearn.tree import export_graphviz

# export_graphviz( )의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함
export_graphviz(dt_clf, out_file="tree.dot", 
                class_names = iris_data.target_names, 
                feature_names = iris_data.feature_names, 
                impurity=True, 
                filled=True)

print('===============max_depth의 제약이 없는 경우의 Decision Tree 시각화==================')
import graphviz
# 위에서 생성된 tree.dot 파일을 Graphiviz 가 읽어서 시각화
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
```

===============max_depth의 제약이 없는 경우의 Decision Tree 시각화==================

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-10-titanic-decision-tree-2.png?raw=true">

* petal_width (cm) <= 1.65 와 같이 조건이 있는 것은 자식노드를 만들기 위한 규칙조건입니다.<br/>이런 조건이 없는 것은 리프노드입니다.
* gini는 해당 서브 데이터셋 분포에서의 지니계수
* samples : 현 규칙에 해당하는 데이터 건수
* values = [ ] : 클레스 값 기반의 데이터 건수<br/>(이번 예제의 경우 0: Setosa, 1 : Veericolor, 2: Virginia)

```python
# DecicionTreeClassifier 생성 (max_depth = 3 으로 제한)
dt_clf = DecisionTreeClassifier(max_depth=3 ,random_state=20)
dt_clf.fit(X_train, y_train)

# export_graphviz( )의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함
export_graphviz(dt_clf, out_file="tree.dot", 
                class_names = iris_data.target_names, 
                feature_names = iris_data.feature_names, 
                impurity=True, 
                filled=True)

print('===============max_depth=3인 경우의 Decision Tree 시각화==================')
import graphviz
# 위에서 생성된 tree.dot 파일을 Graphiviz 가 읽어서 시각화
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
```

===============max_depth=3인 경우의 Decision Tree 시각화==================

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-10-titanic-decision-tree-3.png?raw=true">

max_depth를 3으로 제한한 결과 처음보다 간결한 형태의 트리가 만들어졌습니다. 동일 서브 데이터셋 내에 서로 다른 클래스 값이 있어도 더이상 분할하지 않았습니다.



```python
# DecicionTreeClassifier 생성 (min_samples_split=4로 상향)
dt_clf = DecisionTreeClassifier(min_samples_split=4 ,random_state=20)
dt_clf.fit(X_train, y_train)

# export_graphviz( )의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함
export_graphviz(dt_clf, out_file="tree.dot", 
                class_names = iris_data.target_names, 
                feature_names = iris_data.feature_names, 
                impurity=True, 
                filled=True)

print('===============min_samples_split=4인 경우의 Decision Tree 시각화==================')
import graphviz
# 위에서 생성된 tree.dot 파일을 Graphiviz 가 읽어서 시각화
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
```

===============min_samples_split=4인 경우의 Decision Tree 시각화==================

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-10-titanic-decision-tree-4.png?raw=true">



sample = 3인 경우에는 샘플 내 상이한 값이 있어도 더 이상 분할하지 않았습니다.



```python
# DecicionTreeClassifier 생성 (min_samples_leaf=4로 상향)
dt_clf = DecisionTreeClassifier(min_samples_leaf=4 ,random_state=20)
dt_clf.fit(X_train, y_train)

# export_graphviz( )의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함
export_graphviz(dt_clf, out_file="tree.dot", class_names = iris_data.target_names, 
                           feature_names = iris_data.feature_names, impurity=True, filled=True)

print('===============min_samples_leaf=4인 경우의 Decision Tree 시각화==================')
import graphviz
# 위에서 생성된 tree.dot 파일을 Graphiviz 가 읽어서 시각화
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
```

===============min_samples_leaf=4인 경우의 Decision Tree 시각화==================

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-10-titanic-decision-tree-5.png?raw=true">

자식이 없는 리프노드는 클래스 결정 값이 되는데 min_samples_leaf 는 리프노드가 될 수 있는 샘플 데이터의 최소 갯수를 지정합니다. 위와 비교해보면 기존에 샘플갯수가 3이하이던 리프노드들이 샘플갯수가 4가 되도로 변경되었음을 볼 수 있습니다. 결과적으로 처음보다 트리가 간결해졌습니다.



## Feature Importance 시각화

학습을 통해 규칙을 정하는데 있어 각 피처의 중요도를 DecisionTreeClassifier 객체의 **feature_importance_** 속성으로 확인할 수 있습니다. 기본적으로 ndarray 형태로 값을 반환하며 피처 순서대로 값이 할당됩니다.

```python
import seaborn as sns
import numpy as np
%matplotlib inline

# feature importance 추출
print('Feature Importances:\n{0}\n'.format(np.round(dt_clf.feature_importances_, 3)))

# feature별 feature importance 매핑
for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
  print('{0}: {1:.3f}'.format(name, value))

# feature importance 시각화
sns.barplot(x = dt_clf.feature_importances, y = iris_data.feature_names)
```

```
# 출력:
Feature Importances:
[0.    0.016 0.548 0.436]

sepal length (cm): 0.000
sepal width (cm): 0.016
petal length (cm): 0.548
petal width (cm): 0.436
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-10-titanic-decision-tree-6.png?raw=true">



## Decision Tree의 과적합(Overfitting)

#### 임의의 데이터 셋를 통한 과적합 문제 시각화

```python
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

plt.title("3 Class values with 2 Features Sample Data Creation")

# 2차원 시각화를 위해 피처 2개, 클래스는 3가지 유형의 샘플 데이터 생성
X_features, y_labels = make_classification(n_features = 2, 
                                           n_redundant = 0,
                                           n_informative = 2,
                                           n_classes = 3,
                                           n_clusters_per_class = 1,
                                           random_state = 0)

# 그래프 형태로 2개의 피처로 2차원 좌표 시각화, 각 클래스 값은 다른 색으로 표시
plt.scatter(X_features[:, 0], X_features[:, 1],
            marker = 'o',
            c = y_labels,
            s = 25, 
            edgecolor = 'k',
            cmap='rainbow')
plt.show()
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-10-titanic-decision-tree-7.png?raw=true">

우선 트리 생성 시 파라미터를 디폴트로 놓고, 데이터가 어떻게 분류되는지 확인해보겠습니다.

```python
# Classifier의 Decision Boundary를 시각화 하는 함수
def visualize_boundary(model, X, y):
    fig, ax = plt.subplots()
    
    # 학습 데이타 scatter plot으로 나타내기
    ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim_start , xlim_end = ax.get_xlim()
    ylim_start , ylim_end = ax.get_ylim()
    
    # 호출 파라미터로 들어온 training 데이타로 model 학습 . 
    model.fit(X, y)
    # meshgrid 형태인 모든 좌표값으로 예측 수행
    xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),
                         np.linspace(ylim_start,ylim_end, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # contourf() 를 이용하여 class boundary 를 visualization 수행. 
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='rainbow', clim=(y.min(), y.max()),
                           zorder=1)
```

```python
# 특정한 트리 생성에 제약이 없는(전체 default 값) Decision Tree의 학습과 결정 경계 시각화
dt_clf = DecisionTreeClassifier()
visualize_boundary(dt_clf, X_features, y_labels)
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-10-titanic-decision-tree-8.png?raw=true">

위의 경우 매우 얇은 영역으로 나타난 부분은 이상치에 해당하는데, 이런 이상치까지 모두 분류하기 위해 분할한 결과 결정 기준 경계가 많아졌습니다. 이런 경우 조금만 형태가 다른 데이터가 들어와도 정확도가 매우 떨어지게 됩니다.

```python
# min_samples_leaf = 6 으로 설정한 Decision Tree의 학습과 결정 경계 시각화
dt_clf = DecisionTreeClassifier(min_samples_leaf=6)
visualize_boundary(dt_clf, X_features, y_labels)
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/2020-03-10-titanic-decision-tree-9.png?raw=true">

Default 값으로 실행한 앞선 경우보다 이상치에 크게 반응하지 않으면서 일반화된 분류 규칙에 의해 분류되었음을 확인할 수 있습니다.



#### Decision Tree의 과적합을 줄이기 위한 파라미터 튜닝

(1) **max_depth** 를 줄여서 트리의 깊이 제한<br/>(2) **min_samples_split** 를 높여서 데이터가 분할하는데 필요한 샘플 데이터의 수를 높이기<br/>(3) **min_samples_leaf** 를 높여서 말단 노드가 되는데 필요한 샘플 데이터의 수를 높이기<br/>(4) **max_features** 를 제한하여 분할을 하는데 고려하는 피처의 수 제한



---------

###### Reference

- 파이썬 머신러닝 완벽가이드
- https://bigwhalelearning.wordpress.com/2014/11/27/77/
