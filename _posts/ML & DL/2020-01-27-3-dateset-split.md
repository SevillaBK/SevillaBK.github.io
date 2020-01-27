---
layout: post
title: '데이터셋의 분리-train, test, validation'
excerpt: ML/DL 모델 학습을 위한 데이터셋 관련 개념 학습
category: ML & DL
tags:
  - 사이킷런
  - train_test_split
  - model_selection
---



머신러닝/딥러닝 모델을 학습시키기 위해서는 데이터셋이 필요합니다. 이 데이터셋을 어떻게 구성하고, 활용하는지에 대해 알아보겠습니다.



## train, valiation, test 데이터셋

머신러닝, 딥러닝 모델로 데이터를 학습/예측하기 위해 위해 데이터셋을 크게 세 가지를 분리합니다.

* `학습 데이터셋(training set)` : 모델의 학습을 위해 사용되는 데이터입니다. 
* `검증 데이터셋(validation set)` : 학습과정에서 모델의 성능을 확인하고, 하이퍼 파라미터를 튜닝하는데 사용됩니다. 여러 하이퍼 파라미터로 생성된 모델 중 어떤 것이 성능이 좋은지 평가합니다.
* `테스트 데이터셋(test set)` : 생성된 모델의 예측성능을 평가하는데 사용됩니다. 학습과 튜닝에는 이용되지 않고, 미래의 타겟 값이 관측되지 않은 데이터라 가정하여 예측이 잘 되는지 평가하는데 사용됩니다.


<img src="https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/train-valid-test.png?raw=true" width="90%"></img>



주어진 전체 데이터를 바로 다 모델에 학습시키지 않고, 분할해서 학습시키는 이유는 `과적합(Overfitting)` 을 방지하기 위해서 입니다. 실제로 모델을 활용하기 위해서는 학습하지 않은 데이터에 대해 예측을 잘해야 합니다. 하지만 주어진 데이터만 너무 학습한 경우, 조금만 다른 패턴의 데이터가 나오면 모델의 성능이 떨어지게 됩니다. 때문에, 검증데이터셋을 활용하여 학습 중인 모델이 과적합되었는지 확인하여, 학습을 조기에 종료(early stopping)하고, 하이퍼 파라미터를 튜닝하는 등의 조치를 취합니다. 

그 후, 학습셋과 검증데이터셋 전체를 다시 학습하여 테스트셋에 대해 예측을 수행합니다. 이를 통해 학습하지 않은 데이터에 대해서도 모델의 성능이 잘 나오는지 확인합니다.



## 사이킷런을 활용한 train/test 세트 분리

사이킷런의 `model_selection` 의 `train_test_split` 함수를 활용하면 손쉽게 데이터셋을 train 과 test 세트로 분리할 수 있습니다.

#### train_test_split( ) 함수의 파라미터

* `test_size` : 전체 데이터셋에서 테스트 데이터셋 크기를 얼마로 샘플링할 것인지 결정합니다. (기본값 : 0.25)
* `train_size` : 전체 데이터셋에서 학습 데이터셋 크기를 얼마로 샘플링할 것인지 결정합니다. 다만 test_size를 주로 활용하기 때문에, 잘 쓰이지는 않습니다.
* `shuffle` : 데이터를 분리하기 전에 데이터를 섞을지 결정하는 파라미터입니다. 데이터를 분산시켜 보다 효율적인 학습/테스트 데이터 세트를 만드는데 사용합니다.(기본값: True)
* `random_state` : 난수값을 지정하면 여러 번 다시 수행해도 동일한 결과가 나오게 해줍니다.



train_test_split의 반환 값은 튜플형태로 `(1) 학습 데이터셋, (2) 테스트 데이터셋, (3) 학습 데이터셋의 레이블, (4) 테스트 데이터셋` 의 순서로 되어있습니다.

```python
# 예제 데이터셋 불러오기
from sklearn.datasets import load_iris
dataset = load_iris()
data = dataset['data'] # 데이터 전체(레이블제외)
label = dataset['target'] # 레이블 데이터
print('원 데이터셋의 shape: ',data.shape)
print('레이블 데이터셋의 shape: 'label.shape)

# train_test_split 함수 불러오기
from sklearn.model_selection import train_test_split

# 데이터셋 나누기
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.25, shuffle = True, random_state = 21)

# 나누어진 데이터셋 확인
print('학습셋의 shape: ', X_train.shape)
print('검증셋의 shape: ', X_test.shape)
print('학습셋 label의 shape: ', y_train.shape)
print('검증셋 label의 shape: ', y_test.shape)
```

```
# 출력:
원 데이터셋의 shape: (150, 4)
레이블 데이터셋의 shape: (150)
학습셋의 shape: (112, 4)
검증셋의 shape: (38, 4)
학습셋 label의 shape: (112,)
검증셋 label의 shape: (38,)
```


---------

###### Reference

- 파이썬 머신러닝 완벽가이드
- 블록과 함께하는 파이썬 딥러닝 케라스
