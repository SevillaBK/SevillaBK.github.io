---
layout: post
title: '[케라스] 딥러닝 모델의 시각화'
excerpt: 딥러닝모델 시각화에 대한 간단한 소개
category: ML & DL
tags:
  - 딥러닝
  - Keras
  - visualization
  - 시각화


---

김태영님의 `블록과 함께 하는 파이썬 딥러닝 케라스` 를 학습하며 정리하는 내용입니다.

--------------

케라스에서는 레이어를 모델에 순차적으로 저장하고 있습니다. 그리고 이렇게 만들어진 딥러닝 모델을 시각화해서 볼 수 있습니다. 이 기능을 이용하면, 내가 만든 모델을 구성하는 레이어가 어떤 구조로 이루어져있는지 시각화해서 볼 수 있습니다. 그리고 다른 사람들이 만든 모델들도 이 기능을 사용하여 모델의 구조를 손쉽게 알 수 있습니다.

이번 포스팅에서는 모델을 시각화해서 볼 수 있는 두 방법을 정리했습니다.

우선 이에 앞서 MNIST 데이터를 이용한 숫자분류 모델을 다시 구현해보겠습니다.

```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불어괴
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 데이터셋 전처리
X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0

# 원핫인코딩 처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 데이터셋 분리
X_val = X_train[18000:] # 훈련 셋의 70%를 검증셋으로 사용
X_train = X_train[:18000]
y_val = y_train[18000:]
y_train = y_train[:18000]

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units = 64, input_dim = 28 * 28, activation = 'relu'))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# 4. 모델 학습시키기
model.fit(X_train, y_train, epochs = 5, batch_size = 32, validation_data = (X_val, y_val))
```



## 1. model.summary()

`model.summary()` 기능을 사용하면 model의 구조를 쉽게 볼 수 있습니다.

```python
model.summary()
```

```
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_10 (Dense)             (None, 64)                50240     
_________________________________________________________________
dense_11 (Dense)             (None, 32)                2080      
_________________________________________________________________
dense_12 (Dense)             (None, 10)                330       
=================================================================
Total params: 52,650
Trainable params: 52,650
Non-trainable params: 0
```



## 2. Graphviz 로 시각화하기

graphviz 를 통해 시각화하면 그림을 통해 보기 쉽게 시각화할 수 있습니다.

```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
%matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/keras-visualization.png?raw=true">



---------

###### Reference

- 블록과 함께 하는 파이썬 딥러닝 케라스(김태영 저)
- https://gaussian37.github.io/dl-keras-케라스-모델-시각화/
