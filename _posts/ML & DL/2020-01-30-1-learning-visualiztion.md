---
layout: post
title: '[케라스] matplotlib을 활용하여 학습과정 시각화하기'
excerpt: 딥러닝모델의 학습과정을 matplotlib을 활용해 확인해보자
category: ML & DL
tags:
  - 딥러닝
  - Keras

---

김태영님의 `블록과 함께 하는 파이썬 딥러닝 케라스` 를 학습하며 정리하는 내용입니다.

--------------



케라스로 딥러닝 모델을 학습하면서 fit 함수가 화면에 찍어주는 로그를 많이 보게 됩니다. 로그를 통해 학습이 제대로 되고 있는지, 학습을 조기 종료할지 등을 판단할 수 있습니다. 로그 자체를 볼 수도 있지만 그래프를 통해 시각화한다면 학습 정도와 추이를 직관적으로 파악할 수 있습니다.



## History 객체 사용하기

케라스에서 모델을 학습시킬 때 사용하는 fit 함수는 `history` 객체를 반환합니다.<br/>이 객체는 아래의 정보를 가지고 있습니다.

* `loss` : 각 에포크마다의 학습 손실값
* `accuracy` : 각 에포크마다의 학습 정확도
* `val_loss` : 각 에포크마다의 검증 손실값
* `val_accuracy` : 각 에포크마다의 검증 정확도

위의 정보들은 아래와 같이 개별적으로 확인할 수 있습니다.

```python
# history 객체에서 학습 상태 정보 불러오기 예시
hist = model.fit(X_train, Y_train, epochs = 1000, batch_size = 10, validation = X_val, Y_val)

print(hist.history['loss'])
print(hist.history['accuracy'])
print(hist.history['val_loss'])
print(hist.history['val_accuracy'])
```

하지만 위 코드와 같이 사용할 경우, 각 에포크마다의 값이 배열형태로 저장이 되어 있어 직관적으포 파악하기가 어렵습니다.

matplotlib 를 활용하면 하나의 그래프로 이 배열값들을 쉽게 표시할 수 있습니다.<br/>MNIST 데이터셋을 다층 퍼셉트론 모델로 학습시키는 간단한 예제로 학습과정을 시각화 해보겠습니다.

```python
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# 1. 데이터셋 준비하기

# 훈련셋과 시험셋 로딩
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 훈련셋과 검증셋 분리
X_val = X_train[50000:]
Y_val = Y_train[50000:]
X_train = X_train[:50000]
Y_train = Y_train[:50000]

X_train = X_train.reshape(50000, 784).astype('float32') / 255.0
X_val = X_val.reshape(10000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0

# 훈련셋, 검증셋 고르기
train_rand_idxs = np.random.choice(50000, 700)
val_rand_idxs = np.random.choice(10000, 300)

X_train = X_train[train_rand_idxs]
Y_train = Y_train[train_rand_idxs]
X_val = X_val[val_rand_idxs]
Y_val = Y_val[val_rand_idxs]

# 라벨링 전환
Y_train = np_utils.to_categorical(Y_train)
Y_val = np_utils.to_categorical(Y_val)
Y_test = np_utils.to_categorical(Y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units = 5, input_dim = 28 * 28, activation = 'relu'))
model.add(Dense(units = 15, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

# 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(X_train, Y_train, epochs = 1000, batch_size = 10, validation_data=(X_val, Y_val))

# 5. 모델 학습과정 표시하기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label = 'train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label = 'val loss')
acc_ax.plot(hist.history['accuracy'], 'b', label = 'train accuracy')
acc_ax.plot(hist.history['val_accuracy'], 'g', label = 'valid accuracy')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

```

```
Train on 700 samples, validate on 300 samples
Epoch 1/1000
700/700 [==============================] - 0s 180us/step - loss: 2.2277 - accuracy: 0.1586 - val_loss: 2.1368 - val_accuracy: 0.2300
Epoch 2/1000
700/700 [==============================] - 0s 76us/step - loss: 2.1216 - accuracy: 0.2443 - val_loss: 2.0563 - val_accuracy: 0.2833
Epoch 3/1000
700/700 [==============================] - 0s 74us/step - loss: 2.0542 - accuracy: 0.2643 - val_loss: 1.9933 - val_accuracy: 0.3000
...
...
Epoch 998/1000
700/700 [==============================] - 0s 68us/step - loss: 0.3563 - accuracy: 0.9157 - val_loss: 3.0473 - val_accuracy: 0.5200
Epoch 999/1000
700/700 [==============================] - 0s 70us/step - loss: 0.3571 - accuracy: 0.9086 - val_loss: 3.0239 - val_accuracy: 0.5333
Epoch 1000/1000
700/700 [==============================] - 0s 71us/step - loss: 0.3583 - accuracy: 0.9100 - val_loss: 3.0788 - val_accuracy: 0.5233
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/learning-vis.png?raw=true">

위의 그림과 같이 matplotlib을 이용하면 한 눈에 학습의 추이를 파악할 수 있습니다.<br/>학습셋에 대한 loss와 accuracy는 계속 향상되지만 검증셋에 대한 loss와 accuracy는 각각 100번째 에포크, 200번째 에포크 정도에서 떨어지는(오버피팅되는) 현상을 볼 수 있습니다.


---------

###### Reference

- 블록과 함께 하는 파이썬 딥러닝 케라스(김태영 저)
