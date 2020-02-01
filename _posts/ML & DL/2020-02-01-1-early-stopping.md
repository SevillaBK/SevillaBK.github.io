---
layout: post
title: '[케라스] 딥러닝 모델 학습 조기 종료시키기(early stopping)'
excerpt: 딥러닝모델이 과적합되기 전에 모델 학습을 조기 종료시켜보자.
category: ML & DL
tags:
  - 딥러닝
  - Keras
  - EarlyStopping

---

김태영님의 `블록과 함께 하는 파이썬 딥러닝 케라스` 를 학습하며 정리하는 내용입니다.

--------------



## 과적합되는 모델의 학습과정 시각화

과적합되는 모델을 만들어 학습과정을 시각화해보겠습니다.

```python
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

np.random.seed(10)

# 1. 데이터셋 준비하기
# 훈련셋과 시험셋 로딩
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 훈련셋과 검증셋 분리
X_val = X_train[50000:]
Y_val = Y_train[50000:]
X_train = X_train[:50000]
Y_train = Y_train[:50000]

X_train = X_tran.reshape(50000, 784).astype('float32') / 255.0
X_val = X_val.reshape(50000, 784).astype('float32') / 255.0
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
```

```python
# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units = 2, input_dim = 28 * 28, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

# 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(X_train, Y_train, epochs = 3000, batch_size = 10, 	
                 validation_data=(X_val, Y_val))
```

```
Train on 700 samples, validate on 300 samples
Epoch 1/3000
700/700 [==============================] - 0s 297us/step - loss: 2.3006 - accuracy: 0.1286 - val_loss: 2.2728 - val_accuracy: 0.1733
Epoch 2/3000
700/700 [==============================] - 0s 78us/step - loss: 2.2392 - accuracy: 0.1757 - val_loss: 2.2331 - val_accuracy: 0.1900
Epoch 3/3000
700/700 [==============================] - 0s 74us/step - loss: 2.1800 - accuracy: 0.2214 - val_loss: 2.1829 - val_accuracy: 0.2400
...
...
Epoch 2998/3000
700/700 [==============================] - 0s 64us/step - loss: 0.3907 - accuracy: 0.8929 - val_loss: 4.1276 - val_accuracy: 0.3833
Epoch 2999/3000
700/700 [==============================] - 0s 69us/step - loss: 0.3903 - accuracy: 0.8943 - val_loss: 4.1566 - val_accuracy: 0.3900
Epoch 3000/3000
700/700 [==============================] - 0s 67us/step - loss: 0.3905 - accuracy: 0.8943 - val_loss: 4.1411 - val_accuracy: 0.3900
```

```python
# 5. 모델 학습과정 표시하기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label = 'train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label = 'val loss')
acc_ax.plot(hist.history['accuracy'], 'b', label = 'train accuracy')
acc_ax.plot(hist.history['val_accuracy'], 'g', label = 'val accuracy')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_xlabel('accuracy')

loss_ax.legend(loc = 'upper left')
acc_ax.legend(loc = 'lower left')

plt.show()
```

<img src = 'https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/learning-vis3.png?raw=true'>

검증셋의 손실값(val_loss)를 보면 처음에 감소하다가 200에포크 정도부터 계속 증가하는 것을 볼 수 있습니다. 이 때, 과적합이 발생한 것입니다.

```python
# 6. 모델 사용하기
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size= 32)

print(' ')
print('loss: ' , loss_and_metrics[0])
print('accuracy: ', loss_and_metrics[1])
```

```
10000/10000 [==============================] - 0s 9us/step
 
loss: 4.143502366256714
accuracy: 0.4408000111579895
```



## 학습 조기 종료 시키기(EarlyStopping 적용하기)

과적합을 방지하기 위해서는 `EarlyStopping` 이라는 콜백함수를 사용하여 적절한 시점에 학습을 조기 종료시켜야 합니다. fit 함수에서 `EalryStopping` 콜백함수를 지정하는 방법은 아래와 같습니다.

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping()
model.fit(X_train, Y_train, epoch = 1000, callbacks = [early_stopping])
```

아래와 같이 설정을 하면, 에포크를 1000으로 지정하더라도 콜백함수에서 설정한 조건을 만족하면 학습을 조기 종료시킵니다. `EarlyStopping` 조건 설정을 위한 파라미터를 알아보겠습니다. 

```python
EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 0, mode = 'auto')
```

* `monitor` : 학습 조기종료를 위해 관찰하는 항목입니다. val_loss 나 val_accuracy 가 주로 사용됩니다. (default :  **val_loss**)
* `min_delta` : 개선되고 있다고 판단하기 위한 최소 변화량을 나타냅니다. 만약 변화량이 min_delta 보다 적은 경우에는 개선이 없다고 판단합니다. (default = **0**)
* `patience` : 개선이 안된다고 바로 종료시키지 않고, 개선을 위해 몇번의 에포크를 기다릴지 설정합니다. (default = **0**)
* `mode` : 관찰항목에 대해 개선이 없다고 판단하기 위한 기준을 설정합니다. monitor에서 설정한 항목이 val_loss 이면 값이 감소되지 않을 때 종료하여야 하므로 min 을 설정하고, val_accuracy 의 경우에는 max를 설정해야 합니다. (default = **auto**)
  - `auto` : monitor에 설정된 이름에 따라 자동으로 지정합니다.
  - `min` : 관찰값이 감소하는 것을 멈출 때,  학습을 종료합니다.
  - `max`: 관찰값이 증가하는 것을 멈출 때, 학습을 종료합니다.

```python
# 데이터셋 구성은 위와 동일합니다.

# 2. 모델 구설하기
model = Sequential()
model.add(Dense(units = 2, input_dim = 28*28, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

# 3. 모델 엮기
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# 4. 모델 학습시키기 : 이 부분의 코드가 앞선 예제와 달라집니다.
from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping()
hist = model.fit(X_train, Y_train,
                 epochs = 3000, batch_size = 10, 
                 validation_data = (X_val, Y_val),
                 callbacks = [early_stopping])
```

```
Train on 700 samples, validate on 300 samples
Epoch 1/3000
700/700 [==============================] - 0s 192us/step - loss: 2.2635 - accuracy: 0.2129 - val_loss: 2.2022 - val_accuracy: 0.2000
Epoch 2/3000
700/700 [==============================] - 0s 97us/step - loss: 2.1775 - accuracy: 0.1814 - val_loss: 2.1271 - val_accuracy: 0.1867
Epoch 3/3000
700/700 [==============================] - 0s 76us/step - loss: 2.1053 - accuracy: 0.1871 - val_loss: 2.0634 - val_accuracy: 0.2167
...
...
Epoch 48/3000
700/700 [==============================] - 0s 76us/step - loss: 1.2442 - accuracy: 0.5843 - val_loss: 1.2863 - val_accuracy: 0.5367
Epoch 49/3000
700/700 [==============================] - 0s 77us/step - loss: 1.2354 - accuracy: 0.5971 - val_loss: 1.2831 - val_accuracy: 0.5333
Epoch 50/3000
700/700 [==============================] - 0s 76us/step - loss: 1.2248 - accuracy: 0.6057 - val_loss: 1.2932 - val_accuracy: 0.5133
```

3000번의 에포크를 설정했지만 50번째 에포크에서 학습이 종료되었습니다.
이 경우, 모델의 학습이 어떻게 진행이 되었는지 살펴보겠습니다.

```python
# 5. 모델 학습과정 표시하기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label = 'train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label = 'val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label = 'train accuracy')
acc_ax.plot(hist.history['val_accuracy'], 'g', label = 'val accuracy')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc = 'upper left')
acc_ax.legend(loc = 'lower left')

plt.show()
```

<img src = 'https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/learning-vis4.png?raw=true'>

위의 그래프를 보면 모델이 학습을 진행하다가 val_loss가 살짝 증가하는 순간 바로 학습을 종료시켰음을 알 수 있습니다. 하지만 모델은 학습하면서 loss 값이 증감을 여러 번 반복하기 때문에, 이후의 에포크에서는 loss가 향상되었을 수도 있습니다. 때문에 이렇게 바로 종료시켜 버리면 모델은 과소적합 상태가 될 수 있습니다.

`patience` 파라미터를 사용하여 val_loss가 1번 증가할 때 바로 학습을 멈추지 않고, 30번의 에포크를 기다리고 학습을 종료하는 모델의 학습과정을 시각화해보겠습니다.

```python
# 데이터셋 구성은 위와 동일합니다.

# 2. 모델 구설하기
model = Sequential()
model.add(Dense(units = 2, input_dim = 28*28, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

# 3. 모델 엮기
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# 4. 모델 학습시키기 : 이 부분의 코드가 앞선 예제와 달라집니다.
from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(patience = 30)
hist = model.fit(X_train, Y_train,
                 epochs = 3000, batch_size = 10, 
                 validation_data = (X_val, Y_val),
                 callbacks = [early_stopping])
```

```
Train on 700 samples, validate on 300 samples
Epoch 1/3000
700/700 [==============================] - 0s 192us/step - loss: 2.2698 - accuracy: 0.1371 - val_loss: 2.2143 - val_accuracy: 0.1833
Epoch 2/3000
700/700 [==============================] - 0s 84us/step - loss: 2.1845 - accuracy: 0.1871 - val_loss: 2.1394 - val_accuracy: 0.2033
Epoch 3/3000
700/700 [==============================] - 0s 76us/step - loss: 2.1214 - accuracy: 0.1986 - val_loss: 2.0982 - val_accuracy: 0.2367
...
...
Epoch 143/3000
700/700 [==============================] - 0s 68us/step - loss: 0.9260 - accuracy: 0.7071 - val_loss: 1.3358 - val_accuracy: 0.5467
Epoch 144/3000
700/700 [==============================] - 0s 69us/step - loss: 0.9241 - accuracy: 0.7043 - val_loss: 1.3341 - val_accuracy: 0.5567
Epoch 145/3000
700/700 [==============================] - 0s 68us/step - loss: 0.9219 - accuracy: 0.7014 - val_loss: 1.3297 - val_accuracy: 0.5433
```

`patience` 파라미터 설정을 바꾸니 앞선 모델보다 약 100번의 에포크를 더 학습하여 145번째에서 학습이 종료되었습니다. 

```python
# 5. 모델 학습과정 표시하기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label = 'train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label = 'val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label = 'train accuracy')
acc_ax.plot(hist.history['val_accuracy'], 'g', label = 'val accuracy')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc = 'upper left')
acc_ax.legend(loc = 'lower left')

plt.show()
```

<img src = 'https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/ML&DL/learning-vis5.png?raw=true'>

```python
# 6. 모델 사용하기
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size = 32)

print(' ')
print('loss: ', loss_and_metrics[0])
print('accuracy: ', loss_and_metrics[1])
```

```
10000/10000 [==============================] - 0s 10us/step
 
loss:  1.2959346857070924
accuracy:  0.5569999814033508
```

앞서 바로 종료시켰던 모델보다 loss와 accuracy 모두 향상되었습니다.
이렇게 적절한 조기종료는 모델의 성능을 보다 향상시킬 수 있기 때문에 모델의 학습과정을 살펴보고 어느 시점에서 학습을 종료시킬지 판단해야 합니다.



---------

###### Reference

- 블록과 함께 하는 파이썬 딥러닝 케라스(김태영 저)
