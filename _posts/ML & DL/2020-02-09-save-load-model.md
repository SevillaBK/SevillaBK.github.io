---
layout: post
title: '[케라스] 딥러닝 모델을 저장하고 불러오기'
excerpt: 학습한 모델을 파일로 저장하고, 불러와서 다시 사용하는 간단한 방법 소개
category: ML & DL
tags:
  - 딥러닝
  - Keras

---

김태영님의 `블록과 함께 하는 파이썬 딥러닝 케라스` 를 학습하며 정리하는 내용입니다.

--------------

케라스 딥러닝 모델은 크게 모델 아키텍처와 모델 가중치로 구성되어있습니다. 모델 아키텍처는 모델이 어떤 층으로 어떻게 쌓여있는지에 대한 모델 구성이 정의되어 있고, 모델 가중치는 처음에는 임의의 값으로 초기화되어 있지만, 데이터를 학습하면서 갱신됩니다. 케라스에서는 `save()` 함수를 이용하여 모델 아키텍처와 가중치를 h5 형식으로 저장할 수 있습니다.

```python
# 케라스 모델 저장하기
model.save('저장할이름.h5')
```

MNIST 데이터셋을 학습한 모델을 저장해보겠습니다.

```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터셋 전처리
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# 원핫인코딩 (one-hot encoding) 처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 훈련셋과 검증셋 분리
x_val = x_train[:42000] # 훈련셋의 30%를 검증셋으로 사용
x_train = x_train[42000:]
y_val = y_train[:42000] # 훈련셋의 30%를 검증셋으로 사용
y_train = y_train[42000:]

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))

# 5. 모델 평가하기
print('')
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('loss_and_metrics : ' + str(loss_and_metrics))

# 6. 모델 저장하기
from keras.models import load_model
model.save('mnist_mlp_model.h5')
```

```
# 출력:
Train on 18000 samples, validate on 42000 samples
Epoch 1/5
18000/18000 [==============================] - 1s 61us/step - loss: 1.1149 - accuracy: 0.7234 - val_loss: 0.6306 - val_accuracy: 0.8414
Epoch 2/5
18000/18000 [==============================] - 1s 55us/step - loss: 0.5046 - accuracy: 0.8709 - val_loss: 0.4702 - val_accuracy: 0.8731
Epoch 3/5
18000/18000 [==============================] - 1s 56us/step - loss: 0.4076 - accuracy: 0.8879 - val_loss: 0.4091 - val_accuracy: 0.8863
Epoch 4/5
18000/18000 [==============================] - 1s 56us/step - loss: 0.3630 - accuracy: 0.8982 - val_loss: 0.3780 - val_accuracy: 0.8936
Epoch 5/5
18000/18000 [==============================] - 1s 61us/step - loss: 0.3353 - accuracy: 0.9071 - val_loss: 0.3538 - val_accuracy: 0.9012

10000/10000 [==============================] - 0s 11us/step
loss_and_metrics : [0.3257985237181187, 0.909600019454956]
```

위의 코드를 실행하면 작업 폴더에  `mnist_mlp_model.h5` 라는 파일이 생성됩니다. 이 파일에는 아래와 같은 정보들이 저장되어 있습니다.

* 모델을 재구성하기 위한 모델의 구성정보
* 모델을 구성하는 각 뉴런들의 가중치
* 손실함수, 최적화 등의 학습설정
* 재학습을 할 수 있도록 마지막 학습상태

이렇게 저장된 파일을 `load_model()` 함수를 사용하여 불러와 다시 활용할 수 있습니다.

```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax

# 1. 실무에 사용할 데이터 준비하기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_test = np_utils.to_categorical(y_test)
xhat_idx = np.random.choice(x_test.shape[0], 5)
xhat = x_test[xhat_idx]

# 2. 모델 불러오기
from keras.models import load_model
model = load_model('mnist_mlp_model.h5')

# 3. 모델 사용하기
yhat = model.predict_classes(xhat)

for i in range(5):
    print('True: ', argmax(y_test[xhat_idx[i]]), ', Predict: ', yhat[i]) 
```

```
# 출력:
True:  1 , Predict:  1
True:  1 , Predict:  1
True:  3 , Predict:  3
True:  2 , Predict:  2
True:  2 , Predict:  0
```

모델을 구성하는 작업을 따로 거치지 않고, 저장된 모델을 불러와 결과를 잘 출력하는 것으 볼 수 있습니다.

---------

###### Reference

- 블록과 함께 하는 파이썬 딥러닝 케라스(김태영 저)
