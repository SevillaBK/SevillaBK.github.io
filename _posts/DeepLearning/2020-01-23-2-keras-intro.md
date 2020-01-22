---
layout: post
title: '[Keras] 케라스 딥러닝 모델 작성순서'
excerpt: 케라스로 딥러닝 모델을 만들어 나가는 기본 과정을 대략적으로 알아보자
category: DeepLearning
tags:
  - DeepLearning
  - Keras

---

김태영님의 `블록과 함께 하는 파이썬 딥러닝 케라스` 를 학습하며 정리하는 내용입니다.

## 케라스로 딥러닝 모델 만들기

케라스로 딥러닝 모델을 만들기 위해서는 아래와 같은 기본적인 순서로 코드를 작성하게 됩니다.



#### 1. 데이터셋 생성하기

* 원본 데이터를 불러오거나 시뮬레이션 등을 통해 데이터를 생성합니다.
* 이 데이터들로부터 train, validation, test 데이터 셋을 생성합니다.
* 딥러닝 모델이 데이터를 원활히 읽고, 학습하기 위한 포맷 변환을 합니다.

#### 2. 모델 구성하기

* Sequence 모델을 생성한 뒤 필요한 레이어를 추가하여 구성합니다.
* 복잡한 모델이 필요할 때는 케라스에서 활용가능한 여러 함수들을 이용합니다.

#### 3. 모델 학습과정 설정하기

* 학습하기 전에 학습에 대한 설정을 수행합니다.(손실함수, 최적화 방법 등)
* 케라스에서는 `compile( )` 함수를 이용하여 학습과정을 설정합니다.

#### 4. 모델 학습

* 위에서 구성한 모델을 `fit( )` 함수를 이용하여 train 데이터 셋을 학습시킵니다.

#### 5. 학습과정 살펴보기

* 모델 학습 시, train, validation 데이터셋의 손실 및 정확도 등을 측정합니다.
* 반복횟수에 따른 손실 및 정확도의 추이를 보며 학습 상황을 판단합니다.

#### 6. 모델 평가하기

* `evaluate( )` 함수로 test 데이터셋으로 모델을 평가합니다.

#### 7. 모델 사용하기

* `predict( )` 함수로 임의의 입력값에 대한 모델의 출력값을 얻습니다.



#### MNIST 데이터셋을 통한 분류 모델의 구현 예시

가로세로 픽셀이 28 x 28 인 이미지를 입력받아 784 (28 x 28) 벡터로 구성한 다음, 이를 학습/평가하는 코드 예시입니다.

```python
# 0. 필요한 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# 1. 데이터셋 생성하기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=5, batch_size=32)

# 5. 학습과정 살펴보기
print('## training loss and acc ##')
print(hist.history['loss'])
print(hist.history['accuracy'])

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

# 7. 모델 사용하기
xhat = x_test[0:1]
yhat = model.predict(xhat)
print('## yhat ##')
print(yhat)
```

```
# 결과: 
Epoch 1/5
60000/60000 [==============================] - 1s 22us/step - loss: 0.6904 - accuracy: 0.8220
Epoch 2/5
60000/60000 [==============================] - 1s 22us/step - loss: 0.3511 - accuracy: 0.9025
Epoch 3/5
60000/60000 [==============================] - 1s 22us/step - loss: 0.3037 - accuracy: 0.9139
Epoch 4/5
60000/60000 [==============================] - 1s 22us/step - loss: 0.2756 - accuracy: 0.9222
Epoch 5/5
60000/60000 [==============================] - 1s 22us/step - loss: 0.2542 - accuracy: 0.9288
## training loss and acc ##
[0.6903821740627288, 0.3510775326768557, 0.30374961163202924, 0.2755953017512957, 0.2541987054069837]
[0.82205, 0.90245, 0.91386664, 0.92216665, 0.92878336]
10000/10000 [==============================] - 0s 13us/step
## evaluation loss and_metrics ##
[0.2411062575995922, 0.9319000244140625]
## yhat ##
[[1.5453045e-04 1.0505996e-07 3.3818383e-04 2.3072090e-03 3.8662820e-06
  1.4190700e-04 1.6966163e-07 9.9561751e-01 4.3625954e-05 1.3929385e-03]]
```





---------

###### Reference

- 블록과 함께 하는 파이썬 딥러닝 케라스(김태영 저)
