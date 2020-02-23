---
layout: post
title: '[케라스] 컨볼루션 신경망(CNN) 모델 만들어보기'
excerpt: 간단한 이미지 예제 데이터로 CNN 모델 만들고 학습시키기
category: ML & DL
tags:
  - 딥러닝
  - 케라스
  - CNN

---

김태영님의 `블록과 함께 하는 파이썬 딥러닝 케라스` 를 공부하며 옮겨온 내용입니다.

--------------

이번 포스팅에서는 직접 손으로 삼각형, 사각형, 원을 그려 이미지로 저장한 다음 이를 분류하는 컨볼루션 신경망 모델을 만들어보겠습니다. 

* 문제형태 : 다중 클래스 분류
* 입력 : 손으로 그린 삼각형, 사각형, 원 이미지
* 출력 : 삼각형, 사각형, 원일 확률을 나타내는 벡터

```python
# 필요한 패키지 불러오기 + 동일한 결과를 반복할 수 있도록 랜덤시드 지정
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# 랜덤 시드 고정시키기
np.random.seed(3)
```



## 데이터 준비하기

간단한 그림 툴을 이용하여 24 x 24(픽셀) 크기의 삼각형, 사각형, 원 이미지 png 파일을 각각 20개 만들어 저장했습니다. 각 15개는 훈련에 이용하고, 5개는 테스트에 사용해보겠습니다.

이미지 데이터는 train, test 폴더 아래 circle, rectengular, triangle 폴더에 저장했습니다.

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-2-23_CNN_Tutorial_1.png?raw=true">



## 데이터셋 생성하기

케라스에서는 이미지 파일을 쉽게 학습시킬 수 있도록 `ImageDataGenerator` 클래스를 제공합니다. 이 클래스를 이용하여 특정 폴더에 이미지를 분류해놓았을 때, 이를 학습시키기 위한 데이터셋으로 만들어주는 기능을 사용해보겠습니다.

먼저 `ImageDataGenerator` 클래스를 이용하여 객체를 생성하고, `flow_from_directory()` 함수를 호출하여 제너레이터를 생성합니다. `flow_from_directory()` 함수의 주요 인자는 아래와 같습니다.

* 첫 번째 인자(`directory`) : 이미지 경로를 지정합니다.
* `target_size` : 이미지 크기를 지정합니다. 폴더에 있는 원본 이미지 크기가 다르더라도 targer_size에 지정된 크기로 자동 조절됩니다.
* `batch_size` : 배치 크기를 지정합니다.
* `class_mode` : 분류 방식에 대해서 지정합니다.
  * `categorical`: 2D one-hot 부호화된 라벨이 반환됩니더.
  * `binary` : 1D 이진 라벨이 반환됩니다.
  * `sparse` : 1D 정수 라벨이 반환됩니다.
  * `None` : 라벨이 반환되지 않습니다.

이번 예제에서는 이미지 크기를 24 x 24로 만들었으니, target_size도 (24, 24)로 설정했습니다. 훈련 데이터 수가 클래스 당 15개이니 배치 크기를 3으로 지정하여 총 5번 배치를 수행하면 하나의 epoch가 수행될 수 있도록 했습니다. 세 개의 라벨 값을 가지는 다중 클래스 문제이므로 class_mode는 'categorical'로 지정했습니다. 그리고 제네레이터는 훈련용과 검증용 두 개를 만들었습니다.

```python
train_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(
				'CNN-tutorial/train',
				target_size = (24, 24),
				batch_size = 3,
				class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(
				'CNN-tutorial/test',
				target_size = (24, 24),
				batch_size = 3,
				class_mode = 'categorical')
```

```
# 출력:
Found 45 images belonging to 3 classes.
Found 15 images belonging to 3 classes.
```



## CNN 모델 구성하기

* 컨볼루션 레이어 :

   (입력 이미지) 크기 24 x 24, 채널 3개 / (필터) 크기 3 x 3, 32개 / (활성화함수) 'relu'

* 컨볼루션 레이어 :

   (필터) 크기 3 x 3, 64개 / (활성화함수) 'relu'

* 맥스풀링 레이어:

  (풀 크기) 2 x 2

* 플래튼 레이어

* 덴스 레이어 :

  (출력 뉴런) 128개 / (활성화함수) 'relu'

* 덴스 레이어 :

  (출력 뉴런) 3개 / (활성화함수) 'softmax'

 ```python
# 모델 만들기
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), 
                 padding = 'valid',
                 activation = 'relu',
                 input_shape = (24, 24, 3)))
model.add(Conv2D(64, (3, 3), 
                 padding = 'valid',
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
 ```

```python
# 모델 시각화
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

%matplotlib inline

SVG(model_to_dot(model, show_shapes = True).create(prog = 'dot', format = 'svg'))
```

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-2-23_CNN_Tutorial_2.png?raw=true">



## 모델 학습과정 설정하기

* `loss` : 현재 가중치 세트를 평가하는데 사용할 손실함수, 다중클래스이므로 **categorical_crossentropy** 로 지정합니다.
* `optimizer` : 최적의 가중치를 검색하기 위한 알고리즘으로 경사하강법 알고리즘 중 하나인 **adam** 을 사용합니다.

* `metrics` : 평가 척도를 나타내며 분류문제에서는 일반적으로 **accuracy** 로 지정합니다.

```python
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
```



## 모델 학습시키기

케라스에서 모델을 학습시킬 때, 주로 `fit( )` 함수를 사용하지만, 제네레이터로 생성된 배치로 학습시킬 경우에는 `fit_generator( )` 함수를 사용합니다.

* 첫 번째 인자 : 훈련 데이터셋을 제공할 제네레이터를 지정
* `steps_per_epoch` : 한 epoch에 사용한 스텝 수를 지정합니다. 총 45개의 훈련용 샘플이 있고, 배치 사이즈가 3이므로 15 스텝으로 지정합니다.
* `epochs` : 전체 훈련 데이터셋에 대한 반복 학습횟수를 지정합니다. 총 50번을 반복하겠습니다.
* `validation_data` : 검증 데이터셋을 제공할 제네레이터를 지정합니다.
* `validation_steps` : 한 epoch 종료 시마다 검증할 검증 스텝 수를 지정합니다. 총 15개의 검증 샘플이 있고, 배치사이즈 3이므로 5 스텝으로 지정합니다.

```python
model.fit_generator(
		train_generator,
		steps_per_epoch = 15, 
    epochs = 50,
    validation_data = test_generator,
    validation_steps = 5)
```

```
# 출력:
Epoch 1/50
15/15 [==============================] - 0s 24ms/step - loss: 1.1463 - accuracy: 0.4000 - val_loss: 0.9700 - val_accuracy: 0.6000
Epoch 2/50
15/15 [==============================] - 0s 9ms/step - loss: 0.9142 - accuracy: 0.6222 - val_loss: 0.5880 - val_accuracy: 0.8667
Epoch 3/50
15/15 [==============================] - 0s 9ms/step - loss: 0.4935 - accuracy: 0.8444 - val_loss: 0.2470 - val_accuracy: 0.8667
...
...
Epoch 48/50
15/15 [==============================] - 0s 10ms/step - loss: 4.1630e-05 - accuracy: 1.0000 - val_loss: 0.0061 - val_accuracy: 1.0000
Epoch 49/50
15/15 [==============================] - 0s 9ms/step - loss: 3.9707e-05 - accuracy: 1.0000 - val_loss: 0.0369 - val_accuracy: 1.0000
Epoch 50/50
15/15 [==============================] - 0s 10ms/step - loss: 3.8751e-05 - accuracy: 1.0000 - val_loss: 0.0380 - val_accuracy: 1.0000
```



## 모델 평가하기

학습한 모델을 평가해보겠습니다. 제네레이터로 제공되는 샘플로 평가할 때는 `evaluate_generator` 함수를 사용합니다.

```python
print("--Evaluate--")
scores = model.evaluate_generator(test_generator, steps = 5)
print('%s : %.2f%%' %(model.metrics_names[1], scores[1]*100))
```

```
# 출력:
--Evaluate--
accuracy : 100.00%
```

작은 데이터셋의 간단한 모델에서도 100%의 정확도를 얻었습니다.



## 모델 사용하기

모델 사용 시에 제네레이터에서 제공되는 샘플을 입력할 때는 `predict_generator` 함수를 사용합니다. 예측 결과는 클래스별 확률 벡터로 출력됩니다. 클래스에 해당되는 열을 알기 위해서는 제네레이터의 `class_indices`를 출력하면 해당 열의 클래스 명을 알려줍니다.

```python
print("--Predict--")
output = model.predict_generator(test_generator, steps = 5)
np.set_printoptions(formatter = {'float': lambda x : "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)
```

```
# 출력:
--Predict--
{'circle': 0, 'rectengular': 1, 'triangle': 2}
[[1.000 0.000 0.000]
 [0.902 0.098 0.000]
 [0.000 1.000 0.000]
 [0.981 0.000 0.019]
 [0.000 0.000 1.000]
 [0.000 1.000 0.000]
 [0.000 0.000 1.000]
 [0.000 0.000 1.000]
 [0.000 0.000 1.000]
 [0.315 0.685 0.000]
 [1.000 0.000 0.000]
 [0.000 0.000 1.000]
 [0.000 1.000 0.000]
 [0.107 0.892 0.001]
 [1.000 0.000 0.000]]
```



## 전체 소스 정리

```python
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# 랜덤 시드 고정시키기
np.random.seed(3)

# 1.데이터 생성하기
train_datagen = ImageDataGenerator(rescale= 1./255)
train_generator = train_datagen.flow_from_directory(
    'CNN-tutorial/train',
    target_size = (24, 24),
    batch_size = 3,
    class_mode = 'categorical'
)

test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(
    'CNN-tutorial/test',
    target_size = (24, 24),
    batch_size = 3,
    class_mode = 'categorical'
)

# 2. 모델 구성하기
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3),
                 activation = 'relu',
                 input_shape = (24, 24, 3)))
model.add(Conv2D(64, (3, 3), 
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 4. 모델 학습시키기
model.fit_generator(train_generator,
                    steps_per_epoch = 15,
                    epochs = 50,
                    validation_data = test_generator,
                    validation_steps = 5)

# 5. 모델 평가하기
print("--evaluate")
scores = model.evaluate_generator(test_generator, steps = 5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 6. 모델 사용하기
print("--Predict--")
output = model.predict_generator(test_generator, steps = 5)
np.set_printoptions(formatter = {'float' : lambda x : "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)
```

```
# 출력:
Found 45 images belonging to 3 classes.
Found 15 images belonging to 3 classes.
Epoch 1/50
15/15 [==============================] - 0s 21ms/step - loss: 1.3221 - accuracy: 0.3778 - val_loss: 1.2247 - val_accuracy: 0.3333
Epoch 2/50
15/15 [==============================] - 0s 9ms/step - loss: 1.0986 - accuracy: 0.3333 - val_loss: 1.0775 - val_accuracy: 0.3333
Epoch 3/50
15/15 [==============================] - 0s 9ms/step - loss: 1.0475 - accuracy: 0.5556 - val_loss: 1.1174 - val_accuracy: 0.6667
...
...
Epoch 48/50
15/15 [==============================] - 0s 9ms/step - loss: 3.8904e-04 - accuracy: 1.0000 - val_loss: 0.0394 - val_accuracy: 0.9333
Epoch 49/50
15/15 [==============================] - 0s 9ms/step - loss: 3.7031e-04 - accuracy: 1.0000 - val_loss: 1.0862e-04 - val_accuracy: 0.9333
Epoch 50/50
15/15 [==============================] - 0s 9ms/step - loss: 3.5682e-04 - accuracy: 1.0000 - val_loss: 0.2547 - val_accuracy: 0.9333
--evaluate
accuracy: 93.33%
--Predict--
{'circle': 0, 'rectengular': 1, 'triangle': 2}
[[1.000 0.000 0.000]
 [0.869 0.131 0.000]
 [0.000 1.000 0.000]
 [0.895 0.000 0.105]
 [0.000 0.000 1.000]
 [0.000 1.000 0.000]
 [0.000 0.000 1.000]
 [0.000 0.000 1.000]
 [0.000 0.000 1.000]
 [0.355 0.644 0.001]
 [1.000 0.000 0.000]
 [0.000 0.000 1.000]
 [0.000 1.000 0.000]
 [0.530 0.466 0.003]
 [0.999 0.000 0.000]]
```

---------

###### Reference

- 블록과 함께 하는 파이썬 딥러닝 케라스 (김태영 저)
