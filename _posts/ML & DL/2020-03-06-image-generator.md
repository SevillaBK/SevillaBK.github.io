---
layout: post
title: '[케라스] 컨볼루션 신경망(CNN) 모델 학습을 위한 데이터 부풀리기'
excerpt: ImageDataGenerator 를 활용하여 이미지 데이터를 부풀려 학습에 이용하기
category: ML & DL
tags:
  - 딥러닝
  - 케라스
  - CNN
  - ImageDataGenerator

---

김태영님의 `블록과 함께 하는 파이썬 딥러닝 케라스` 를 공부하며 옮겨온 내용입니다.

--------------

이번에는 데이터셋을 부풀리는 방법에 대해 알아보겠습니다. 앞선 포스팅에서 사용한 그림 예제를 사용하겠습니다. 훈련셋이 부족하거나 훈련셋이 테스트셋의 성능을 충분히 반영하지 못할 때 사용하면 모델의 성능이 향상될 수 있습니다.

앞선 포스팅(https://sevillabk.github.io/1-CNN-Tutorial/)에서는 훈련셋을 원, 사각형, 삼각형에 대해 각 15개, 테스트셋을 5개 만들어서 딥러닝 모델을 학습, 평가해보았습니다.

#### 훈련셋

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-03-total_data.png?raw=true">

#### 테스트셋

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-03-test_data.png?raw=true">

위의 훈련셋과 테스트셋으로는 적은 데이터로도 100%의 정확도를 얻을 수 있었습니다. 하지만 다른 테스트셋에 대해서도 동일한 성능을 보일 수 있을지는 모릅니다.

#### 추가 테스트셋

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-03-additional-test-data.png?raw=true">

추가 테스트셋은 기존에 훈련시켰던 도형들의 그림과는 다소 상이하여 잘 예측해내기는 어려울 것 같습니다. 그래도 먼저 만들었던 것과 동일한 모델을 이용해서 예측을 해보겠습니다.



## 기존 모델과 데이터셋을 활용한 예측

```python
import numpy as np

# 랜덤시드 고정시키기
np.random.seed(3)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

# 데이터셋 불러오기
train_datagen = ImageDataGenerator(rescale= 1./255)
train_generator = train_datagen.flow_from_directory(
    'CNN-tutorial/train',
    target_size = (24, 24),
    batch_size = 3,
    class_mode = 'categorical'
)

test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(
    'CNN-tutorial/additional test',
    target_size = (24, 24),
    batch_size = 3,
    class_mode = 'categorical'
)

# 모델 구성하기
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), 
                 padding = 'valid', 
                 activation = 'relu',
                 input_shape = (24, 24, 3)))
model.add(Conv2D(64, kernel_size = (3, 3),
                 padding = 'valid',
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

# 모델 학습과정 설정하기
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 학습하기
model.fit_generator(train_generator, 
                    steps_per_epoch = 15,
                    epochs = 200,
                    validation_data = test_generator,
                    validation_steps = 5)

# 모델 평가하기
print('--Evaluate--')

scores = model.evaluate_generator(test_generator,
                                  steps = 5)

print('%s : %.2f%%' %(model.metrics_names[1], scores[1]*100))

# 모델 예측하기
print('--Predict--')

output = model.predict_generator(test_generator,
                                 steps = 5)

np.set_printoptions(formatter = {'float' : lambda x : "{0:0.3f}".format(x)})

print(output)
```

```
# 출력:
Found 45 images belonging to 3 classes.
Found 15 images belonging to 3 classes.
Epoch 1/200
15/15 [==============================] - 0s 21ms/step - loss: 1.2583 - accuracy: 0.3333 - val_loss: 1.1428 - val_accuracy: 0.4667
Epoch 2/200
15/15 [==============================] - 0s 9ms/step - loss: 1.0100 - accuracy: 0.5778 - val_loss: 1.0434 - val_accuracy: 0.3333
Epoch 3/200
15/15 [==============================] - 0s 10ms/step - loss: 0.7234 - accuracy: 0.8444 - val_loss: 1.3652 - val_accuracy: 0.5333
...
...
Epoch 199/200
15/15 [==============================] - 0s 10ms/step - loss: 9.7487e-07 - accuracy: 1.0000 - val_loss: 3.7344 - val_accuracy: 0.7333
Epoch 200/200
15/15 [==============================] - 0s 10ms/step - loss: 9.5897e-07 - accuracy: 1.0000 - val_loss: 3.0683 - val_accuracy: 0.7333
--Evaluate--
accuracy : 73.33%
--Predict--
[[0.002 0.069 0.929]
 [0.092 0.598 0.311]
 [0.001 0.000 0.999]
 [0.000 0.203 0.796]
 [0.008 0.957 0.035]
 [0.996 0.004 0.000]
 [0.001 0.951 0.049]
 [1.000 0.000 0.000]
 [0.000 0.008 0.992]
 [0.703 0.296 0.001]
 [0.003 0.997 0.000]
 [0.175 0.010 0.814]
 [0.000 0.000 1.000]
 [0.000 0.999 0.001]
 [0.009 0.964 0.026]]
```

앞선 포스팅에서는 100%의 정확도를 달성했지만, 테스트셋이 바뀌니 73.33로 정확도가 떨어졌습니다.



## 데이터 부풀리기

모델이 활용도가 높으려면 어떤 데이터가 시험셋으로 주어져도 성능이 안정적으로 나와야합니다. 하지만 데이터셋이 제한적인 경우에는 안정적인 모델의 성능을 끌어내기 어렵습니다. 이러한 경우에 케라스의 `ImageDataGenerator` 함수를 통해 데이터를 부풀려서 모델을 학습시킬 수 있습니다.

```python
keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, 
    samplewise_center=False, 
    featurewise_std_normalization=False, 
    samplewise_std_normalization=False, 
    zca_whitening=False, 
    zca_epsilon=1e-06, 
    rotation_range=0, 
    width_shift_range=0.0, 
    height_shift_range=0.0, 
    brightness_range=None, 
    shear_range=0.0, 
    zoom_range=0.0, 
    channel_shift_range=0.0, 
    fill_mode='nearest', 
    cval=0.0, 
    horizontal_flip=False, 
    vertical_flip=False, 
    rescale=None, 
    preprocessing_function=None, 
    data_format='channels_last', 
    validation_split=0.0, 
    interpolation_order=1, 
    dtype='float32')
```

`ImageDataGenerator`에는 다양한 파라미터들이 존재하지만 이번 데이터 부풀리기를 위해 사용될 몇 가지 파라미터들에 대해서만 살펴보겠습니다. 이에 대한 예시로 훈련셋에 있는 이미지 데이터 중 아래의 하나의 이미지로 `ImageGenertor` 를 사용한 데이터 부풀리기를 해보겠습니다.

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-04-triangle001.png?raw=true">

* **rotation_range = 90 **

  지정된 각도 범위 내에서 원본 이미지를 임의로 회전시킵니다. 단위는 도이며 int형 입니다.<br/>이번 케이스에서는 0도에서 90도 사이의 임의의 각도로 이미지를 회전시킵니다.

  <img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-04-rotation_range.png?raw=true">

* **width_shift_range = 0.3**

  지정된 수평이동 범위 내에서 임의로 원본 이미지를 이동시킵니다. 수치는 전체 넓이의 비율로 나타냅니다. 가령, 이번 케이스에서 이미지가 100픽셀이라면 30픽셀 이내 범위에서 좌우 이동을 합니다.

  <img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-04-width_shift_range.png?raw=true">

* **height_shift_range = 0.3**

  지정된 수직이동 범위 내에서 임의로 원본 이미지를 이동시킵니다. 수치는 전체 넓이의 비율로 나타냅니다. 가령, 이번 케이스에서 이미지가 100픽셀이라면 30픽셀 이내 범위에서 상하 이동을 합니다.

  <img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-04-height_shift_range.png?raw=true">

* **vertical_flip = True**

  수직방향으로 뒤집기를 합니다.

  <img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-04-vertical_flip.png?raw=true">

* **horizontal_flip = True**

  수평방향으로 뒤집기를 합니다.

  <img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-04-horizontal_flip.png?raw=true">

* **zoom_range = 0.5** or **zoom_range = [0.5, 1.5]**

  무작위 줌의 범위로 부동소수점 혹은 [하한, 상한]으로 입력합니다.  실수 하나만 입력하면, `[하한, 상한] = [1-zoom_range, 1+zoom_range]`입니다.

  <img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-04-zoom_range.png?raw=true">

아래 코드는 `ImageDataGenerator` 함수를 사용해서 원본 이미지에 대해 데이터를 부풀리기를 수행하고, 그 결과를 지정 폴더에 저장하는 코드입니다. 여러 파라미터를 동시로 사용하여 다양하게 생성해보겠습니다. 

```python
import numpy as np

# 랜덤시드 고정시키기
np.random.seed(5)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 데이터셋 불러오기
data_aug_gen = ImageDataGenerator(rescale=1./255, 
                                  rotation_range=15,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.5,
                                  zoom_range=[0.8, 2.0],
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')

img = load_img('CNN-tutorial/train/triangle/triangle001.png')
x = img_to_array(img)
x = x.reshape((1, ) + x.shape)

i = 0
# 이 구문은 무한 반복되기 때문에 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 세팅해주어야 합니다.
for batch in data_aug_gen.flow(x, batch_size = 1, save_to_dir = 'CNN-tutorial/dummy', save_prefix = 'triange', save_format = 'png'):
    i += 1
    if i >= 100:
        break
```

아래는 위 코드 결과로 생성된 이미지들의 일부입니다. 처음 손으로 그렸던 것보다 다양한 형태의 삼각형 이미지들이 만들어졌습니다. 

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-04-ImageGenerator.png?raw=true">



이미지 데이터를 부풀려서 예측을 수행해보겠습니다. `train_data_gen` 객체에만 위에서 사용한 코드를 사용하였습니다. `test_data_gen` 객체는 별도 파라미터 추가가 없습니다. 그리고 `fit_generator` 함수에서 steps_per_epoch의 값은 기존 15개에서 더 많은 수로 설정합니다. 이번 예시의 경우에는 750개로 설정했습니다. 

```python
import numpy as np

# 랜덤시드 고정시키기
np.random.seed(3)

from keras.preprocessing.image import ImageDataGenerator

# 데이터셋 불러오기 : 이 부분이 수정되었습니다.
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.7,
                                   zoom_range=[0.9, 2.2],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        'CNN-tutorial/train',
        target_size=(24, 24),
        batch_size=3,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        'CNN-tutorial/test',
        target_size=(24, 24),    
        batch_size=3,
        class_mode='categorical')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.layers import Dropout

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(24,24,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습시키기
model.fit_generator(
        train_generator,
        steps_per_epoch=15 * 50,
        epochs=200,
        validation_data=test_generator,
        validation_steps=5)

# 모델 평가하기
print("-- Evaluate --")

scores = model.evaluate_generator(
            test_generator, 
            steps = 5)

print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 모델 예측하기
print("-- Predict --")

output = model.predict_generator(
            test_generator, 
            steps = 5)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

print(output)
```

```
# 출력:
Found 45 images belonging to 3 classes.
Found 15 images belonging to 3 classes.
Epoch 1/200
750/750 [==============================] - 8s 10ms/step - loss: 0.7090 - accuracy: 0.6613 - val_loss: 0.0696 - val_accuracy: 0.6667
Epoch 2/200
750/750 [==============================] - 7s 10ms/step - loss: 0.1660 - accuracy: 0.9458 - val_loss: 1.3452 - val_accuracy: 0.6000
Epoch 3/200
750/750 [==============================] - 7s 10ms/step - loss: 0.0896 - accuracy: 0.9689 - val_loss: 2.1614 - val_accuracy: 0.6667
...
...
Epoch 199/200
750/750 [==============================] - 7s 10ms/step - loss: 0.0078 - accuracy: 0.9987 - val_loss: 0.0000e+00 - val_accuracy: 0.8000
Epoch 200/200
750/750 [==============================] - 8s 10ms/step - loss: 0.0013 - accuracy: 0.9991 - val_loss: 0.0821 - val_accuracy: 0.8667
-- Evaluate --
accuracy: 86.67%
-- Predict --
[[1.000 0.000 0.000]
 [0.000 0.000 1.000]
 [0.000 0.000 1.000]
 [0.000 0.000 1.000]
 [1.000 0.000 0.000]
 [1.000 0.000 0.000]
 [0.000 1.000 0.000]
 [0.000 1.000 0.000]
 [0.000 0.000 1.000]
 [1.000 0.000 0.000]
 [0.000 0.000 1.000]
 [0.000 0.000 1.000]
 [1.000 0.000 0.000]
 [0.000 1.000 0.000]
 [0.200 0.019 0.782]]
```

데이터를 부풀린 결과, 변경된 테스트셋에 대해서도 86.67%의 정확도를 얻었습니다. 이와 같이 훈련셋이 충분하지 않거나 테스트셋의 다양한 특성이 반영되어 있지 않다면 데이터 부풀리기는 성능 개선에 도움을 줄 수 있습니다.

---------

###### Reference

- 블록과 함께 하는 파이썬 딥러닝 케라스 (김태영 저)
