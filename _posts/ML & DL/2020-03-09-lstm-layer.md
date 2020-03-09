---
layout: post
title: '[케라스] 순환신경망(RNN)의 LSTM 레이어'
excerpt: 순환신경망 모델에서 사용되는 LSTM 레이어의 소개
category: ML & DL
tags:
  - 딥러닝
  - 케라스
  - RNN
  - LSTM

---

김태영님의 `블록과 함께 하는 파이썬 딥러닝 케라스` 를 공부하며 옮겨온 내용입니다.

--------------

`순환신경망(Recurrent Neural Network)`은 순차적인 자료에서 패턴을 인식하거나 의미를 추론할 수 있는 모델입니다. 케라스에서 제공하는 순환신경망의 레이어는 `SimpleRNN`, `GRU`, `LSTM` 등이 있는데, 이번 포스팅에서는 `LSTM` 에 대해 알아보겠습니다.



## 긴 시퀀스를 기억하는 LSTM(Long Short-Term Memory) 레이어

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-07-LSTM-input.png?raw=true">

LSTM 레이어의 인풋 데이터는 위의 그림과 같은 3차원 array 입니다. 

* **batch_size** 
* **input_length(time_steps)** : 시퀀스 데이터의 입력 길이
* **input_dim** : 입력 값의 속성 수 입니다.

LSTM 레이어는 아래와 같이 간단하게 사용할 수 있습니다.

```python
LSTM(3, input_shape = (4, 1))
```

위의 **input_shape** 의 입력값을 보면 2D array 같아 보입니다. **input_length** 가 4이며, **input_dim**은 1입니다. **batch_size**의 경우, 데이터를 네트워크에 피팅할 때 유연하게 조정됩니다.

**batch_size** 를 고정하고 싶으면, 아래와 같이 **batch_input_shape** 를 인자로 사용하면 됩니다.

```python
LSTM(3, batch_input_shape = (4, 4, 1))
```

Dense와 LSTM을 블록으로 도식화하면 아래와 같습니다. 왼쪽은 Dense, 가운데는 input_length가 1인 LSTM, 오른쪽은 input_length가 4인 LSTM입니다. Dense 레이어와 비교하면 히든 뉴런들이 밖으로 도출되어 있는 형태입니다. input_length가 길다고 해서 각 입력마다 다른 가중치를 사용하는 것이 아니라 중앙에 있는 볼륵을 입력길이 만큼 연결한 것이기 때문에 모두 동일한 가중치를 공유합니다.

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-08-RNN_Layer_Talk_LSTM1.png?raw=true">

## 출력형태

* **return_sequence** : 시퀀스 출력 여부

LSTM 레이어는 **return_sequence** 인자에 따라 마지막 시퀀스에서 한 번만 출력할 수도, 각 시퀀스에서 출력을 할 수도 있습니다. many to many 문제를 풀거나 LSTM 레이어를 여러 개로 쌓아올릴 때는 **return_sequence = True** 옵션을 사용합니다. 아래에서 왼쪽 그림은 **return_sequence = False** 일 때, 오른쪽은 **return_sequence = True** 일 때 입니다.

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-09-RNN_Layer_Talk_LSTM2.png?raw=true">

## 상태유지

* **stateful** : 상태유지 여부

학습 생플의 가장 마지막 상태가 다음 샘플 학습 시에 입력으로 전달되는지 여부를 정하는 것입니다. 하나의 샘플은 4개의 시퀀스 입력이 있고, 총 3개의 샘플이 있을 때, 상단에 있는 블록들은 **stateful = False** 일 때의 그림이고, 하단에 있는 블록들은 **stateful = True** 일 때의 그림입니다. 각 샘플별 도출된 가중치가 다음 샘플 학습시 초기 상태의 입력 값으로 입력이 됩니다.

<img src = "https://github.com/SevillaBK/SevillaBK.github.io/blob/master/img/Keras/2020-03-09-RNN_Layer_Talk_LSTM3.png?raw=true">

---------

###### Reference

- 블록과 함께 하는 파이썬 딥러닝 케라스 (김태영 저)
- https://medium.com/@shivajbd/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
