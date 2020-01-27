---
layout: post
title: '[Python] 클래스의 정의'
excerpt: Python에서의 클래스 개념에 대해 알아보자.
category: Python
tags:
  - Python
  - class

---



## 클래스(class)의 정의

`클래스` 란 고유의 속성(attribute)와 동작(method)를 갖는 데이터 타입입니다.<br/>파이썬에서 string, list, float, dictionary 와 같은 것들이 모두 클래스로 존재합니다.

우리는 다루려는 데이터와 데이터를 다루는 함수를 하나의 클래스로 표현할 수 있습니다.<br/>가령, 게임의 직업에 관해 '궁수'라는 클래스를 만든다면, 궁수의 고유속성과 행동을 클래스를 통해 정의할 수 있습니다.



### ※ 객체(object)란?

클래스로 생성되어 구체화된 것이 객체입니다. 클래스를 빵틀이라고 한다면, 객체는 그 빵틀로 찍어낸 빵으로 비유할 수 있습니다.

```python
class Niniz:
  pass # 정의만 하고 구체적인 표현을 미루고자 할 때는 pass를 사용합니다.
```

```python
# jordy와 angmond라는 Niniz 클래스의 객체를 생성합니다.
jordy = Niniz()
angmond = Niniz()

print(type(jordy), type(angmond))
```

```
# 출력:
<class '__main__.Niniz'> <class '__main__.Niniz'>
```

위의 결과에서 보듯이 jordy와 angmond라는 Niniz의 객체를 생성하면, 데이터 타입은 Niniz입니다.



## \__init__(self)

클래스가 생성될 때, 먼저 호출되고, 해당 클래스가 다루는 데이터를 정의합니다.<br/>`self` 인자는 항상 첫 번째에 오며, 입력되는 객체 자신을 가리킵니다.

여기에서 정의된 데이터를 `멤버변수(member variable)` , 또는 `속성(attribute)` 이라고 합니다. 

```python
class Niniz:
  # Niniz라는 클래스가 가지는 속성 정의
  def __init__(self):
    print(self, 'is generated')
    self.name = '죠르디' # Niniz라는 클래스의 name 속성 정의
    self.age = 100 # Niniz라는 클래스의 age 속성 정의
    
p1 = Niniz()
print(p1.name, p1.age)

# 개별 객체 속성값은 클래스에서 정의한 값에서 변경해줄 수 있습니다.
p1.name = '앙몬드'
p1.age = 10
print(p1.name, p1.age)
```

```
# 출력:
<__main__.Niniz object at 0x108d5e400> is generated
죠르디 100
앙몬드 10
```

```python
class Niniz:
  def __init__(self, name, age, status):
    # Niniz라는 클래스의 name, age, status 속성의 동적 정의
    self.name = name
    self.age = age
    self.status = status
    
p1 = Niniz('죠르디', 100, '취준생공룡')
p2 = Niniz('앙몬드', 10, '하프물범')
p3 = Niniz('스카피', 30, '토끼')
print(p1.name, p1.age, p1.status)
print(p2.name, p2.age, p2.status)
print(p3.name, p3.age, p3.status)
```

```
# 출력: 
죠르디 100 취준생공룡
앙몬드 10 하프물범
스카피 30 토끼
```



## self

파이썬의 메소드는 항상 첫 번째 인자로 self를 전달합니다.<br/>위에서 이야기한 것처럼 self는 해당 메소드가 호출하는 객체 자기자신을 가리킵니다.

이름이 반드시 self일 필요는 없지만, 관례적으로 self를 사용합니다.

```python
class Diablo:
  def __init__(self, name, job, skill)
  self.name = name
  self.job = job
  self.skill = skill
  
  def hunt(self):
    print('{}은 {}이기 때문에 {}로 디아블로를 사냥합니다.'.format(self.name, self.job, self.skill))
    
a = Diablo('티리얼', '바바리안', '휠윈드')
b = Diablo('데커드 케인', '성전사', '해머')

print(a)
print(b)

a.hunt()
b.hunt()
```

```
# 출력:
<__main__.Diablo object at 0x108d6d160>
<__main__.Diablo object at 0x108d6d1d0>
티리얼은 바바리안이기 때문에 휠윈드로 디아블로를 사냥합니다.
데커드 케인은 성전사이기 때문에 해머로 디아블로를 사냥합니다.
```



---------

###### Reference

- 패스트캠퍼스 파이썬 강의
