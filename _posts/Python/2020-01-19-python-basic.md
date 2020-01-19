---
layout: post
title: '[Python] Python 기초사항 및 변수'
excerpt: Python의 기초 사항 및 변수에 대해 알아보자.
category: Python
tags:
  - Python
  - Variable

---



## 변수(Variable)

변수란 데이터를 저장하는 공간입니다.

- 각 변수에는 이름(Name)을 지정할 수 없습니다.

* **변수이름 = 변수 값** 의 형태로 만들어 줍니다.

  ```python
  # 변수의 할당
  a = 10 # int(정수형 변수)
  b = 11.4 # float(실수형 변수)
  ```



## 주석(Comment)

코드에서 # 으로 시작되는 부분은 실행되지 않습니다.<br/>개발자가 코드에 대해 설명을 달아두기 위한 용도로 사용합니다.

```python
# this line is very important
# so don't delete those lines   
a = 10
b = 11.4
```



## print 함수

```python
print(value, ..., sep = ' ', end ='\n')
```

함수안에 들어간 변수의 값을 출력하는 함수입니다.

- `,` 로 여러 변수를 나열하면 한 줄에 출력합니다.
- 여러 변수 나열 시 기본적으로 한 칸 띄운 채로 출력합니다.

```python
print()
print(a, b)
print(a, 10, 200, b)

# 출력: 
10 11.4
10 10 200 11.4
```

* print 함수의 파라미터
  - `sep` : 각 출력할 변수 사이에 입력되는 구분자를 설정하는 것으로 default는 빈 한 칸(`' '`)입니다.
  - `end` : 마지막에 출력할 문자열로 default는 한 줄 띄우기(`\n`)입니다.

```python
print(a, b, 10, 100, sep='*', end='!!'))

# 출력:
10*11.4*10*100!!
```



## 변수 값 확인

* print( ) 함수 활용
* 변수 값을 코드의 마지막에 위치시킨 후 실행

```python
a = 10
b = 11.4
print(a)
b

# 출력:
10
11.4
```



## 변수 이름 생성규칙

* `영대소문자, _ , 숫자`로 구성가능합니다. 하지만 숫자로 시작할 수 없습니다.
* 일반적으로 해당 변수가 표현하려는 정확하고 간결한 이름을 사용합니다.<br/>이를 통해 코드를 쉽게 이해할 수 있습니다.<br/>e.g) a = 1000 보다 policy_num = 1000 으로 명시하는 것이 보다 이해하기 좋습니다.

```python
# 아래는 모두 유효한 변수명
abcABC = 100
abc_123 = 200
_abc123 = 300
A45s3bC = 100

# 다만, 아래와 같이 숫자를 시작으로 변수명을 생성할 경우 오류 메시지가 나타냅니다.
12asd = 100
```



## 예약어 (reserved keywords)

파이썬에서 미리 선점해서 사용 중인 키워드로, 예약어들은 변수, 함수, 클래스 등등에서 이름으로 지정할 수 없습니다.

```python
# 예약어의 예시
class
for
while
if
elif
try
except
```



## 기본 파이썬 데이터 타입

정수(int), 실수(float), 문자열(str), 불리언(boolean)이 있습니다.<br/>`type( )` 함수를 이용하면 변수의 타입을 확인할 수 있습니다.

```python
a = 10 
b = 11.45
type(a), type(b)

# 출력:
(int, float)
```



## None

아무런 값을 갖지 않을 때 사용하며, 일반적으로 변수가 초기 값을 갖지 않은 채 변수를 생성할 때 사용합니다.

```python
c = None
print(c)

# 출력:
None
```



## 비교 연산자

* `==` : 같다 ⟷ `!=`: 같지 않다
* `< , > , <= , >=` : 대소 비교
* 비교 연산자의 결과는 `True, False`로 반환됩니다.

```python
a = 5
b = 4
    
print(a>b)
print(a<b)
print(a==b)
print(a<=b)
print(a>=b)
print(a!=b)

# 출력:
True
False
False
False
True
True
```

```python
c = a > b
print(type(c))
print(c)

# 출력:
<class 'bool'>
True
```



## 숫자형 타입

정수(int)와 실수(float)로 구성되어 있습니다. 숫자형 데이터는 수학의 기본 연산을 사용할 수 있습니다.

```python
a = 5 
b = 3
print(a + b) # 더하기 : 8
print(a * b) # 곱하기 : 15
print(a - b) # 빼기 : 2
print(a / b) # 나누기 : 1.66666...
print(a % b) # 나머지 : 2
print(a // b) # 몫 : 1
print(a ** b) # a의 b제곱 : 125

# 출력:
8
15
2
1.6666666666666667
2
1
125
```



## 연산의 우선순위

기본적으로 수학에서 사용하는 것과 동일합니다.<br/>괄호를 사용하면 연산을 우선해서 수행할 수 있습니다.

```python
a = 5 
b = 4
print(a + b * 4)
print((a + b) * 4)

# 출력:
21
36
```



## 변수값의 변경

변수값을 변경하려면 변수명에 = 를 사용하여 다른 값을 입력하면 됩니다.

```python
# = 없이 연산만 수행하면 변수값은 변하지 않습니다.
a = 9
a - 3
print(a)
    
# = 을 통해 변경될 값을 지정해주면 변수값은 변합니다.
a = 9
a = a - 3
print(a)
    
# -= : 우측에 있는 값을 뺀 값으로 대체합니다.
a = 9
a -= 3
    print(a)
    
# += : 우측에 있는 값을 더한 값으로 대체합니다.
a = 9
a += 3
print(a)
    
# *= : 우측에 있는 값을 곱한 값으로 대체합니다.
a = 9
a *= 3
print(a)
    
# += : 우측에 있는 값만큼 제곱한 값으로 대체합니다.
a = 9
a **= 3
print(a)

# 출력:
9
6
6
12
27
729
```



----------

###### Reference

- 패스트캠퍼스 파이썬 강의
- 점프투파이썬 (https://wikidocs.net/book/1)

