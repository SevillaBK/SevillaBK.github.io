---
layout: post
title: '[Python] 딕셔너리'
excerpt: Python의 데이터 타입 중 딕셔너리에 대해 알아보자.
category: Python
tags:
  - Python
  - dictionary

---



`딕셔너리(dictionary)` 는 **키(key)**와 **값(value)**을 갖는 데이터 구조입니다. 앞서 보았던 리스트, 튜플 등과 달리 딕셔너리는 내부 데이터에 별도 순서, 즉, 인덱스가 없습니다. 



## 딕셔너리의 생성/추가/변경

중괄호 안에 키와 값을 넣어 딕셔너리 값을 생성합니다.<br/>딕셔너리의 내부 항목 변경시 기존에 동일한 키가 존재하면 새로운 값으로 업데이트합니다.<br/>키가 존재하지 않는다면 새로운 키와 값을 생성합니다.

```python
# 딕셔너리는 중괄호 안에 키와 값을 정의해주며 생성한다.
a = {'Korea' : 'Seoul',
     'Canada' : 'Ottawa', 
     'USA' : 'Washington D.C'}
    
print(type(a))
print(a)

# 딕셔너리는 인덱스라는 개념이 없으며 키 값으로 데이터를 불러옵니다.
a['Korea']
```

```
# 출력: 
<class 'dict'>
{'Korea': 'Seoul', 'Canada': 'Ottawa', 'USA': 'Washington D.C'}
'Seoul'
```

```python
a = {'Korea' : 'Jeju',
     'Canada' : 'Ottawa',
     'USA' : 'Washington'}
# 같은 키에 다른 값을 입력하면 마지막 값으로 업데이트가 된다.    
a['Korea'] = 'Seoul' 
a['Spain'] = 'Madrid'
a['Italy'] = 'Rome'
    
print(a)
```

```
# 출력:
{'Korea': 'Seoul', 'Canada': 'Ottawa', 'USA': 'Washington', 'Spain': 'Madrid', 'Italy': 'Rome'}
```



##### (1) update( )

두 딕셔너리를 병합할 수 있습니다. 겹치는 키가 있다면 파라미터로 전달되는 키 값 기준으로 데이터를 변경합니다.

```python
a = {'a':1, 'b':2, 'c':3}
b = {'a':2, 'd':4, 'e':5}
    
a.update(b)
print(a)
```

```
# 출력:
{'a': 2, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
```



##### (2) key 삭제

`del` 키워드를 사용하는 방법과 `pop( )` 함수를 사용하는 방법이 있습니다.

```python
# del 키워드를 사용해서 키 삭제
a = {'a':1, 'b':2, 'c':3}
del a['b']
print(a)
```

```
# 출력:
{'a': 1, 'c': 3}
```

```python
# pop( )함수를 사용하여 키 삭제
a = {'a':1, 'b':2, 'c':3}
a.pop('b')
print(a)
```

```
# 출력:
{'a': 1, 'c': 3}
```



##### (3) clear( )

딕셔너리의 모든 값을 초기화 하는 기능입니다.

```python
a = {'a':1, 'b':2, 'c':3}
a.clear()
print(a)
```

```
# 출력:
{}
```



## 딕셔너리 내부의 값 확인

##### (1) in

키 값이 존재하는지 확인할 수 있습니다. 리스트의 경우 데이터가 커지면 in을 사용할 때 느려질 수 있지만, 딕셔너리의 경우 일정하게 검색할 수 있습니다.

```python
a = {'a':1, 'b':2, 'c':3}
print('b' in a)
print('d' in a)
```

```
# 출력:
True
False
```

```python
a = {'a':1, 'b':2, 'c':3}
b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
# 비슷해 보여도 리스트에서 in을 사용할 때 리스트의 길이가 길면 느려질 수 있다. 
# 하지만 딕셔너리는 일정하게 금방 찾아낸다.
print('b' in a)
print(100 in b)
```

```
# 출력:
True
False
```



##### (2) dict[key]

키에 따른 값을 확인할 수 없습니다. 없는 키를 사용하면 에러가 발생합니다.

```python
a = {'a':1, 'b':2, 'c':3}
a['a']
```

```
# 출력: 
1
```

```python
# 존재하지않는 키를 넣으면 에러가 발생
a['d']
```

```
# 출력 오류메시지:
KeyError: 'd'
```



##### (3) get( )

키에 따른 값을 역시 확인할 수 있습니다. 다만 이 경우에는 키가 없으면 None을 반환합니다.

```python
a = {'a':1, 'b':2, 'c':3}
print(a.get('d'))
```

```
# 출력:
None
```



##### (4) 모든 키와 값에 접근하기

- **keys( )**: 존재하는 모든 키를 반환
- **values( )**: 존재하는 모든 값을 반환
- **items( )**: 키와 값의 튜플을 반환

```python
a = {'a':1, 'b':2, 'c':3}
print(a)
print(a.keys())
print(a.values())
print(a.items())

print()
### 키와 값을 리스트로 반환하기
print(list(a.keys()))
print(list(a.values()))
```

```
# 출력
{'a': 1, 'b': 2, 'c': 3}
dict_keys(['a', 'b', 'c'])
dict_values([1, 2, 3])
dict_items([('a', 1), ('b', 2), ('c', 3)])

['a', 'b', 'c']
[1, 2, 3]
```









---------

###### Reference

- 패스트캠퍼스 파이썬 강의
- 점프투파이썬 (https://wikidocs.net/book/1)
