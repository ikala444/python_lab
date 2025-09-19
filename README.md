## Лабораторная работа 1

### Задание 1
```python
a,b= input(), int(input())
print(f'Имя: {a}\nВозраст: {b}\nПривет, {a}! Через год тебе будет {b+1}.')
```
![Картинка 1](./images/lab01/img01.png)

### Задание 2
```python
a,b=input(),input()
if ',' in a:
    a=a.replace(',','.')
if ',' in b:
    b=b.replace(',','.')
print(a,b)
print(f'a: {float(a)}\nb: {float(b)}\nsum: {round(float(a)+float(b),2)}\navg: {round((float(a)+float(b))/2,2)}')
```
![Картинка 1](./images/lab01/img02.png)

### Задание 3
```python
price, discount, vat = float(input()), float(input()), float(input())

base = price * (1 - discount / 100)
vat_amount = base * (vat / 100)
total = base + vat_amount

print(f'База после скидки: {base:.2f}')
print(f'НДС:               {vat_amount:.2f}')
print(f'Итого к оплате:    {total:.2f}')
```
![Картинка 1](./images/lab01/img03.png)


### Задание 4
```python
min_all = int(input('Минуты: '))
hour=min_all//60
min=min_all-hour*60
days=0
if hour>=24:
    days=hour//24
    hour-=days*24
print(f'Дней: {days} ЧЧ:ММ - {hour}:{min}')
```
![Картинка 1](./images/lab01/img04.png)

### Задание 5
```python
fio = input()
fio_apart = fio.strip()
words = fio_apart.split()
initials = "".join([word[0].upper() for word in words])
print(f'Инициалы: {initials}')
fio_full = fio_apart.replace(' ','')
print(f'Длина (символов): {len(fio_full)+2}')
```
![Картинка 1](./images/lab01/img05.png)
