min_all = int(input('Минуты: '))
hour=min_all//60
min=min_all-hour*60
days=0
if hour>=24:
    days=hour//24
    hour-=days*24
print(f'Дней: {days} ЧЧ:ММ - {hour}:{min}')
