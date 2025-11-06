import csv
import json
import os

def json_to_csv(json_path: str, csv_path: str) -> None: 
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'Файл не найден: {json_path}')
    if os.path.getsize(json_path) == 0:
        raise ValueError('Файл пустой')
    with open(json_path,'r',encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        if not isinstance(json_data,list):
            raise ValueError('Json файл не является списком')
        if not(all(isinstance(x,dict) for x in json_data)):
            raise ValueError('Данные json файла не являются словарями')
        keys = set()
        for items in json_data:
            keys.update(items.keys())
        x = sorted(keys)
        with open(csv_path,'w',newline='',encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file,fieldnames=x)
            writer.writeheader()
            writer.writerows(json_data)

def csv_to_json(csv_path: str, json_path: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Файл не найден: {csv_path}')
    if os.path.getsize(csv_path) == 0:
        raise ValueError('Файл полностью пуст')
    try:
        with open(csv_path,'r',encoding='utf-8') as csv_file:
            csv_data = csv.DictReader(csv_file)

            if not csv_data.fieldnames:
                raise ValueError('В файле нет заголовка')
            
            row_l = list(csv_data)
            if len(row_l) == 0:
                raise ValueError('Файл не содержит никаких данных')
    except Exception as e:
            raise ValueError(f'Ошибка: {e}')
    with open(json_path,'w',encoding='utf-8') as json_file:
        json.dump(row_l,json_file,ensure_ascii=False,indent=2)
csv_to_json(r"C:\Users\Калачев Иван\Documents\GitHub\python_lab\data\samples\people.csv",
            r"C:\Users\Калачев Иван\Documents\GitHub\python_lab\data\out\people_from_csv.json")
json_to_csv(r"C:\Users\Калачев Иван\Documents\GitHub\python_lab\data\samples\people.json",
            r"C:\Users\Калачев Иван\Documents\GitHub\python_lab\data\out\people_from_json.csv" )    