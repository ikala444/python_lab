
import json
from pathlib import Path
from typing import List
from models import Student


def students_to_json(students: List[Student], path: str) -> None:
    if not isinstance(students, list):
        raise TypeError("Ожидается список студентов")
    
    if not students:
        raise ValueError("Список студентов пуст")
    
    data = [student.to_dict() for student in students]
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"Сохранено {len(students)} студентов в {path}")


def students_from_json(path: str) -> List[Student]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON должен содержать массив объектов")
    
    students = []
    errors = []
    
    for i, item in enumerate(data, 1):
        try:
            student = Student.from_dict(item)
            students.append(student)
        except ValueError as e:
            errors.append(f"Строка {i}: {e}")
    
    if errors:
        print("Обнаружены ошибки при загрузке:")
        for error in errors:
            print(f"  - {error}")
    
    print(f"Загружено {len(students)} студентов из {path}")
    return students


