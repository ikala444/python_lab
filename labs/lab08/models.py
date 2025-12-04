
from dataclasses import dataclass, field
from datetime import datetime, date


@dataclass
class Student:
    fio: str
    birthdate: str
    group: str
    gpa: float
    
    def __post_init__(self):
        try:
            datetime.strptime(self.birthdate, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Неверный формат даты: {self.birthdate}. "
                f"Ожидается YYYY-MM-DD"
            )
        
        if not (0 <= self.gpa <= 5):
            raise ValueError(
                f"Средний балл должен быть от 0 до 5. Получено: {self.gpa}"
            )
        
        birth_date = datetime.strptime(self.birthdate, "%Y-%m-%d").date()
        if birth_date > date.today():
            raise ValueError(
                f"Дата рождения не может быть в будущем: {self.birthdate}"
            )
    
    def age(self) -> int:
        birth_date = datetime.strptime(self.birthdate, "%Y-%m-%d").date()
        today = date.today()
        
        age = today.year - birth_date.year
        
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        return age
    
    def to_dict(self) -> dict:
        return {
            "fio": self.fio,
            "birthdate": self.birthdate,
            "group": self.group,
            "gpa": self.gpa,
            "age": self.age()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Student":
        required_fields = ["fio", "birthdate", "group", "gpa"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Отсутствует обязательное поле: {field}")
        
        return cls(
            fio=data["fio"],
            birthdate=data["birthdate"],
            group=data["group"],
            gpa=data["gpa"]
        )
    
    def __str__(self) -> str:
        return (
            f"Студент: {self.fio}\n"
            f"Дата рождения: {self.birthdate} (возраст: {self.age()})\n"
            f"Группа: {self.group}\n"
            f"Средний балл: {self.gpa:.2f}"
        )

