from collections import deque
from typing import Any, Optional

class Stack: # Структура данных 'Стек' (LIFO - Last In First Out)
    def __init__(self): # Инициализация пустого стека
        self._data = []
    
    def push(self, item: Any) -> None: # Добавить элемент на вершину стека
        self._data.append(item) # item: Элемент для добавления
    
    def pop(self) -> Any: # Снять верхний элемент стека и вернуть его
        if self.is_empty(): # IndexError: Если стек пуст
            raise IndexError("Невозможно извлечь элемент: стек пуст")
        return self._data.pop()
    
    def peek(self) -> Optional[Any]: # Вернуть верхний элемент без удаления
        if self.is_empty():
            return None # Верхний элемент стека или None если стек пуст
        return self._data[-1]
    
    def is_empty(self) -> bool: # Проверить пуст ли стек
        return len(self._data) == 0 # True если стек пуст, иначе False
    
    def __len__(self) -> int: # Количество элементов в стеке
        return len(self._data) # Количество элементов
    
    def __str__(self) -> str: # Строковое представление стека
        return f"Stack({self._data})"
    
    def __repr__(self) -> str:
        return str(self)

class Queue: # Структура данных 'Очередь' (FIFO - First In First Out)    
    def __init__(self): # Инициализация пустой очереди
        self._data = deque()
    
    def enqueue(self, item: Any) -> None: # Добавить элемент в конец очереди
        self._data.append(item) # item: Элемент для добавления
    
    def dequeue(self) -> Any: # Взять элемент из начала очереди
        if self.is_empty():
            raise IndexError("Невозможно извлечь элемент: очередь пуста") # IndexError: Если очередь пуста
        return self._data.popleft()
    
    def peek(self) -> Optional[Any]: # Вернуть первый элемент без удаления
        if self.is_empty():
            return None # Первый элемент очереди или None если очередь пуста
        return self._data[0]
    
    def is_empty(self) -> bool: # Проверить пуста ли очередь
        return len(self._data) == 0 # True если очередь пуста, иначе False
    
    def __len__(self) -> int: # Количество элементов в очереди
        return len(self._data) # Количество элементов
    
    def __str__(self) -> str: # Строковое представление очереди
        return f"Queue({list(self._data)})"
    
    def __repr__(self) -> str:
        return str(self)