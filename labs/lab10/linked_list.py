from typing import Any, Optional, Iterator

class Node: # Узел односвязного списка
    def __init__(self, value: Any, next_node: Optional['Node'] = None): # Инициализация узла
        self.value = value # value: Значение узла
        self.next = next_node # next_node: Ссылка на следующий узел
    
    def __str__(self) -> str: # Строковое представление узла
        return f"[{self.value}]"
    
    def __repr__(self) -> str:
        return str(self)

class SinglyLinkedList: # Односвязный список
    def __init__(self): # Инициализация пустого списка
        self.head = None
        self.tail = None # Для ускорения операций с концом списка
        self._size = 0
    
    def append(self, value: Any) -> None: # Добавить элемент в конец списка
        new_node = Node(value) # value: Значение для добавления
        
        if self.head is None: # Если список пуст
            self.head = new_node
            self.tail = new_node
        else: # Если в списке уже есть элементы
            self.tail.next = new_node
            self.tail = new_node
        
        self._size += 1
    
    def prepend(self, value: Any) -> None: # Добавить элемент в начало списка
        new_node = Node(value, self.head) # value: Значение для добавления
        self.head = new_node
        
        if self.tail is None: # Если список был пуст
            self.tail = new_node
        
        self._size += 1
    
    def insert(self, idx: int, value: Any) -> None: # Вставить элемент по указанному индексу
                                                    # idx: Индекс для вставки (0 <= idx <= len(list))
                                                    # value: Значение для вставки
        if idx < 0 or idx > self._size:
            raise IndexError(f"Индекс {idx} вне диапазона [0, {self._size}]") # IndexError: Если индекс вне допустимого диапазона
        
        if idx == 0:
            self.prepend(value)
            return
        
        if idx == self._size:
            self.append(value)
            return
        
        # Вставка в середину списка
        current = self.head
        for _ in range(idx - 1):
            current = current.next
        
        new_node = Node(value, current.next)
        current.next = new_node
        self._size += 1
    
    def remove(self, value: Any) -> bool: # Удалить первое вхождение значения
                                          # value: Значение для удаления
        if self.head is None: # True если элемент был найден и удален, иначе False
            return False
        
        # Если нужно удалить первый элемент
        if self.head.value == value:
            self.head = self.head.next
            if self.head is None:  # Если список стал пустым
                self.tail = None
            self._size -= 1
            return True
        
        # Поиск элемента для удаления
        current = self.head
        while current.next is not None and current.next.value != value:
            current = current.next
        
        if current.next is None: # Элемент не найден
            return False
        
        # Удаление найденного элемента
        if current.next == self.tail: # Если удаляем последний элемент
            self.tail = current
        
        current.next = current.next.next
        self._size -= 1
        return True
    
    def remove_at(self, idx: int) -> Any: # Удалить элемент по индексу
                                          # idx: Индекс элемента для удаления
        if idx < 0 or idx >= self._size:
            raise IndexError(f"Индекс {idx} вне диапазона [0, {self._size})") # IndexError: Если индекс вне допустимого диапазона
        
        if idx == 0: # Удаление первого элемента
            value = self.head.value
            self.head = self.head.next
            if self.head is None:
                self.tail = None
            self._size -= 1
            return value
        
        # Удаление из середины или конца
        current = self.head
        for _ in range(idx - 1):
            current = current.next
        
        value = current.next.value
        
        if current.next == self.tail: # Если удаляем последний элемент
            self.tail = current
        
        current.next = current.next.next
        self._size -= 1
        return value
    
    def get(self, idx: int) -> Any: # Получить элемент по индексу
                                    # idx: Индекс элемента
        if idx < 0 or idx >= self._size:
            raise IndexError(f"Индекс {idx} вне диапазона [0, {self._size})") # IndexError: Если индекс вне допустимого диапазона
        
        current = self.head
        for _ in range(idx):
            current = current.next
        
        return current.value
    
    def __iter__(self) -> Iterator[Any]: # Итератор по значениям списка
                                         # Значения списка в порядке от головы к хвосту
        current = self.head
        while current is not None:
            yield current.value
            current = current.next
    
    def __len__(self) -> int: # Количество элементов в списке
        return self._size
    
    def __str__(self) -> str: # Красивое строковое представление списка
        if self.head is None:
            return "None"
        
        parts = []
        current = self.head
        while current is not None:
            parts.append(str(current))
            current = current.next

        return " -> ".join(parts) + " -> None"
    
    def __repr__(self) -> str: # Формальное строковое представление списка
        values = list(self)
        return f"SinglyLinkedList({values})"
    
    def is_empty(self) -> bool: # Проверить пуст ли список
        return self._size == 0 # True если список пуст, иначе False
    
if __name__ == "__main__":
   # Пример использования
   lst = SinglyLinkedList()
   lst.append(1)
   lst.append(2)
   lst.prepend(0)
   lst.insert(2, 1.5)
   lst.remove_at(3)
   print(lst)