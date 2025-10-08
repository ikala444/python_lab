import time
def MyFunc():
    start_time = time.time()
    a = [1 for i in range(1000000)]
    for g in a:
        pass
    end_time = time.time()
    return end_time - start_time
def MyFunc1():
    start_time = time.time()
    a = (1 for i in range(1000000))
    for g in a:
        pass
    end_time = time.time()
    return end_time - start_time
print(MyFunc())
print(MyFunc1())