import time


class Timer:
    def __init__(self):
        self.times = {}
    
    def start(self, name):
        self.times[name] = time.time()
    
    def end(self, name):
        if name not in self.times:
            return
        self.times[name] = time.time() - self.times[name]
        print(f"> Time for {name}: {self.times[name]:.2f}s")
        return self.times[name]


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__}: {end_time - start_time:.2f}s")
        return result
    return wrapper

