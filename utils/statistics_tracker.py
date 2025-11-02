import wandb
import threading


class StatisticsTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.counter = {}
        self.thread_lock = threading.Lock()
    
    def load(self, wandb_history):
        self.counter = {k: v for k, v in wandb_history.iloc[-1].to_dict().items() if not k.startswith("_")}
    
    def add(self, key, val=None):
        with self.thread_lock:
            if key not in self.counter:
                self.counter[key] = 0
            self.counter[key] += 1 if val is None else val
            return self.counter[key]
    
    def set(self, key, value):
        with self.thread_lock:
            self.counter[key] = value
            return self.counter[key]
    
    def get(self, key):
        with self.thread_lock:
            if key not in self.counter:
                self.counter[key] = 0
            return self.counter[key]

    def log(self):
        with self.thread_lock:
            wandb.log(self.counter)

