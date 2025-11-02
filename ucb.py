import math
from collections import deque
import json
import threading
import random


class UCB:
    def __init__(self, save_path, history_len=32, exploration=0.5):
        self.save_path = save_path
        self.history_len = history_len
        self.exploration = exploration

        self.rewards = {}
        self.history = []
        self.t = 1

        self.thread_lock = threading.Lock()

    def select_arm(self, valid_arms=None):
        with self.thread_lock:
            ucb_values = {}
            for arm, rewards in self.rewards.items():
                if valid_arms is not None and arm not in valid_arms:
                    continue
                counts = len(rewards)
                if counts == 0:
                    ucb = 100
                else:
                    avg_reward = sum(rewards) / counts
                    bonus = math.sqrt(2 * math.log(self.t) / counts)
                    ucb = avg_reward + self.exploration * bonus
                ucb_values[arm] = ucb
            sorted_arms = sorted(ucb_values.items(), key=lambda x: (x[1], random.random()), reverse=True)
            sorted_arms, sorted_scores = [i[0] for i in sorted_arms], [i[1] for i in sorted_arms]
            return sorted_arms[0], sorted_arms, sorted_scores

    def add_arm(self, arm):
        with self.thread_lock:
            if arm not in self.rewards:
                self.rewards[arm] = deque(maxlen=self.history_len)
            self.save_state()

    def update_arm(self, arm, reward):
        with self.thread_lock:
            self.t = min(self.t + 1, self.history_len)
            self.history.append((arm, reward))
            self.rewards[arm].append(reward)
            self.save_state()
    
    def reset_time(self):
        with self.thread_lock:
            self.t = 1
    
    def save_state(self):
        if self.save_path.exists():
            self.save_path.rename(self.save_path.with_suffix(f'.json.old'))
        with open(self.save_path, 'w') as f:
            json.dump({
                "rewards": {arm: list(rewards) for arm, rewards in self.rewards.items()},
                "history": self.history,
                "t": self.t
            }, f, indent=4)

    def load_state(self):
        with self.thread_lock:
            if not self.save_path.exists():
                return
            with open(self.save_path, 'r') as f:
                data = json.load(f)
                self.rewards = {arm: deque(rewards, maxlen=self.history_len) for arm, rewards in data["rewards"].items()}
                self.history = data["history"]
                self.t = data["t"]

