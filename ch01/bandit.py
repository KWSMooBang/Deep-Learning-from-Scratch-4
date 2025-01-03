import numpy as np


class Bandit:
    def __init__(self, arms=10): 
        """
        Args:
            arms: the number of slot machines
        """
        self.rates = np.random.rand(arms) # 슬롯머신 각각의 승률 설정(무작위)
        
    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1 
        else:
            return 0
        
    
class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)
        
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]
        
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs)) # 무작위 행동 선택
        return np.argmax(self.Qs) # 탐욕 행동 선택