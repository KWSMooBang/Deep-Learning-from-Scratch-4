import numpy as np

from collections import defaultdict
from common.utils import greedy_probs



class MCAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1      # (첫 번째 개선) eps-탐욕 정책의 eps
        self.alpha = 0.1        # (두 번째 개선) Q 함수 갱신 시의 고정값 alpha
        self.action_size = 4
        
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0) # V가 아닌 Q를 사용
        # self.counts = defaultdict(lambda: 0)
        self.memory = []
        
        
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)
        
    def reset(self):
        self.memory.clear()
        
    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            # self.Q 갱신
            # self.counts[key] += 1
            # self.Q[key] += (G - self.Q[key]) / self.counts[key]
            self.Q[key] += (G - self.Q[key]) * self.alpha
            # state의 정책 탐욕화
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)