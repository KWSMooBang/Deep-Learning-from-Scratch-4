import numpy as np


def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    # 이 시점에서 action_probs는 {0: eps/4, 1: eps/4, 2: eps/4, 3: eps/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs # 탐욕 행동을 위하는 확률 분포 반환
            