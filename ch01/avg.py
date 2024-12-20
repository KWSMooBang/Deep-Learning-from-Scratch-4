import numpy as np

np.random.seed(0)

Q = 0

for n in range(1, 11):
    reward = np.random.rand() # 보상(무작위 수로 시뮬레이션)
    Q = Q + (reward - Q) / n
    
    print(Q)