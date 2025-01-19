
def value_iter_onestep(V, env, gamma):
    for state in env.states():       # 모든 상태에 차례로 접근
        if state == env.goal_state:  # 목표 상테에서의 가치 함수는 항상 0
            V[state] = 0
            continue
        
        action_values = []
        for action in env.actions(): # 모든 행동에 차례로 접근
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)
            
        V[state] = max(action_values)
    return V


def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    while True:
        if is_render:
            env.render_v(V)
            
        old_V = V.copy()    # 갱신 전 가치 함수
        V = value_iter_onestep(V, env, gamma)
        
        # 갱신된 양의 최댓값 구하기
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
                
        # 임곗값과 비교
        if delta < threshold:
            break
    
    return V
            