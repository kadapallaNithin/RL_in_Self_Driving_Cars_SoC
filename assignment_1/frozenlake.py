import gymnasium as gym
import random
import json
import numpy as np

def run(render=False, episodes=1000, is_training=False):
    # map_name = "8x8"
    map_name = "4x4"
    model_file = f'frozen_lake_{map_name}.json'
    eps = 1
    eps_decay = 0.001
    gamma = 0.99
    lr = 0.2

    q = {}
    if not is_training:
        tq = json.load(open(model_file))
        for k,v in tq.items():
            q[int(k)] = v
    # v = [0]*25
    env = gym.make('FrozenLake-v1', render_mode='human' if render else None, map_name=map_name, is_slippery=True)
    for i in range(episodes):
        env.reset()
        done = False
        trunc = False
        state = 0 # TODO
        while not done and not trunc:
            if state not in q:
                q[state] = [0] * env.action_space.n
            if is_training and eps > random.random():
                action = env.action_space.sample()  # Take a random action
            else:
                action = np.argmax(q[state])
            new_state,r,done,trunc,prob = env.step(action)  # Apply the action to the environment
            if r != 0:
                print(r)
            else:
                r = -0.01
            
            # v_new = v[new_state]
            if new_state in q:
                v_new = np.max(q[new_state])
            else:
                v_new = 0
            if is_training:
                target = r + gamma*v_new
                qsa = q[state][action]
                q[state][action] += lr*(target - qsa)
            state = new_state
        if is_training:
            eps = max(0, eps-eps_decay)
        env.close()

    if is_training:
        json.dump(q, open(model_file,'w'))
# episodes, is_training = 15000, True
episodes, is_training = 1, False
render = not is_training

run(render, episodes, is_training=is_training)