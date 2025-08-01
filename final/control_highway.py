import gymnasium as gym
import highway_env

def getch():
    i = input()
    if len(i) == 0:
        return 'n'
    return i[0]

def simulate(render=True):
    model_name = "highway-v0"
    env = gym.make(model_name, render_mode='human' if render else None)
    env.reset()
    done = False
    total_reward = 0
    cnt = 0
    while not done:
        c = getch()
        if c == 'q':
            break

        # 0 LEFT 1 IDLE 2 RIGHT 3 SPEED 4 SLOW
        action = {'j':4,'l':3,'i':0,'k':2}.get(c,1)
        new_state, rew, term, trunc, _ = env.step(action)
        print(rew,term,trunc)
        done = term or trunc
        cnt += 1
        total_reward += rew
    print('total reward',total_reward,cnt)
simulate()