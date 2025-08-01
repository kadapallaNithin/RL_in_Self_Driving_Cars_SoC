from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import highway_env

from gymnasium.wrappers import RecordVideo
import gymnasium as gym

def test(custom=False):
    env = gym.make("highway-v0", render_mode='rgb_array')
    name_prefix = "highway"
    if custom:
        env.unwrapped.config["vehicles_count"] = 30
        env.unwrapped.config["lanes_count"] = 3
        env.reset()
        name_prefix = "custom_highway"

    env = RecordVideo(env, video_folder="videos", name_prefix=name_prefix, episode_trigger=lambda x: True)

    model_path = 'models/ppo_highway_cpu'
    model = PPO.load(model_path)
    for i in range(5):
        obs, _ = env.reset()
        print('reset')
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            done = term or trunc
        print(i, 'total reward',total_reward)


def test_vec_env(vec_env, model_path):
    model = PPO.load(model_path)
    for i in range(5):
        obs = vec_env.reset()
        print('reset')
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            total_reward += rewards[0]
            done = dones[0]
            # vec_env.render("human")
        print(i, 'total reward',total_reward)


def train():
    env = make_vec_env("highway-v0", n_envs=8, vec_env_cls=SubprocVecEnv)
    model_path = 'models/ppo_highway_cpu'
    if os.path.exists(model_path+'.zip'):
        print('loading model')
        model = PPO.load(model_path, env=env)
    else:
        model = PPO("MlpPolicy", env, device="cpu", verbose=1)
    for i in range(10):
        print('Learning')
        model.learn(total_timesteps=10_000)
        print('Saving',i)
        model.save(model_path)

if __name__=="__main__":
    # train()
    test(custom=False)

