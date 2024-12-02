# Playing with OpenAI Gym: CartPole-v0

import time
import gym
import numpy as np

##################################################################################################
# policies


def naive_policy(obs):
    angle = obs[0]
    return 0 if angle.any() < 0 else 1


def random_policy(obs):
    angle = obs[1]
    return 0 if np.random.uniform() < 0.5 else 1


##################################################################################################


def naive_main(policy):
    debug = True
    env = gym.make("CartPole-v0")
    obs = env.reset()
    env.render()

    # episodic reinforcement learning
    totals = []
    for episode in range(100):
        episode_rewards = 0
        obs = env.reset()
        for step in range(10000):
            action = policy(obs)
            obs, reward, terminated, trucated, info = env.step(action)
            env.render()
            time.sleep(0.1)
            episode_rewards += reward
            if terminated or trucated:
                print("Game over. Number of steps = ", step)
                env.render()
                time.sleep(3.14)
        totals.append(episode_rewards)
    print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))


##################################################################################################

if __name__ == "__main__":
    naive_main(naive_policy)

##################################################################################################
