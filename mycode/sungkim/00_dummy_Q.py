import gym
import numpy as np
import matplotlib.pyplot as plt
import random


def rargmax(vector):
    m = np.argmax(vector)
    asd = vector == m
    inx = np.nonzero(vector == m)
    if len(inx) == 0:
        print(vector)
        print(m)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(indices)


gym.envs.registration.register(
    id="FrozenLake-v3",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False},
)

env = gym.make("FrozenLake-v3")
Q = np.zeros((env.observation_space.n, env.action_space.n))

rewardlist = []
num_episodes = 2000
for i in range(num_episodes):
    state = env.reset()
    rewardall = 0
    done = False

    while not done:
        action = rargmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)
        Q[state, action] = reward + np.max(Q[new_state, :])
        rewardall += reward
        state = new_state

    rewardlist.append(rewardall)

print(f"success rate: {sum(rewardlist)/num_episodes}")
print("Final Q-table values")
print("left down right up")
print(Q)
plt.bar(range(len(rewardlist)), rewardlist, color="blue")
plt.show()
