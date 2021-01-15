import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import torch


def get_best_action(actions):
    best_action = 0
    max_action_value = 0
    for k in range(len(actions)):
        cur_action_value = get_action_value(actions[k])
        if cur_action_value > max_action_value:
            best_action = k
            max_action_value = cur_action_value
    return best_action


def get_reward(prob, n_=10):
    reward_ = 0
    for _ in range(n_):
        if np.random.rand() < prob:
            reward_ += 1
    return reward_


n = 10
probs = np.random.rand(n)
eps = 0.1
reward_test = [get_reward(0.7) for i in range(2000)]
print(np.mean(reward_test))

plt.figure(figsize=(9, 5))
plt.xlabel("Reward", fontsize=22)
plt.ylabel("# Observations", fontsize=22)
plt.hist(reward_test, bins=9)
plt.show()


def get_best_arm(record_):
    arm_index = np.argmax(record_[:, 1], axis=0)
    return arm_index


def update_record(record_, action, r_):
    new_r = (record_[action, 0] * record_[action, 1] + r_) / (record_[action, 0] + 1)
    record_[action, 0] += 1
    record_[action, 1] = new_r
    return record_


rewards = [0]
record = np.zeros((n, 2))
for i in range(500):
    if np.random.rand() > 0.2:
        choice = get_best_arm(record)
    else:
        choice = np.random.randint(10)
    r = get_reward(probs[choice])
    record = update_record(record, choice, r)
    mean_reward = ((i + 1) * rewards[-1] + r) / (i + 2)
    rewards.append(mean_reward)
print(probs)
print(record)

p_arg = np.argmax(probs)
r_arg = np.argmax(record[:, 1], axis=0)
print(p_arg, r_arg)
print(f"max.arm {p_arg} lever and prob. is {probs[p_arg]:.2f}")
print(f"max. {r_arg}th lever, trial no. {record[r_arg, 0]}, mean reward {record[r_arg, 1]}")

fig, ax = plt.subplots(1, 1)
plt.figure(figsize=(8, 2))
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward")
ax.scatter(np.arange(len(rewards)), rewards)
plt.show()


def softmax(av, tau=1.12):
    softm = np.exp(av / tau) / np.sum(np.exp(av / tau))
    return softm


_, ax = plt.subplots(1, 1)
plt.figure(figsize=(8, 2))
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward")
rewards = [0]
record = np.zeros((n, 2))
for i in range(500):
    p = softmax(record[:, 1], tau=0.7)
    choice = np.random.choice(np.arange(n), p=p)
    r = get_reward(probs[choice])
    record = update_record(record, choice, r)
    mean_reward = ((i + 1) * rewards[-1] + r) / (i + 2)
    rewards.append(mean_reward)
ax.scatter(np.arange(len(rewards)), rewards)
plt.show()


class ContextBandit:
    def __init__(self, arms_=10):
        self.arms = arms_
        # Num states = Num Arms to keep things simple
        self.bandit_matrix = np.random.rand(arms_, arms_)
        # each row represents a state, each column an arm
        self.state = np.random.randint(0, arms_)

    def reward(self, prob):
        reward_ = 0
        for i in range(self.arms):
            if np.random.rand() < prob:
                reward_ += 1
        return reward_

    def get_state(self):
        return self.state

    def update_state(self):
        self.state = np.random.randint(0, self.arms)

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])

    def choose_arm(self, arm):
        reward_ = self.get_reward(arm)
        self.update_state()
        return reward_


arms = 10
N, D_in, H, D_out = 1, arms, 100, arms

env = ContextBandit(arms_=10)
state = env.get_state()
reward = env.choose_arm(1)
print(state)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.ReLU(),
)
loss_fn = torch.nn.MSELoss()
env = ContextBandit(arms)


def one_hot(N_, pos, val=1):
    one_hot_vec = np.zeros(N_)
    one_hot_vec[pos] = val
    return one_hot_vec


def running_mean(x, N_=50):
    c = x.shape[0] - N_
    y = np.zeros(c)
    conv = np.ones(N_)
    for i in range(c):
        y[i] = (x[i : i + N_] @ conv) / N_
    return y


def train(env_, epochs=5000, learning_rate=1e-2):
    cur_state = torch.Tensor(one_hot(arms, env_.get_state()))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rewards_ = []
    for i in range(epochs):
        y_pred = model(cur_state)
        av_softmax = softmax(y_pred.data.numpy(), tau=2.0)
        av_softmax /= av_softmax.sum()
        choice_ = np.random.choice(arms, p=av_softmax)
        cur_reward = env_.choose_arm(choice_)
        one_hot_reward = y_pred.data.numpy().copy()
        one_hot_reward[choice_] = cur_reward
        reward_ = torch.Tensor(one_hot_reward)
        rewards_.append(cur_reward)
        loss = loss_fn(y_pred, reward_)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_state = torch.Tensor(one_hot(arms, env_.get_state()))
    return np.array(rewards_)


rewards = train(env)
plt.plot(running_mean(rewards, N_=500))
plt.show()
