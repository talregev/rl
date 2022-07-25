import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import gym


class Environment(gym.Env):

    def __init__(self, matrix_transition, rand_generator):
        self.s = 0
        self.rand_generator = rand_generator
        self.transition = np.array(matrix_transition)
        self.all_state = np.arange(self.transition.shape[1])

    @staticmethod
    def build_matrix_transition(p_array):
        matrix_len = len(p_array)
        matrix_transition = np.zeros((matrix_len, matrix_len, matrix_len))
        for i in range(matrix_len):
            transition = np.full((matrix_len, matrix_len), p_array[i])
            for j in range(matrix_len):
                transition[j, j] = 1 - p_array[i]
            matrix_transition[i] = transition
        return matrix_transition

    def reset(self, seed=None, return_info=False, options=None):
        self.s = 0
        return self.s

    def step(self, action):
        state = self.rand_generator.choice(self.all_state, p=self.transition[action][self.s])
        self.s = state
        return self.s, self.s


class Policy(nn.Module):

    def __init__(self, k, num_states, num_actions) -> None:
        super(Policy, self).__init__()
        len_s = len(F.one_hot(torch.tensor(0), num_classes=num_states))
        self.fc1 = nn.Linear(len_s, k)
        self.fc2 = nn.Linear(k, num_actions)
        self.num_states = num_states
        self.num_actions = num_actions

    def forward(self, s):
        input = F.one_hot(torch.tensor(s), num_classes=self.num_states).float()
        x1 = self.fc1(input)
        # Use the rectified-linear activation function over x
        x2 = F.relu(x1)
        x3 = self.fc2(x2)
        output = F.softmax(x3, dim=0)
        return output


class QValue(nn.Module):

    def __init__(self, k, num_states, num_actions) -> None:
        super(QValue, self).__init__()
        len_s = len(F.one_hot(torch.tensor(0), num_classes=num_states))
        len_a = len(F.one_hot(torch.tensor(0), num_classes=num_actions))
        self.fc1 = nn.Linear(len_s + len_a, k)
        self.fc2 = nn.Linear(k, 1)
        self.num_states = num_states
        self.num_actions = num_actions

    def forward(self, s, a):
        input_s = F.one_hot(torch.tensor(s), num_classes=self.num_states).float()
        input_a = F.one_hot(torch.tensor(a), num_classes=self.num_actions).float()
        x = torch.cat((input_s, input_a))
        x = self.fc1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        x = self.fc2(x)
        output = F.relu(x)
        return output


class Controller:

    def __init__(self, agent, episodes, episode_step, gamma, p_array, seed) -> None:
        self.episodes = episodes
        self.episode_step = episode_step
        self.gamma = gamma
        self.p_array = p_array
        self.seed = seed
        self.rand_generator = np.random.RandomState(np.random.seed(self.seed))
        self.matrix_transition = Environment.build_matrix_transition(self.p_array)
        matrix_shape = self.matrix_transition.shape
        self.agent = agent
        self.agent.set_num_states_action(matrix_shape[1], matrix_shape[0])
        self.agent.set_generator(self.rand_generator)
        self.env = Environment(self.matrix_transition, self.rand_generator)
        self.returns = deque(maxlen=100)

    def train_td0(self):
        # reset environment
        state = self.env.reset()
        for n_episode in range(self.episodes):
            rewards = []
            for _ in range(self.episode_step):
                action = self.agent.choose(state)
                new_state, reward = self.env.step(action)
                state = new_state
                self.agent.update(state, action, new_state, reward)
                rewards.append(reward)

            # calculate average return and print it out
            self.returns.append(np.sum(rewards))
            print(f"Episode: {n_episode:6d}\tAvg. Return: { np.mean(self.returns):6.2f}")

    def train_mc(self):
        # reset environment
        state = self.env.reset()
        for n_episode in range(self.episodes):
            rewards = []
            states = []
            for _ in range(self.episode_step):
                states.append(state)
                action = self.agent.choose(state)
                new_state, reward = self.env.step(action)
                state = new_state
                rewards.append(reward)
            self.agent.update(states, action, new_state, rewards)
            state = self.env.reset()

            # calculate average return and print it out
            self.returns.append(np.sum(rewards))
            print(f"Episode: {n_episode:6d}\tAvg. Return: { np.mean(self.returns):6.2f}")
