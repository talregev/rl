import numpy as np
import torch
import torch.nn as nn


class AgentDQN():

    def __init__(self, rand_generator, q_network, num_states, num_actions, lr, gamma, epsilon) -> None:
        self.rand_generator = rand_generator
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = q_network
        self.num_states = num_states
        self.num_actions = num_actions
        self.optimizer = torch.optim.Adam(self.q.parameters())
        # Compute Huber loss
        self.criterion = nn.SmoothL1Loss()

    def get_expected(self, s, a, r, s_tag):
        values = []
        for a_ in range(self.num_actions):
            values.append(self.q.forward(s_tag, a_))
        val_max = max(values)
        q_s_a = self.q.forward(s, a)
        expected = q_s_a + self.lr * (r + self.gamma * val_max - q_s_a)
        return expected

    def choose(self, s):
        a = None
        if self.rand_generator.random() < self.epsilon:
            a = self.rand_generator.choice(self.num_actions)
        else:
            values = []
            for a_ in range(self.num_actions):
                values.append(self.q(s, a_).item())
            a = np.argmax(values)
        return a

    def step(self, state, action, new_state, reward):
        # calculate current q_value
        q_value = self.q(state, action)
        expected = self.get_expected(state, action, reward, new_state)

        pseudo_loss = self.criterion(q_value, expected)
        # update policy weights
        self.optimizer.zero_grad()
        pseudo_loss.backward()
        self.optimizer.step()
