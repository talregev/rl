import numpy as np
import torch
import torch.nn as nn
from controllers.replay import ReplayMemory


class AgentDQNReplay():

    def __init__(self, rand_generator, q_network, num_states, num_actions, lr, gamma, epsilon, capacity,
                 batch_size) -> None:
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
        self.replay = ReplayMemory(capacity, rand_generator)
        self.batch_size = batch_size

    def get_expected(self, state, action, reward, next_state):
        values = []
        for a_ in range(self.num_actions):
            a_vector = torch.tensor(np.full(len(next_state), a_))
            values.append(self.q(next_state, a_vector))
        values = torch.cat(values, -1)
        val_max = torch.max(values, dim=-1).values.unsqueeze(-1)
        q_s_a = self.q(state, action)
        expected = q_s_a + self.lr * (reward.unsqueeze(-1) + self.gamma * val_max - q_s_a)
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

    def push(self, state, action, next_state, reward):
        self.replay.push(
            torch.tensor(state).unsqueeze(0),
            torch.tensor(action).unsqueeze(0),
            torch.tensor(next_state).unsqueeze(0),
            torch.tensor(reward).unsqueeze(0),
        )

    def step(self):
        if len(self.replay) < self.batch_size:
            return
        transitions = self.replay.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = ReplayMemory.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
        #                               dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        q_values = self.q(state_batch, action_batch)
        expected = self.get_expected(state_batch, action_batch, reward_batch, next_state_batch)

        pseudo_loss = self.criterion(q_values, expected)
        # update policy weights
        self.optimizer.zero_grad()
        pseudo_loss.backward()
        self.optimizer.step()
