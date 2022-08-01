import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class Environment:

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

    def reset(self):
        self.s = 0
        return self.s

    def step(self, action):
        state = self.rand_generator.choice(self.all_state, p=self.transition[action][self.s])
        self.s = state
        return self.s, self.s


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


class Agent():

    def __init__(self, rand_generator, lr, gamma, epsilon, k, num_states, num_actions) -> None:
        self.rand_generator = rand_generator
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = QValue(k, num_states, num_actions)
        self.optimizer = torch.optim.Adam(self.q.parameters())
        # Compute Huber loss
        self.criterion = nn.SmoothL1Loss()

    def get_expected(self, s, a, r, s_tag):
        values = []
        for a_ in range(2):
            values.append(self.q.forward(s_tag, a_))
        val_max = max(values)
        q_s_a = self.q.forward(s, a)
        expected = q_s_a + self.lr * (r + self.gamma * val_max - q_s_a)
        return expected

    def choose(self, s):
        a = None
        if self.rand_generator.random() < self.epsilon:
            a = self.rand_generator.choice(list(range(2)))
        else:
            values = []
            for a_ in range(2):
                values.append(self.q(s, a_).item())
            a = np.argmax(values)
        return a

    def update(self, state, action, new_state, reward):
        # calculate current q_value
        q_value = self.q(state, action)
        expected = self.get_expected(state, action, reward, new_state)

        pseudo_loss = self.criterion(q_value, expected)
        # update policy weights
        self.optimizer.zero_grad()
        pseudo_loss.backward()
        self.optimizer.step()


class Controller:

    def __init__(self, episodes, gamma, epsilon, lr, p_array, k, seed) -> None:
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.p_array = p_array
        self.k = k
        self.seed = seed
        self.rand_generator = np.random.RandomState(np.random.seed(self.seed))
        self.matrix_transition = Environment.build_matrix_transition(self.p_array)
        matrix_shape = self.matrix_transition.shape
        self.agent = Agent(
            self.rand_generator,
            self.lr,
            self.gamma,
            self.epsilon,
            self.k,
            matrix_shape[1],
            matrix_shape[0],
        )
        self.env = Environment(self.matrix_transition, self.rand_generator)
        self.returns = deque(maxlen=100)

    def train(self):
        # reset environment
        state = self.env.reset()
        for n_episode in range(1, self.episodes + 1):
            rewards = []
            for _ in range(20):
                action = self.agent.choose(state)
                new_state, reward = self.env.step(action)
                self.agent.update(state, action, new_state, reward)
                state = new_state
                rewards.append(reward)

            # calculate average return and print it out
            self.returns.append(np.sum(rewards))
            print(f"Episode: {n_episode:6d}\tAvg. Return: { np.mean(self.returns):6.2f}")


if __name__ == "__main__":
    print("tal")

    episodes = 5000
    gamma = 0.99
    epsilon = 0.02
    lr = 0.1
    p_array = [0.9, 0.1]
    k = 10
    seed = 42
    controller = Controller(episodes, gamma, epsilon, lr, p_array, k, seed)
    controller.train()
