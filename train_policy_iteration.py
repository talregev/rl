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


class Agent():

    def __init__(self, matrix_transition, gamma, theta) -> None:
        self.matrix_transition = matrix_transition
        self.gamma = gamma
        self.theta = theta
        self.g_t = 0

    def set_generator(self, rand_generator):
        self.rand_generator = rand_generator

    def set_num_states_action(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.value = np.zeros((num_states,))
        self.policy = np.full((num_states, num_actions), 1 / num_actions)

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in range(self.num_states):
                value = self.value[state]
                sum_v = 0
                for action in range(self.num_actions):
                    p_policy = self.policy[state][action]
                    sum_p = 0
                    for state_tag in range(self.num_states):
                        p = matrix_transition[action][state][state_tag]
                        r = state_tag
                        sum_p += p * (r + self.gamma * self.value[state_tag])

                    sum_v += p_policy * sum_p
                self.value[state] = sum_v
                delta = max(delta, abs(value - self.value[state]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for state in range(self.num_states):
            prev_actions = self.policy[state]
            p_value = []
            for action in range(self.num_actions):
                sum_p = 0
                for state_tag in range(self.num_states):
                    p = matrix_transition[action][state][state_tag]
                    r = state_tag
                    sum_p += p * (r + self.gamma * self.value[state_tag])
                p_value.append(sum_p)
            a = np.argmax(p_value)
            actions = np.zeros((self.num_actions,))
            actions[a] = 1
            actions_test = prev_actions == actions
            if not actions_test.all():
                policy_stable = False
            self.policy[state] = actions
        return policy_stable

    # def choose(self, s):
    #     p = self.policy(s).cpu().detach().numpy()
    #     a = self.rand_generator.choice(list(range(self.num_actions)), p=p)
    #     return a

    def update(self):
        self.policy_evaluation()
        return self.policy_improvement()


class Controller:

    def __init__(self, agent, episodes, episode_step, gamma, lr, p_array, seed) -> None:
        self.episodes = episodes
        self.episode_step = episode_step
        self.gamma = gamma
        self.lr = lr
        self.p_array = p_array
        self.seed = seed
        self.rand_generator = np.random.RandomState(np.random.seed(self.seed))
        self.matrix_transition = Environment.build_matrix_transition(self.p_array)
        matrix_shape = self.matrix_transition.shape
        self.agent = agent
        self.agent.set_num_states_action(matrix_shape[1], matrix_shape[0])
        self.set_generator(self.rand_generator)
        self.env = Environment(self.matrix_transition, self.rand_generator)
        self.returns = deque(maxlen=100)

    def train(self):
        # reset environment
        state = self.env.reset()
        for n_episode in range(self.episodes):
            rewards = []
            states = []
            for _ in range(episode_step):
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


if __name__ == "__main__":
    print("tal")

    episodes = 500000
    episode_step = 20
    gamma = 0.9
    p_array = [0.9, 0.1]
    seed = 42
    theta = 0.001
    matrix_transition = Environment.build_matrix_transition(p_array)
    agent = Agent(matrix_transition, gamma, theta)
    matrix_shape = matrix_transition.shape
    agent.set_num_states_action(matrix_shape[1], matrix_shape[0])
    policy_stable = False
    step = 0
    while not policy_stable:
        policy_stable = agent.update()
        print(f'step: {step}')
        step += 1
    print('finish')
    print(agent.policy)
    print(agent.value)
    print(matrix_transition)
