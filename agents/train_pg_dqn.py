import torch
import torch.nn as nn
from utils import Controller, Policy, QValue


class Agent():

    def __init__(self, lr_policy, lr_q, gamma, k) -> None:

        self.lr_policy = lr_policy
        self.lr_q = lr_q
        self.gamma = gamma
        self.k = k
        self.criterion = nn.SmoothL1Loss()

    def set_generator(self, rand_generator):
        self.rand_generator = rand_generator

    def set_num_states_action(self, num_states, num_actions):
        self.policy = Policy(self.k, num_states, num_actions)
        self.q = QValue(k, num_states, num_actions)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters())
        self.q_optimizer = torch.optim.Adam(self.q.parameters())
        self.num_actions = num_actions

    def calc_loss(self, s, a, r, s_tag):
        return -torch.sum(self.lr_policy * torch.log(self.policy(s)) * self.q(s, a))

    def choose(self, s):
        p = self.policy(s).cpu().detach().numpy()
        a = self.rand_generator.choice(list(range(self.num_actions)), p=p)
        return a

    def update(self, state, action, new_state, reward):
        self.update_value(state, action, new_state, reward)

        pseudo_loss = self.calc_loss(state, action, reward, new_state)
        # update policy weights
        self.policy_optimizer.zero_grad()
        pseudo_loss.backward()
        self.policy_optimizer.step()

    def get_expected(self, s, a, r, s_tag):
        values = []
        for a_ in range(2):
            values.append(self.q.forward(s_tag, a_))
        val_max = max(values)
        q_s_a = self.q.forward(s, a)
        expected = q_s_a + self.lr_q * (r + self.gamma * val_max - q_s_a)
        return expected

    def update_value(self, state, action, new_state, reward):
        # calculate current q_value
        q_value = self.q(state, action)
        expected = self.get_expected(state, action, reward, new_state)

        pseudo_loss = self.criterion(q_value, expected)
        # update policy weights
        self.q_optimizer.zero_grad()
        pseudo_loss.backward()
        self.q_optimizer.step()


if __name__ == "__main__":
    print("tal")

    episodes = 500000
    episode_step = 20
    gamma = 0.99
    lr_policy = 0.01
    lr_q = 0.1
    p_array = [0.9, 0.1]
    k = 10
    seed = 100
    agent = Agent(lr_policy, lr_q, gamma, k)
    controller = Controller(agent, episodes, episode_step, gamma, p_array, seed)
    controller.train_td0()
