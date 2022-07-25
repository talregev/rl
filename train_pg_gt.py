import torch
from utils import Controller, Policy


class Agent():

    def __init__(self, lr, gamma, k) -> None:

        self.lr = lr
        self.gamma = gamma
        self.k = k
        self.g_t = 0

    def set_generator(self, rand_generator):
        self.rand_generator = rand_generator

    def set_num_states_action(self, num_states, num_actions):
        self.policy = Policy(self.k, num_states, num_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters())
        self.num_actions = num_actions

    def calc_loss(self, s, a, r, s_tag):
        return -torch.sum(self.lr * torch.log(self.policy(s)) * self.g_t)

    def choose(self, s):
        p = self.policy(s).cpu().detach().numpy()
        a = self.rand_generator.choice(list(range(self.num_actions)), p=p)
        return a

    def update(self, state_list, action, new_state, reward_list):
        self.g_t = 0
        for state, reward in zip(reversed(state_list), reversed(reward_list)):
            self.g_t = self.g_t * self.gamma + reward
        pseudo_loss = self.calc_loss(state, action, reward, new_state)
        # update policy weights
        self.optimizer.zero_grad()
        pseudo_loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    print("tal")

    episodes = 500000
    episode_step = 20
    gamma = 0.99
    lr = 0.01
    p_array = [0.9, 0.1]
    k = 10
    seed = 42
    agent = Agent(lr, gamma, k)
    controller = Controller(agent, episodes, episode_step, gamma, lr, p_array, seed)
    controller.train()
