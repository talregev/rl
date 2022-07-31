import torch
from utils import Controller, Policy


class Agent():

    def __init__(self, lr, gamma, k) -> None:

        self.lr = lr
        self.gamma = gamma
        self.k = k

    def set_generator(self, rand_generator):
        self.rand_generator = rand_generator

    def set_num_states_action(self, num_states, num_actions):
        self.policy = Policy(self.k, num_states, num_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters())
        self.num_actions = num_actions

    def calc_loss(self, s, gamma_t, g_t):
        return -torch.sum(self.lr * gamma_t * g_t * torch.log(self.policy(s)))

    def choose(self, s):
        p = self.policy(s).cpu().detach().numpy()
        a = self.rand_generator.choice(list(range(self.num_actions)), p=p)
        return a

    def step(self, state_list, action_list, reward_list):
        gamma_t = 1
        for t in range(len(state_list)):
            g_t = 0
            for k in reversed(range(t, len(state_list))):
                g_t = g_t * self.gamma + reward_list[k]
            state = state_list[t]
            # action = action_list[t]
            # reward = reward_list[t]
            pseudo_loss = self.calc_loss(state, gamma_t, g_t)
            # update policy weights
            self.optimizer.zero_grad()
            pseudo_loss.backward()
            self.optimizer.step()
            gamma_t *= self.gamma


if __name__ == "__main__":
    print("tal")

    episodes = 500000
    episode_step = 20
    gamma = 0.99
    lr = 0.3
    p_array = [0.9, 0.1]
    k = 10
    seed = 42
    agent = Agent(lr, gamma, k)
    controller = Controller(agent, episodes, episode_step, gamma, p_array, seed)
    controller.train_mc()
