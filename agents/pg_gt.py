import torch


class Agent():

    def __init__(self, rand_generator, num_states, num_actions, policy: torch.nn.Module, lr, gamma) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.rand_generator = rand_generator
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters())

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
            pseudo_loss = self.calc_loss(state, gamma_t, g_t)
            # update policy weights
            self.optimizer.zero_grad()
            pseudo_loss.backward()
            self.optimizer.step()

            # update gamma_t
            gamma_t *= self.gamma
