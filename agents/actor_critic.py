import torch
import torch.nn as nn


class AgentActorCritic():

    def __init__(
        self,
        policy,
        value,
        rand_generator,
        lr_reward,
        lr_critic,
        lr_actor,
        num_states,
        num_actions,
    ) -> None:
        self.rand_generator = rand_generator
        self.lr_reward = lr_reward
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.num_actions = num_actions
        self.policy = policy
        self.value = value
        self.optimizer_critic = torch.optim.Adam(self.value.parameters())
        self.optimizer_actor = torch.optim.Adam(self.policy.parameters())
        self.criterion = nn.SmoothL1Loss()
        self.reward_tag = 0

    def loss_actor(self, delta, state, action):
        return -(self.lr_actor * delta * torch.log(self.policy(state)[action]))

    def loss_critic(self, delta, state):
        return -self.lr_critic * delta * self.value(state)

    def choose(self, s):
        p = self.policy(s).cpu().detach().numpy()
        a = self.rand_generator.choice(self.num_actions, p=p)
        return a

    def step(self, state, action, new_state, reward):

        delta = (reward - self.reward_tag + self.value(new_state) - self.value(state)).item()
        self.reward_tag += self.lr_reward * delta
        self.update_critic(delta, state)
        self.update_actor(delta, state, action)

    def update_actor(self, delta, state, action):
        pseudo_loss = self.loss_actor(delta, state, action)

        # update policy weights
        self.optimizer_actor.zero_grad()
        pseudo_loss.backward()
        self.optimizer_actor.step()

    def update_critic(self, delta, state):
        pseudo_loss = self.loss_critic(delta, state)

        # update policy weights
        self.optimizer_critic.zero_grad()
        pseudo_loss.backward()
        self.optimizer_critic.step()
