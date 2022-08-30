import torch
import torch.nn as nn
from controllers.replay import ReplayMemory


class AgentActorCriticReplay():

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
        capacity,
        batch_size,
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
        self.replay = ReplayMemory(capacity, rand_generator)
        self.batch_size = batch_size

    def loss_actor(self, delta, state, action):
        policy_select = self.policy(state).gather(1, action.unsqueeze(-1))
        return -(self.lr_actor * delta * torch.log(policy_select))

    def loss_critic(self, delta, state):
        return -self.lr_critic * delta * self.value(state)

    def choose(self, s):
        p = self.policy(s).cpu().detach().numpy()
        a = self.rand_generator.choice(self.num_actions, p=p)
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

        with torch.no_grad():
            delta = reward_batch.unsqueeze(-1) - self.reward_tag + \
                self.value(next_state_batch) - self.value(state_batch)
            self.reward_tag += self.lr_reward * torch.sum(delta).item()
        if self.reward_tag != 0:
            tal = 1
        self.update_critic(delta, state_batch)
        self.update_actor(delta, state_batch, action_batch)

    def update_actor(self, delta, state, action):
        pseudo_loss = self.loss_actor(delta, state, action)
        pseudo_loss_mean = torch.mean(pseudo_loss)
        # self.print_gradgraph(pseudo_loss_mean.grad_fn)
        # update policy weights
        self.optimizer_actor.zero_grad()
        pseudo_loss_mean.backward()
        self.optimizer_actor.step()

    def update_critic(self, delta, state):
        pseudo_loss = self.loss_critic(delta, state)
        pseudo_loss_mean = torch.mean(pseudo_loss)

        # update policy weights
        self.optimizer_critic.zero_grad()
        pseudo_loss_mean.backward()
        self.optimizer_critic.step()
