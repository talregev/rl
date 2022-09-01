from copy import deepcopy
import torch
import torch.nn as nn
from controllers.replay import ReplayMemory
import itertools
import numpy as np


class AgentSoftActorCriticReplay():

    def __init__(
        self,
        policy,
        qvalue,
        rand_generator,
        lr_critic,
        lr_actor,
        num_states,
        num_actions,
        capacity,
        batch_size,
        gamma,
        polyak,
    ) -> None:
        self.rand_generator = rand_generator
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.num_actions = num_actions
        self.policy = policy
        self.q1 = deepcopy(qvalue)
        self.q2 = deepcopy(qvalue)
        self.q1_target = deepcopy(qvalue)
        self.q2_target = deepcopy(qvalue)
        q_params = [self.q1.parameters(), self.q2.parameters()]
        self.optimizer_critic = torch.optim.Adam(itertools.chain(*q_params))
        self.optimizer_actor = torch.optim.Adam(self.policy.parameters())
        self.criterion = nn.MSELoss()
        self.average_reward = 0
        self.replay = ReplayMemory(capacity, rand_generator)
        self.batch_size = batch_size
        self.gamma = gamma
        self.polyak = polyak

    def loss_actor(self, state):
        action_probabilities = self.policy(state).cpu().detach().numpy()
        next_actions = np.array([self.rand_generator.choice(self.num_actions, p=p) for p in action_probabilities])
        p = torch.tensor(action_probabilities[np.arange(action_probabilities.shape[0]), next_actions]).unsqueeze(-1)

        q1_pi = self.q1(state, next_actions)
        q2_pi = self.q2(state, next_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (self.lr_actor * p - q_pi).mean()
        return loss_pi

    def loss_critic(self, state, action, reward, next_state):
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)

        with torch.no_grad():
            action_probabilities = self.policy(next_state).cpu().detach().numpy()
            next_actions = np.array([self.rand_generator.choice(self.num_actions, p=p) for p in action_probabilities])
            p = torch.tensor(action_probabilities[np.arange(action_probabilities.shape[0]), next_actions]).unsqueeze(-1)

            q1_pi_target = self.q1_target(next_state, next_actions)
            q2_pi_target = self.q2_target(next_state, next_actions)

            q_pi_target = torch.min(q1_pi_target, q2_pi_target)
            bellman = reward.unsqueeze(-1) + self.gamma * (q_pi_target - self.lr_critic * torch.log(p))

        loss_q1 = self.criterion(q1, bellman)
        loss_q2 = self.criterion(q2, bellman)
        return loss_q1 + loss_q2

        # return -self.lr_critic * delta * self.value(state)

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
        with torch.autograd.detect_anomaly():
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

            self.update_critic(state_batch, action_batch, reward_batch, next_state_batch)
            self.update_actor(state_batch)

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                q_params = [self.q1.parameters(), self.q2.parameters()]
                q_targets_params = [self.q1_target.parameters(), self.q2_target.parameters()]
                for p, p_targ in zip(itertools.chain(*q_params), itertools.chain(*q_targets_params)):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def update_actor(self, state):
        pseudo_loss = self.loss_actor(state,)

        # update policy weights
        self.optimizer_actor.zero_grad()
        pseudo_loss.backward()
        self.optimizer_actor.step()

    def update_critic(self, state, action, reward, next_state):
        pseudo_loss = self.loss_critic(state, action, reward, next_state)

        # update policy weights
        self.optimizer_critic.zero_grad()
        pseudo_loss.backward()
        self.optimizer_critic.step()
