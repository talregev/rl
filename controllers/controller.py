import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class Controller:

    def __init__(self, environnement, agent, episodes, episode_step) -> None:
        self.episodes = episodes
        self.episode_step = episode_step
        self.agent = agent
        self.env = environnement
        self.returns = deque(maxlen=100)
        self.writer = SummaryWriter()

    def train_td0(self):
        # reset environment
        state = self.env.reset()
        for n_episode in range(self.episodes):
            rewards = []
            for _ in range(self.episode_step):
                action = self.agent.choose(state)
                new_state, reward = self.env.step(action)
                state = new_state
                self.agent.step(state, action, new_state, reward)
                rewards.append(reward)

            # calculate average return and print it out
            self.returns.append(np.sum(rewards))
            print(f"Episode: {n_episode:6d}\tAvg. Return: { np.mean(self.returns):6.2f}")

    def train_mc(self):
        # reset environment
        state = self.env.reset()
        for n_episode in range(self.episodes):
            rewards_list = []
            states_list = []
            actions_list = []
            for _ in range(self.episode_step):
                states_list.append(state)
                action = self.agent.choose(state)
                state_next, reward = self.env.step(action)
                state = state_next
                rewards_list.append(reward)
                actions_list.append(action)
            self.agent.step(states_list, actions_list, rewards_list)
            state = self.env.reset()

            sum_episode = np.sum(rewards_list)
            # calculate average return and print it out
            self.returns.append(sum_episode)
            avg_sum_episode = np.mean(self.returns)
            print(f"Episode: {n_episode:6d}\tAvg. Return: {avg_sum_episode:6.2f}")
            self.writer.add_scalar("train/sum", sum_episode, n_episode)
            self.writer.add_scalar("train/avg", avg_sum_episode, n_episode)
