from collections import namedtuple, deque
import random


class ReplayMemory(object):
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def __init__(self, capacity, rand_generator):
        self.memory = deque([], maxlen=capacity)
        self.rand_generator = rand_generator

    def push(self, *args):
        """Save a transition"""
        self.memory.append(ReplayMemory.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
