import numpy as np
import gym


class Environment(gym.Env):

    def __init__(self, matrix_transition, rand_generator):
        self.s = 0
        self.rand_generator = rand_generator
        self.transition = np.array(matrix_transition)
        self.num_state = self.transition.shape[1]

    def reset(self, seed=None, return_info=False, options=None):
        self.s = 0
        return self.s

    def step(self, action):
        state = self.rand_generator.choice(self.num_state, p=self.transition[action][self.s][..., 0])
        reward = self.transition[action][self.s][state][1]
        self.s = state
        return state, reward

    def get_num_states_actions(self):
        matrix_shape = self.transition.shape
        return matrix_shape[1], matrix_shape[0]
