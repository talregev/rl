import numpy as np
import gym


class Environment(gym.Env):

    def __init__(self, matrix_transition, rand_generator):
        self.s = 0
        self.rand_generator = rand_generator
        self.transition = np.array(matrix_transition)
        self.all_state = np.arange(self.transition.shape[1])

    @staticmethod
    def build_matrix_transition(p_array):
        matrix_len = len(p_array)
        matrix_transition = np.zeros((matrix_len, matrix_len, matrix_len))
        for i in range(matrix_len):
            transition = np.full((matrix_len, matrix_len), p_array[i])
            for j in range(matrix_len):
                transition[j, j] = 1 - p_array[i]
            matrix_transition[i] = transition
        return matrix_transition

    def reset(self, seed=None, return_info=False, options=None):
        self.s = 0
        return self.s

    def step(self, action):
        state = self.rand_generator.choice(self.all_state, p=self.transition[action][self.s])
        self.s = state
        return self.s, self.s

    def get_num_states_actions(self):
        matrix_shape = self.transition.shape
        return matrix_shape[1], matrix_shape[0]
