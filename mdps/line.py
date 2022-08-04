import numpy as np


class LineMDP():

    @staticmethod
    def build_matrix_transition(p, line_size):
        # line have 2 directions. 0 left (decrease), 1 right (increase)
        num_actions = 2
        matrix_transition = np.zeros((num_actions, line_size, line_size, 2))

        # Fill the probabilities
        action_left = np.zeros((line_size, line_size, 2))
        for i in range(1, line_size):
            action_left[i][i - 1][0] = p
        for i in range(line_size - 1):
            action_left[i][i + 1][0] = 1 - p
        action_left[0][0][0] = p
        action_left[-1][-1][0] = 1 - p
        matrix_transition[0] = action_left

        action_right = np.zeros((line_size, line_size, 2))
        for i in range(line_size - 1):
            action_right[i][i + 1][0] = p
        for i in range(1, line_size):
            action_right[i][i - 1][0] = 1 - p
        action_right[0][0][0] = 1 - p
        action_right[-1][-1][0] = p
        matrix_transition[1] = action_right

        matrix_transition[..., 1][..., -1] = 1

        return matrix_transition
