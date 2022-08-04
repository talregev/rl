import numpy as np


class SimpleMDP():

    @staticmethod
    def build_matrix_transition(p_array):
        matrix_len = len(p_array)
        matrix_transition = np.zeros((matrix_len, matrix_len, matrix_len, 2))
        for i in range(matrix_len):
            transition = np.full((matrix_len, matrix_len, 2), [p_array[i], 0])
            # fill the probabilities
            for j in range(matrix_len):
                transition[j, j, 0] = 1 - p_array[i]
            # fill the reward
            for k in range(matrix_len):
                transition[..., 1][..., k] = k
            matrix_transition[i] = transition
        return matrix_transition
