import numpy as np


class GridMDP():

    @staticmethod
    def build_matrix_transition(p, grid_w, grid_h):
        # line have 4 directions.
        # 0 left (decrease), 1 right (increase)
        # 2 up (decrease) 3 down (increase)
        num_actions = 4
        state_size = grid_w * grid_h
        matrix_transition = np.zeros((num_actions, state_size, state_size, 2))

        # Fill the probabilities
        # action 0: left
        action_left = np.zeros((state_size, state_size, 2))
        for i in range(state_size):
            column = i % grid_w
            if column > 0:
                action_left[i][i - 1][0] = p
                action_left[i][i][0] = 1 - p
            else:
                action_left[i][i][0] = 1

        matrix_transition[0] = action_left

        # action 1: right
        action_right = np.zeros((state_size, state_size, 2))
        for i in range(state_size):
            column = i % grid_w

            if column < (grid_w - 1):
                action_right[i][i + 1][0] = p
                action_right[i][i][0] = 1 - p
            else:
                action_right[i][i][0] = 1

        matrix_transition[1] = action_right

        # action 2: up
        action_up = np.zeros((state_size, state_size, 2))
        for i in range(state_size):
            row = i // grid_w

            if row > 0:
                action_up[i][i - grid_w][0] = p
                action_up[i][i][0] = 1 - p
            else:
                action_up[i][i][0] = 1

        matrix_transition[2] = action_up

        # action 3: down
        action_down = np.zeros((state_size, state_size, 2))
        for i in range(state_size):
            row = i // grid_w

            if row < (grid_h - 1):
                action_down[i][i + grid_w][0] = p
                action_down[i][i][0] = 1 - p
            else:
                action_down[i][i][0] = 1

        matrix_transition[3] = action_down

        matrix_transition[..., 1][..., -1] = 1

        return matrix_transition
