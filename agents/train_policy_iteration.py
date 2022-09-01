import numpy as np
from environments.environment import Environment


class Agent():

    def __init__(self, matrix_transition, gamma, theta) -> None:
        self.matrix_transition = matrix_transition
        self.gamma = gamma
        self.theta = theta
        self.g_t = 0

    def set_generator(self, rand_generator):
        self.rand_generator = rand_generator

    def set_num_states_action(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.value = np.zeros((num_states,))
        self.policy = np.full((num_states, num_actions), 1 / num_actions)

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in range(self.num_states):
                value = self.value[state]
                sum_v = 0
                for action in range(self.num_actions):
                    p_policy = self.policy[state][action]
                    sum_p = 0
                    for state_tag in range(self.num_states):
                        p = matrix_transition[action][state][state_tag]
                        r = state_tag
                        sum_p += p * (r + self.gamma * self.value[state_tag])

                    sum_v += p_policy * sum_p
                self.value[state] = sum_v
                delta = max(delta, abs(value - self.value[state]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for state in range(self.num_states):
            prev_actions = self.policy[state]
            p_value = []
            for action in range(self.num_actions):
                sum_p = 0
                for state_tag in range(self.num_states):
                    p = matrix_transition[action][state][state_tag]
                    r = state_tag
                    sum_p += p * (r + self.gamma * self.value[state_tag])
                p_value.append(sum_p)
            a = np.argmax(p_value)
            actions = np.zeros((self.num_actions,))
            actions[a] = 1
            actions_test = prev_actions == actions
            if not actions_test.all():
                policy_stable = False
            self.policy[state] = actions
        return policy_stable

    def update(self):
        self.policy_evaluation()
        return self.policy_improvement()


if __name__ == "__main__":
    print("tal")

    episodes = 500000
    episode_step = 20
    gamma = 0.9
    p_array = [0.9, 0.1]
    seed = 42
    theta = 0.001
    matrix_transition = Environment.build_matrix_transition(p_array)
    agent = Agent(matrix_transition, gamma, theta)
    matrix_shape = matrix_transition.shape
    agent.set_num_states_action(matrix_shape[1], matrix_shape[0])
    policy_stable = False
    step = 0
    while not policy_stable:
        policy_stable = agent.update()
        print(f'step: {step}')
        step += 1
    print('finish')
    print(agent.policy)
    print(agent.value)
    print(matrix_transition)
