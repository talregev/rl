import numpy as np
from agents.pg_gt import Agent
from controllers.controller import Controller
from networks.policy import Policy
from environments.simple_mdp import Environment

if __name__ == "__main__":
    print("train policy gradient gt")

    episodes = 500000
    episode_step = 20
    gamma = 0.99
    lr = 0.3
    p_array = [0.9, 0.1]
    k = 10
    seed = 42
    rand_generator = np.random.RandomState(np.random.seed(seed))
    matrix_transition = Environment.build_matrix_transition(p_array)
    env = Environment(matrix_transition, rand_generator)
    num_states, num_actions = env.get_num_states_actions()
    policy = Policy(k, num_states, num_actions)
    agent = Agent(rand_generator, num_states, num_actions, policy, lr, gamma)
    controller = Controller(env, agent, episodes, episode_step)
    controller.train_mc()
