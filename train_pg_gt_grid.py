import numpy as np
from agents.pg_gt import AgentPolicyGradientGt
from controllers.controller import Controller
from networks.policy import Policy
from environments.environment import Environment
from mdps.grid import GridMDP

if __name__ == "__main__":
    print("train policy gradient gt")

    episodes = 500000
    episode_step = 7
    gamma = 0.99
    lr = 0.3
    p = 0.9
    grid_w = 5
    grid_h = 4
    k = 10
    seed = 42
    rand_generator = np.random.RandomState(np.random.seed(seed))
    matrix_transition = GridMDP.build_matrix_transition(p, grid_w, grid_h)
    env = Environment(matrix_transition, rand_generator)
    num_states, num_actions = env.get_num_states_actions()
    policy = Policy(k, num_states, num_actions)
    agent = AgentPolicyGradientGt(rand_generator, num_states, num_actions, policy, lr, gamma)
    controller = Controller(env, agent, episodes, episode_step)
    controller.train_mc()
