import numpy as np
from agents.dqn import AgentDQN
from controllers.controller import Controller
from networks.qvalue import QValue
from environments.environment import Environment
from mdps.simple_mdp import SimpleMDP

if __name__ == "__main__":
    print("dqn")

    episodes = 5000
    episode_step = 20
    gamma = 0.8
    epsilon = 0.05
    lr = 0.1
    p_array = [0.9, 0.1]
    k = 10
    seed = 42
    rand_generator = np.random.RandomState(np.random.seed(seed))
    matrix_transition = SimpleMDP.build_matrix_transition(p_array)
    env = Environment(matrix_transition, rand_generator)
    num_states, num_actions = env.get_num_states_actions()
    q_network = QValue(k, num_states, num_actions)
    agent = AgentDQN(rand_generator, q_network, num_states, num_actions, lr, gamma, epsilon)
    controller = Controller(env, agent, episodes, episode_step)
    controller.train_td0()
