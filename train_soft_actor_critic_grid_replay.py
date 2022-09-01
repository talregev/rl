import numpy as np
from agents.soft_actor_critic_replay import AgentSoftActorCriticReplay
from controllers.controller import Controller
from networks.policy import Policy
from networks.qvalue import QValue
from environments.environment import Environment
from mdps.grid import GridMDP

if __name__ == "__main__":
    print("tal")

    episodes = 50000
    episode_step = 7
    lr_reward = 0.9
    lr_critic = 0.3
    lr_actor = 0.3
    p_array = [0.9, 0.1]
    k = 10
    p = 0.9
    grid_w = 5
    grid_h = 4
    seed = 42
    capacity = 1000
    batch_size = 128
    gamma = 0.99
    polyak = 0.995
    rand_generator = np.random.RandomState(np.random.seed(seed))
    matrix_transition = GridMDP.build_matrix_transition(p, grid_w, grid_h)
    env = Environment(matrix_transition, rand_generator)
    num_states, num_actions = env.get_num_states_actions()
    policy = Policy(k, num_states, num_actions)
    qvalue = QValue(k, num_states, num_actions)
    agent = AgentSoftActorCriticReplay(
        policy,
        qvalue,
        rand_generator,
        lr_critic,
        lr_actor,
        num_states,
        num_actions,
        capacity,
        batch_size,
        gamma,
        polyak,
    )
    controller = Controller(env, agent, episodes, episode_step)
    controller.train_replay()
