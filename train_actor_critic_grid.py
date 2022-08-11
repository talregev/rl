import numpy as np
from agents.actor_critic import AgentActorCritic
from controllers.controller import Controller
from networks.policy import Policy
from networks.value import Value
from environments.environment import Environment
from mdps.grid import GridMDP

if __name__ == "__main__":
    print("tal")

    episodes = 50000
    episode_step = 7
    lr_reward = 0.3
    lr_critic = 0.3
    lr_actor = 0.3
    p_array = [0.9, 0.1]
    k = 10
    p = 0.9
    grid_w = 5
    grid_h = 4
    seed = 42
    rand_generator = np.random.RandomState(np.random.seed(seed))
    matrix_transition = GridMDP.build_matrix_transition(p, grid_w, grid_h)
    env = Environment(matrix_transition, rand_generator)
    num_states, num_actions = env.get_num_states_actions()
    policy = Policy(k, num_states, num_actions)
    value = Value(k, num_states)
    agent = AgentActorCritic(policy, value, rand_generator, lr_reward, lr_critic, lr_actor, num_states, num_actions)
    controller = Controller(env, agent, episodes, episode_step)
    controller.train_td0()
