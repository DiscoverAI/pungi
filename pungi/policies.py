import random

import numpy as np

import pungi.config as conf
from pungi.qlearning import DIRECTIONS


def max_policy(q_values, episode_number):
    return max(q_values, key=q_values.get)


def epsilon_greedy_max_policy(q_values, episode_number):
    initial_epsilon = float(conf.CONF.get_value("initial_epsilon_greedy"))
    epsilon_factor = float(conf.CONF.get_value("epsilon_factor_per_episode"))
    current_epsilon = initial_epsilon * (epsilon_factor ** episode_number)
    return np.random.choice([max(q_values, key=q_values.get), random.choice(DIRECTIONS)],
                            p=[1 - current_epsilon, current_epsilon])


def get_policy(policy_name):
    return globals()[policy_name]
