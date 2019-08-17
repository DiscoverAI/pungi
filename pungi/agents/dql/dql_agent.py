from pungi.agents.agent import Agent
import random
from collections import deque
import pungi.agents.policies as policies
from pungi.agents.qlearning.qlearning import DIRECTIONS


class DQLAgent(Agent):

    def __init__(self, configuration, q_network):
        self.configuration = configuration
        self.replay_memory = None
        self.init_replay_memory(configuration['replay_memory_limit'])
        self.q_network = q_network

    def init_replay_memory(self, limit):
        self.replay_memory = deque(maxlen=limit)

    def next_action(self, state, episode_number):
        # this method has failing tests! TODO:
        # 1. get the prediction from self.q_network
        # 2. label them accordingly in order of DIRECTIONS array
        # e.g. {"left": 0.0, "right": 0.1, etc.}
        # 3. call policies.epsilon_greedy_max_policy
        pass

    def update(self, state, action, nextstate, reward):
        self.replay_memory.append((state, action, nextstate, reward))
        return True

    def sample_memory(self, sample_size):
        min_sample_size = min(sample_size, len(self.replay_memory))
        return random.sample(self.replay_memory, min_sample_size)
