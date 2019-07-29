from pungi.agents.agent import Agent
import random
from collections import deque


class DQLAgent(Agent):
    def __init__(self, configuration=None):
        self.configuration = configuration
        self.replay_memory = None
        self.init_replay_memory(configuration['replay_memory_limit'])

    def next_action(self, state, episode_number):
        pass

    def init_replay_memory(self, limit):
        self.replay_memory = deque(maxlen=limit)

    def update(self, state, action, nextstate, reward):
        self.replay_memory.append((state, action, nextstate, reward))
        return True

    def sample_memory(self, sample_size):
        min_sample_size = min(sample_size, len(self.replay_memory))
        return random.sample(self.replay_memory, min_sample_size)
