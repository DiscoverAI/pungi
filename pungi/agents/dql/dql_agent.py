from pungi.agents.agent import Agent
import random
from collections import deque


class DQLAgent(Agent):
    def next_action(self, state, episode_number):
        pass

    def __init__(self):
        self.replay_memory = deque()

    def update(self, state, action, nextstate, reward):
        self.replay_memory.append((state, action, nextstate, reward))
        return True

    def sample_memory(self, sample_size):
        min_sample_size = min(sample_size, len(self.replay_memory))
        return random.sample(self.replay_memory, min_sample_size)
