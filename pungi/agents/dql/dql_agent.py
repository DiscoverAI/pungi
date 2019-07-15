from pungi.agents.agent import Agent
import random
from collections import deque


class DQLAgent(Agent):
    def next_action(self, state, reward):
        pass

    def __init__(self):
        self.deque_memory = deque()

    def remember(self, state, action, nextstate, reward):
        self.deque_memory.append((state, action, nextstate, reward))
        return True

    def sample_memory(self, sample_size):
        min_sample_size = min(sample_size, len(self.deque_memory))
        return random.sample(self.deque_memory, min_sample_size)
