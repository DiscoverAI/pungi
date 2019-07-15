from pungi.agents.agent import Agent
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
        return [self.deque_memory.pop()]
