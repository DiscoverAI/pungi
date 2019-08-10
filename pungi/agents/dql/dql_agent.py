from pungi.agents.agent import Agent
import random
from collections import deque


class DQLAgent(Agent):

    def __init__(self, configuration):
        self.configuration = configuration
        self.replay_memory = None
        self.q_network = None
        self.init_replay_memory(configuration['replay_memory_limit'])

    def load_neural_network_model(self, model_file_name, optimizer):
        pass

    def init_replay_memory(self, limit):
        self.replay_memory = deque(maxlen=limit)

    def next_action(self, state, episode_number):
        # calculate epsilon based on episode number
        # with probability epsilon select random action
        # otherwise run inference on neural network, and pick action with argmax
        pass

    def update(self, state, action, nextstate, reward):
        self.replay_memory.append((state, action, nextstate, reward))
        return True

    def sample_memory(self, sample_size):
        min_sample_size = min(sample_size, len(self.replay_memory))
        return random.sample(self.replay_memory, min_sample_size)
