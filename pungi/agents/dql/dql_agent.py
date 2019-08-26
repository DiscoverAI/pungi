import numpy as np

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
        self.gamma = float(configuration['gamma'])
        self.batch_size = int(configuration['batch_size'])

    def init_replay_memory(self, limit):
        self.replay_memory = deque(maxlen=limit)

    def next_action(self, state, episode_number):
        prediction = self.q_network.predict(state)[0]
        q_values = {k: v for k, v in zip(DIRECTIONS, prediction)}
        return policies.epsilon_greedy_max_policy(q_values, episode_number)
        # 1. get the prediction from self.q_network
        # 2. label them accordingly in order of DIRECTIONS array
        # e.g. {"left": 0.0, "right": 0.1, etc.}
        # 3. call policies.epsilon_greedy_max_policy
        pass

    def get_q_update(self, reward, game_over, next_state):
        # calculate the q update. if its game_over, our markov chain terminates and we just return the reward.
        # If not, return the simplified q formula (learning rate will be 1)
        pass

    def build_training_examples(self, batch):
        # loop through the batch, which is a list of (state, action, reward, next_state, game_over) tuples
        # add the state to the list of inputs
        # add the prediction of the network, "patched" with the q update to the list of outputs
        # return the two lists as tuple (inputs, outputs)
        pass

    def memory_replay(self):
        # sample batch from memory
        # build the training examples
        # train the network with the training examples
        pass

    def update(self, state, action, nextstate, reward):
        self.replay_memory.append((state, action, nextstate, reward))
        return True

    def sample_memory(self, sample_size):
        min_sample_size = min(sample_size, len(self.replay_memory))
        return random.sample(self.replay_memory, min_sample_size)
