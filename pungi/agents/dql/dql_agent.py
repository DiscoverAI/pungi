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

    def get_q_update(self, reward, game_over, next_state):
        if game_over:
            return reward
        else:
            prediction = self.q_network.predict(next_state)[0]
            return reward + self.gamma * max(prediction)

    def build_training_examples(self, batch):
        output_batch = []
        input_batch = []
        for (state, action, reward, next_state, game_over) in batch:
            q_update = self.get_q_update(reward, game_over, next_state)
            predictions = self.q_network.predict(state)[0]
            predictions[action] = q_update
            input_batch.append(state)
            output_batch.append(predictions)
        return input_batch, output_batch

    def memory_replay(self):
        batch = self.sample_memory(self.batch_size)
        input_batch, output_batch = self.build_training_examples(batch)
        self.q_network.fit(input_batch, output_batch)

    def update(self, state, action, nextstate, reward):
        self.replay_memory.append((state, action, nextstate, reward))
        return True

    def sample_memory(self, sample_size):
        min_sample_size = min(sample_size, len(self.replay_memory))
        return random.sample(self.replay_memory, min_sample_size)
