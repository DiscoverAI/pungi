import numpy as np

from pungi.agents.agent import Agent
import random
from collections import deque
import pungi.agents.policies as policies
from pungi.agents.qlearning.qlearning import DIRECTIONS
import time

STRING_TO_ACTION = {"left": 0, "right": 1, "up": 2, "down": 3}


class DQLAgent(Agent):

    def on_after_episode(self):
        self.memory_replay()

    def __init__(self, configuration, q_network, policy=policies.epsilon_greedy_max_policy):
        self.configuration = configuration
        self.replay_memory = None
        self.init_replay_memory(configuration['replay_memory_limit'])
        self.q_network = q_network
        self.gamma = float(configuration['gamma'])
        self.batch_size = int(configuration['batch_size'])
        self.policy = policy

    def init_replay_memory(self, limit):
        self.replay_memory = deque(maxlen=limit)

    def next_action(self, state, episode_number):
        prediction = self.q_network.predict(state.reshape(1, *state.shape, 1))[0]
        q_values = {k: v for k, v in zip(DIRECTIONS, prediction)}
        return self.policy(q_values, episode_number)

    def get_q_update(self, reward, game_over, next_state):
        if game_over:
            return reward
        else:
            prediction = self.q_network.predict(next_state.reshape(1, *next_state.shape, 1))[0]
            return reward + self.gamma * max(prediction)

    def build_training_examples(self, batch):
        output_batch = []
        input_batch = []
        for (state, action, next_state, reward, game_over) in batch:
            q_update = self.get_q_update(reward, game_over, next_state)
            predictions = self.q_network.predict(state.reshape(1, *state.shape, 1))[0]
            predictions[STRING_TO_ACTION[action]] = q_update
            input_batch.append(state)
            output_batch.append(predictions)
        return input_batch, output_batch

    def memory_replay(self):
        batch = self.sample_memory(self.batch_size)
        input_batch, output_batch = self.build_training_examples(batch)
        x = np.array(input_batch)
        y = np.array(output_batch)
        self.q_network.fit(x.reshape((-1, *x.shape[1:], 1)), y, verbose=0)

    def update(self, state, action, next_state, reward, game_over):
        self.replay_memory.append((state, action, next_state, reward, game_over))
        return True

    def sample_memory(self, sample_size):
        min_sample_size = min(sample_size, len(self.replay_memory))
        return random.sample(self.replay_memory, min_sample_size)

    def persist(self, path_to_output_folder):
        self.q_network.save(path_to_output_folder + "/" + str(int(time.time())) + ".h5")

