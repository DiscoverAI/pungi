from pungi.agents.agent import Agent
import pungi.config as conf
from pungi.agents.qlearning import qlearning
import pungi.agents.policies as policies
import pungi.persistence as persistence
import time


class QLearningAgent(Agent):
    def __init__(self):
        q_table_initial_value = float(conf.CONF.get_value("q_table_initial_value"))
        self.q_table = qlearning.initialize_q_table(initial_value=q_table_initial_value)
        self.policy = policies.get_policy(policy_name=conf.CONF.get_value("policy"))
        self.learning_rate = float(conf.CONF.get_value("learning_rate"))
        self.discount_factor = float(conf.CONF.get_value("discount_factor"))

    def next_action(self, state, episode_number):
        return qlearning.next_move(self.q_table,
                                   tuple(state),
                                   lambda q_values: self.policy(q_values, episode_number))

    def update(self, state, action, next_state, reward, game_over):
        self.q_table = qlearning.update_q_value(self.q_table,
                                                tuple(state),
                                                action,
                                                tuple(next_state),
                                                self.learning_rate,
                                                self.discount_factor,
                                                reward)

    def persist(self, path_to_output_folder):
        persistence.save_q_table(path_to_output_folder + "/" + str(int(time.time())) + ".pkl", self.q_table)
