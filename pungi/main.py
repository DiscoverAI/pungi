#!/usr/bin/env python3
import json
import logging
import sys
import time

import gym
import pungi.agents.trainer as trainer
import pungi.config as conf
import pungi.metrics as metrics
import pungi.persistence as persistence
from pungi.agents import q_network_factory
from pungi.agents.dql.dql_agent import DQLAgent
from pungi.agents.qlearning import greedy_policy_agent as agent
from pungi.agents.qlearning.agent import QLearningAgent
import pungi.environment.environment  # import registers the environment

logger = logging.getLogger(__name__)


def load_q_table_from_args(argv):
    if len(argv) <= 2:
        raise FileNotFoundError("Please provide a model path that the agent should use to play.")
    q_table_file = argv[2]
    q_table = persistence.load_q_table(q_table_file)
    return q_table


def run(argv):
    mode = argv[1]
    env = gym.make(conf.CONF.get_value("gym_environment"))
    if mode == "train":
        model = argv[2]
        if model == "dqn":
            dqn = q_network_factory.make_simple_sequential(json.load(open("resources/dqn-architecture.json")))
            dqn.compile(optimizer="adam", loss="mse")
            q_learning_agent = DQLAgent(configuration=dict(conf.CONF),
                                        q_network=dqn)
        elif model == "q_table":
            q_learning_agent = QLearningAgent()
        else:
            raise ValueError("Invalid value for model. Usage: main.py train <dqn|q_table>")
        q_learning_agent = trainer.train(q_learning_agent, env)
        q_learning_agent.persist("./out")
    elif mode == "play":
        q_table = load_q_table_from_args(argv)
        agent.play_in_spectator_mode(q_table, env)
    elif mode == "eval":
        q_table = load_q_table_from_args(argv)
        metrics.calculate_and_write_metrics(env=env,
                                            episodes=10,
                                            q_table=q_table,
                                            output_path="./out/metrics-" + str(int(time.time())) + ".json")
    else:
        logging.warning("First argument must be either train, test or play.")
        return -1


if __name__ == '__main__':
    run(sys.argv)
