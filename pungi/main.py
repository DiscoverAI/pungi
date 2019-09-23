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
from pungi.agents import q_network_factory, policies
from pungi.agents.dql.dql_agent import DQLAgent
from pungi.agents.qlearning import greedy_policy_agent as agent
from pungi.agents.qlearning.agent import QLearningAgent
import pungi.environment.environment  # import registers the environment

logger = logging.getLogger(__name__)


def load_default_dqn():
    return q_network_factory.make_simple_sequential(json.load(open("resources/dqn-architecture.json")))


def load_model_from_args(argv):
    if len(argv) <= 2:
        raise FileNotFoundError("Please provide a model path that the agent should use to play.")
    model_file = argv[2]
    if model_file.endswith(".pkl"):
        return persistence.load_q_table(model_file)
    elif model_file.endswith(".h5"):
        dqn = load_default_dqn()
        dqn.load_weights(model_file)
        dqn.compile(optimizer="adam", loss="mse")
        return dqn


def run(argv):
    mode = argv[1]
    env = gym.make(conf.CONF.get_value("gym_environment"))
    if mode == "train":
        model = argv[2]
        if model == "dqn":
            dqn = load_default_dqn()
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
        model = load_model_from_args(argv)
        q_learning_agent = DQLAgent(configuration=dict(conf.CONF),
                                    q_network=model, policy=policies.max_policy)
        agent.play_in_spectator_mode(q_learning_agent, env)
    elif mode == "eval":
        q_table = load_model_from_args(argv)
        metrics.calculate_and_write_metrics(env=env,
                                            episodes=10,
                                            q_table=q_table,
                                            output_path="./out/metrics-" + str(int(time.time())) + ".json")
    else:
        logging.warning("First argument must be either train, test or play.")
        return -1


if __name__ == '__main__':
    run(sys.argv)
