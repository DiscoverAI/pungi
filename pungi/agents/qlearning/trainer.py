import logging

import pungi.agents.qlearning.policies as policies
import pungi.agents.qlearning.qlearning as qlearning
import pungi.config as conf

logger = logging.getLogger(__name__)


def run_episode(q_table, policy, environment):
    learning_rate = float(conf.CONF.get_value("learning_rate"))
    discount_factor = float(conf.CONF.get_value("discount_factor"))
    state = tuple(environment.reset())
    game_over = False
    last_game_info = None
    while not game_over:
        next_action = qlearning.next_move(q_table, state, policy)
        next_state, reward, game_over, info = environment.step(next_action)
        next_state = tuple(next_state)
        logger.debug('Game info: %s', info)
        q_table = qlearning.update_q_value(q_table,
                                           state,
                                           next_action,
                                           next_state,
                                           learning_rate,
                                           discount_factor,
                                           reward)
        state = next_state
        last_game_info = info
    logger.info('Ended game with info: %s', last_game_info)
    return q_table


def train(env):
    logger.info('Starting training')
    q_table_initial_value = float(conf.CONF.get_value("q_table_initial_value"))
    q_table = qlearning.initialize_q_table(initial_value=q_table_initial_value)
    policy = policies.get_policy(policy_name=conf.CONF.get_value("policy"))
    total_episodes = int(conf.CONF.get_value("episodes"))
    for episode in range(total_episodes):
        logger.info('Starting new episode %s/%s', episode, total_episodes)
        q_table = run_episode(q_table, lambda q_values: policy(q_values, episode), env)
        logger.info('Done with episode %s/%s', episode, total_episodes)
    logger.info('Done with training')
    return q_table
