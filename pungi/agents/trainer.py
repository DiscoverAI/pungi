import logging

import pungi.config as conf

logger = logging.getLogger(__name__)


def run_episode(agent, environment, episode_number):
    state = tuple(environment.reset())
    game_over = False
    last_game_info = None
    while not game_over:
        action = agent.next_action(state, episode_number)
        next_state, reward, game_over, info = environment.step(action)
        next_state = tuple(next_state)
        logger.debug('Game info: %s', info)
        agent.update(state, action, next_state, reward)
        state = next_state
        last_game_info = info
    logger.info('Ended game with info: %s', last_game_info)


def train(agent, env):
    logger.info('Starting training')
    total_episodes = int(conf.CONF.get_value("episodes"))
    for episode in range(total_episodes):
        logger.info('Starting new episode %s/%s', episode, total_episodes)
        run_episode(agent, env, episode)
        logger.info('Done with episode %s/%s', episode, total_episodes)
    logger.info('Done with training')
    return agent
