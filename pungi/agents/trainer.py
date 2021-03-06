import logging

import pungi.config as conf

logger = logging.getLogger('root')


def run_episode(agent, environment, episode_number):
    state = environment.reset()
    game_over = False
    last_game_info = None
    while not game_over:
        action = agent.next_action(state, episode_number)
        next_state, reward, game_over, info = environment.step(action)
        logger.debug('Game info: %s', info)
        agent.update(state, action, next_state, reward, game_over)
        state = next_state
        last_game_info = info
    agent.on_after_episode()
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
