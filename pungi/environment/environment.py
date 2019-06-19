import logging

import pungi.agents.qlearning.qlearning as qlearning
import pungi.config as conf
import pungi.environment.backend as backend
import pungi.environment.state as state

logger = logging.getLogger(__name__)


def reset():
    state_extractor = state.get_state_extractor(conf.CONF.get_value("state_extractor_function"))
    game_id = backend.register_new_game()
    logger.info('Registered new game: %s',game_id)
    game_info = backend.get_game_info(game_id)
    return game_id, state_extractor(game_info)


def step(action, game_id):
    state_extractor = state.get_state_extractor(conf.CONF.get_value("state_extractor_function"))
    next_state = backend.make_move(action, game_id)
    reward = qlearning.get_reward(next_state)
    next_position = state_extractor(next_state)
    game_over = next_state["game-over"]
    info = {"score": next_state["score"]}
    return reward, next_position, game_over, info
