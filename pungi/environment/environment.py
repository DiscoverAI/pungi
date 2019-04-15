import pungi.qlearning as qlearning
import pungi.environment.backend as backend
import pungi.config as conf

state_extractor = conf.CONF.get_state_extractor(conf.CONF.get_value("state_extractor_function"))


def reset():
    game_id = backend.register_new_game()
    game_info = backend.get_game_info(game_id)
    return game_id, state_extractor(game_info)


def step(action, game_id):
    next_state = backend.make_move(action, game_id)
    reward = qlearning.get_reward(next_state)
    next_position = state_extractor(next_state)
    game_over = next_state["game-over"]
    info = {"score": next_state["score"]}
    return reward, next_position, game_over, info
