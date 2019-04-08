import pungi.qlearning as qlearning
import pungi.environment.backend as backend
import logging


def reset():
    game_id = backend.register_new_game()
    game_info = backend.get_game_info(game_id)
    return game_id, qlearning.get_state_from_game_info(game_info)


def step(action, game_id):
    next_state = backend.make_move(action, game_id)
    reward = qlearning.get_reward(next_state)
    next_position = qlearning.get_state_from_game_info(next_state)
    game_over = next_state["game-over"]
    info = {"score": next_state["score"]}
    return reward, next_position, game_over, info
