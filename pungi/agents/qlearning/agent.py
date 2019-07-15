import time
import webbrowser
import logging

from pungi.agents.qlearning import policies, qlearning
from pungi.config import CONF

logger = logging.getLogger(__name__)




def play_game(q_table, on_before, on_step, environment):
    game_id, state = environment.reset()
    game_over = False
    policy = policies.get_policy(policy_name="max_policy")
    on_before(game_id)
    total_reward = 0
    while not game_over:
        next_action = qlearning.next_move(q_table, state, lambda q_values: policy(q_values, None))
        reward, state, game_over, info = environment.step(next_action)
        total_reward = total_reward + reward
        on_step()
    return total_reward


def initialize_spectator_mode(game_id):
    spectate_url = CONF.get_value("backend") + '/?spectate-game-id=' + game_id
    logger.info("Playing game, you can spectate it at: {}".format(spectate_url))
    webbrowser.open_new(spectate_url)


def play_in_spectator_mode(q_table):
    return play_game(q_table,
                     on_before=initialize_spectator_mode,
                     on_step=lambda: time.sleep(0.5))


def play_in_background(q_table, environment):
    return play_game(q_table,
                     on_before=lambda game_id: None,
                     on_step=lambda: None, environment=environment)
