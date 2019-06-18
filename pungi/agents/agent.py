import time
import webbrowser

from pungi.agents import policies, qlearning
from pungi.config import CONF
from pungi.environment import environment


def play_game(q_table, on_before, on_step):
    game_id, state = environment.reset()
    game_over = False
    policy = policies.get_policy(policy_name="max_policy")
    on_before(game_id)
    total_reward = 0
    while not game_over:
        next_action = qlearning.next_move(q_table, state, lambda q_values: policy(q_values, None))
        reward, state, game_over, info = environment.step(next_action, game_id)
        total_reward = total_reward + reward
        on_step()
    return total_reward


def play_in_spectator_mode(q_table):
    return play_game(q_table,
                     on_before=lambda game_id: webbrowser.open_new(
                         CONF.get_value("backend") + '/?spectate-game-id=' + game_id),
                     on_step=lambda: time.sleep(0.5))


def play_in_background(q_table):
    return play_game(q_table,
                     on_before=lambda game_id: None,
                     on_step=lambda: None)