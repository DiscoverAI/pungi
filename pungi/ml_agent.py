import pungi.config as conf
import pungi.environment.environment as env
import pungi.qlearning as qlearning
import pungi.policies as policies

import webbrowser
import time


def play_in_spectator_mode(q_table):
    registered_gameid, state = env.reset()
    webbrowser.open_new(conf.CONF.get_value("backend") + '/?spectate-game-id=' + registered_gameid)
    game_over = False
    policy = policies.get_policy(policy_name="max_policy")
    while not game_over:
        next_action = qlearning.next_move(q_table, state, lambda q_values: policy(q_values, None))
        reward, state, game_over, info = env.step(next_action, registered_gameid)
        time.sleep(0.5)


def play_in_background(q_table):
    registered_gameid, state = env.reset()
    game_over = False
    policy = policies.get_policy(policy_name="max_policy")
    total_reward = 0
    while not game_over:
        next_action = qlearning.next_move(q_table, state, lambda q_values: policy(q_values, None))
        reward, state, game_over, info = env.step(next_action, registered_gameid)
        total_reward = total_reward + reward
    return total_reward

