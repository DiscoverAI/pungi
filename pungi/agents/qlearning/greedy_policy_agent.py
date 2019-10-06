import time
import webbrowser
import logging

from pungi.agents.qlearning import qlearning
from pungi.agents import policies
from pungi.config import CONF

logger = logging.getLogger('root')


def play_game(agent, on_before, on_step, environment):
    state = environment.reset()
    game_over = False
    on_before(environment.current_game_id)
    total_reward = 0
    while not game_over:
        next_action = agent.next_action(state, episode_number=-1)
        state, reward, game_over, info = environment.step(next_action)
        total_reward = total_reward + reward
        on_step()
    return total_reward


def initialize_spectator_mode(game_id):
    spectate_url = CONF.get_value("backend") + '/?spectate-game-id=' + str(game_id)
    logger.info("Playing game, you can spectate it at: {}".format(spectate_url))
    webbrowser.open_new(spectate_url)


def play_in_spectator_mode(agent, environment):
    return play_game(agent,
                     on_before=initialize_spectator_mode,
                     on_step=lambda: time.sleep(0.5),
                     environment=environment)


def play_in_background(agent, environment):
    return play_game(agent,
                     on_before=lambda game_id: None,
                     on_step=lambda: None,
                     environment=environment)
