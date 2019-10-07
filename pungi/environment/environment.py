import logging

import pungi.agents.qlearning.qlearning as qlearning
import pungi.config as conf
import pungi.environment.backend as backend
import pungi.environment.state as state

logger = logging.getLogger('root')

import gym

from gym import spaces
import numpy as np


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.current_game_id = None
        self.observation_space = spaces.Box(0,
                                            3,
                                            (int(conf.CONF.get_value("board_width")),
                                             int(conf.CONF.get_value("board_height"))))
        self.action_space = spaces.Discrete(4)
        self._ACTION_TO_STRING = {0: "left", 1: "right", 2: "up", 3: "down"}

    def step(self, action):
        action_to_string = self._ACTION_TO_STRING.get(action, action)
        next_state = backend.make_move(action_to_string, self.current_game_id)
        reward = self.get_reward(next_state)
        game_over = next_state["game-over"]
        info = {"score": next_state["score"]}
        return np.array(next_state["board"], dtype=np.int8) * (1 / 3), reward, game_over, info

    def reset(self):
        self.current_game_id = backend.register_new_game()
        game_info = backend.get_game_info(self.current_game_id)
        return np.array(game_info["board"], dtype=np.int8) * (1 / 3)

    def render(self, mode='human', close=False):
        pass  # Here one can log human readable state of the environment

    def close(self):
        pass

    @staticmethod
    def get_reward(game_state):
        if game_state["game-over"]:
            return float(conf.CONF.get_value("game_over_reward"))
        elif game_state["ate-food"]:
            return float(conf.CONF.get_value("ate_food_reward"))
        else:
            return float(conf.CONF.get_value("snake_moved_reward"))


class PartialInformationSnakeEnv(SnakeEnv):
    def step(self, action):
        next_state, reward, done, info = super().step(action)
        return np.array(state.extract_head_and_food(next_state)), reward, done, info

    def reset(self):
        return np.array(state.extract_head_and_food(super().reset()))
