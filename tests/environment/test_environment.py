from pungi.environment import environment
from unittest.mock import patch
import gym
import numpy as np

game_over_game_state = {
    "game-over": True,
    "board": [[0, 3], [1, 0]],
    "ate-food": False,
    "score": 10
}

game_in_progress_game_state = {
    "game-over": False,
    "board": [[3, 0], [1, 0]],
    "ate-food": False,
    "score": 22
}

ENV_NAME = "snake-v1"
PARTIAL_INFO_ENV_NAME= "snake-partial-info-v1"

@patch('pungi.environment.backend.register_new_game', return_value="foobar")
@patch('pungi.environment.backend.get_game_info', return_value=game_over_game_state)
def test_reset(get_game_info_mock, register_new_game_mock):
    env = gym.make(PARTIAL_INFO_ENV_NAME)
    initial_state = env.reset()
    assert "foobar", [0, 1] == initial_state
    get_game_info_mock.assert_called_once_with("foobar")
    register_new_game_mock.assert_called_once()


@patch('pungi.environment.backend.register_new_game', return_value="foobar")
@patch('pungi.environment.backend.get_game_info', return_value=game_over_game_state)
@patch('pungi.environment.backend.make_move', return_value=game_over_game_state)
def test_make_step_game_over(make_move_mock, get_game_info, _register_mock):
    env = gym.make(ENV_NAME)
    env.reset()
    next_state, reward, done, info = env.step("up")
    assert reward == 42
    assert (next_state == np.array([[0, 3], [1, 0]])).all()
    assert done
    assert info == {"score": 10}
    make_move_mock.assert_called_once_with("up", "foobar")


@patch('pungi.environment.backend.register_new_game', return_value="foobar")
@patch('pungi.environment.backend.get_game_info', return_value=game_over_game_state)
@patch('pungi.environment.backend.make_move', return_value=game_in_progress_game_state)
def test_make_step(make_move_mock, get_game_info, _register_mock):
    env = gym.make(ENV_NAME)
    env.reset()
    next_state, reward, done, info = env.step("left")
    assert reward == 32
    assert (next_state == np.array([[3, 0], [1, 0]])).all()
    assert not done
    assert info == {"score": 22}
    make_move_mock.assert_called_once_with("left", "foobar")
