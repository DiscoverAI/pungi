from unittest.mock import MagicMock

from pungi.environment import backend
from pungi.environment import environment
from unittest.mock import patch

mock_game_state = {
    "game-over": True,
    "board": "mock-board",
    "ate-food": False,
    "score": 10
}


@patch('pungi.environment.backend.register_new_game', return_value="foobar")
@patch('pungi.environment.backend.get_game_info', return_value=mock_game_state)
@patch('pungi.qlearning.get_state_from_game_info', return_value=[0, 1])
def test_reset(get_state_mock, get_game_info_mock, register_new_game_mock):
    initialized_game = environment.reset()
    assert "foobar", [0, 1] == initialized_game
    get_game_info_mock.assert_called_once_with("foobar")
    get_state_mock.assert_called_once_with(mock_game_state)
    register_new_game_mock.assert_called_once()


def test_make_step_game_over(mocker):
    backend.make_move = MagicMock(return_value={
        "game-over": True,
        "board": "mock-board",
        "ate-food": False,
        "score": 10
    })
    get_reward_mock = mocker.patch("pungi.qlearning.get_reward")
    get_reward_mock.return_value = 42

    get_reward_mock = mocker.patch("pungi.qlearning.get_state_from_game_info")
    get_reward_mock.return_value = [0, 1]
    reward, next_state, done, info = environment.step("up", "foo bar")
    assert reward == 42
    assert next_state == [0, 1]
    assert done
    assert info == {"score": 10}


def test_make_step(mocker):
    backend.make_move = MagicMock(return_value={
        "game-over": False,
        "board": "mock-board",
        "ate-food": False,
        "score": 22
    })

    get_reward_mock = mocker.patch("pungi.qlearning.get_reward")
    get_reward_mock.return_value = 32

    get_game_state_from_game_info = mocker.patch("pungi.qlearning.get_state_from_game_info")
    get_game_state_from_game_info.return_value = [1, 1]

    reward, next_state, done, info = environment.step("up", "foo bar")
    assert reward == 32
    assert next_state == [1, 1]
    assert not done
    assert info == {"score": 22}
