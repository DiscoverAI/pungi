from unittest.mock import MagicMock

from pungi.environment.environment import Environment
from pungi.environment.snake_client import SnakeClient


def test_reset(mocker):
    client = SnakeClient("https://localhost:8080")
    client.register_new_game = MagicMock(return_value="foobar")
    mock_game_state = {
        "game-over": True,
        "board": "mock-board",
        "ate-food": False,
        "score": 10
    }
    get_reward_mock = mocker.patch("pungi.qlearning.get_state_from_game_info")
    get_reward_mock.return_value = [0, 1]

    client.get_game_info = MagicMock(return_value=mock_game_state)
    environment = Environment(client)
    assert [0, 1] == environment.reset()
    get_reward_mock.assert_called_once_with(mock_game_state)


def test_make_step_game_over(mocker):
    client = SnakeClient("https://localhost:8080")
    client.make_move = MagicMock(return_value={
        "game-over": True,
        "board": "mock-board",
        "ate-food": False,
        "score": 10
    })
    get_reward_mock = mocker.patch("pungi.qlearning.get_reward")
    get_reward_mock.return_value = 42

    get_reward_mock = mocker.patch("pungi.qlearning.get_state_from_game_info")
    get_reward_mock.return_value = [0, 1]

    test_environment = Environment(client)
    reward, next_state, done, info = test_environment.step("up")
    assert reward == 42
    assert next_state == [0, 1]
    assert done
    assert info == {"score": 10}


def test_make_step(mocker):
    client = SnakeClient("https://localhost:8080")
    client.make_move = MagicMock(return_value={
        "game-over": False,
        "board": "mock-board",
        "ate-food": False,
        "score": 22
    })

    get_reward_mock = mocker.patch("pungi.qlearning.get_reward")
    get_reward_mock.return_value = 32

    get_reward_mock = mocker.patch("pungi.qlearning.get_state_from_game_info")
    get_reward_mock.return_value = [1, 1]

    test_environment = Environment(client)
    reward, next_state, done, info = test_environment.step("up")
    assert reward == 32
    assert next_state == [1, 1]
    assert not done
    assert info == {"score": 22}
