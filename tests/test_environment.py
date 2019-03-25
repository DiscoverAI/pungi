from pungi.Environment import Environment
from unittest.mock import MagicMock
from pungi.snake_client import SnakeClient


def test_game_over_false():
    test_environment = Environment({
        "encoded-board": {
            "game-over": False
        }
    }, None)

    assert test_environment.is_game_over() is False


def test_game_over_true():
    test_environment = Environment({
        "encoded-board": {
            "game-over": True
        }
    }, None)

    assert test_environment.is_game_over() is True


def test_make_step_game_over(mocker):
    client = SnakeClient("https://localhost:8080")
    client.make_move = MagicMock(return_value={"encoded-board": {
        "game-over": True,
        "board": "mock-board",
        "ate-food": False,
        "score": 10
    }})
    client.is_game_over = MagicMock(return_value=True)
    get_reward_mock = mocker.patch("pungi.qlearning.get_reward")
    get_reward_mock.return_value = 42

    get_reward_mock = mocker.patch("pungi.qlearning.get_state_from_game_info")
    get_reward_mock.return_value = [0, 1]

    test_environment = Environment(None, client)
    reward, next_state, done, info = test_environment.step("up")
    assert reward == 42
    assert next_state == [0, 1]
    assert done
    assert info == {"score": 10}


def test_make_step(mocker):
    client = SnakeClient("https://localhost:8080")
    client.make_move = MagicMock(return_value={"encoded-board": {
        "game-over": False,
        "board": "mock-board",
        "ate-food": False,
        "score": 22
    }})
    client.is_game_over = MagicMock(return_value=False)

    get_reward_mock = mocker.patch("pungi.qlearning.get_reward")
    get_reward_mock.return_value = 32

    get_reward_mock = mocker.patch("pungi.qlearning.get_state_from_game_info")
    get_reward_mock.return_value = [1, 1]

    test_environment = Environment(None, client)
    reward, next_state, done, info = test_environment.step("up")
    assert reward == 32
    assert next_state == [1, 1]
    assert not done
    assert info == {"score": 22}