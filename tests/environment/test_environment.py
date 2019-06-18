from pungi.environment import environment
from unittest.mock import patch
import tests.mock_states

game_over_game_state = {
    "game-over": True,
    "board": "mock-board",
    "ate-food": False,
    "score": 10
}

game_in_progress_game_state = {
    "game-over": False,
    "board": "mock-board",
    "ate-food": False,
    "score": 22
}


@patch('pungi.environment.backend.register_new_game', return_value="foobar")
@patch('pungi.environment.backend.get_game_info', return_value=game_over_game_state)
@patch('pungi.environment.state.globals', return_value={"mocked_state_extractor": tests.mock_states.mocked_state_extractor})
def test_reset(_globals_mock, get_game_info_mock, register_new_game_mock):
    initialized_game = environment.reset()
    assert "foobar", [0, 1] == initialized_game
    get_game_info_mock.assert_called_once_with("foobar")
    register_new_game_mock.assert_called_once()


@patch('pungi.environment.backend.make_move', return_value=game_over_game_state)
@patch('pungi.agents.qlearning.qlearning.get_reward', return_value=42)
@patch('pungi.environment.state.globals', return_value={"mocked_state_extractor": tests.mock_states.mocked_state_extractor})
def test_make_step_game_over(_globals, get_reward_mock, make_move_mock):
    reward, next_state, done, info = environment.step("up", "foo bar")
    assert reward == 42
    assert next_state == [0, 1]
    assert done
    assert info == {"score": 10}
    get_reward_mock.assert_called_once_with(game_over_game_state)
    make_move_mock.assert_called_once_with("up", "foo bar")


@patch('pungi.environment.backend.make_move', return_value=game_in_progress_game_state)
@patch('pungi.agents.qlearning.qlearning.get_reward', return_value=32)
@patch('pungi.environment.state.globals', return_value={"mocked_state_extractor": tests.mock_states.mocked_state_extractor})
def test_make_step(_globals, get_reward_mock, make_move_mock):
    reward, next_state, done, info = environment.step("left", "bar foo")
    assert reward == 32
    assert next_state == [0, 1]
    assert not done
    assert info == {"score": 22}
    get_reward_mock.assert_called_once_with(game_in_progress_game_state)
    make_move_mock.assert_called_once_with("left", "bar foo")
