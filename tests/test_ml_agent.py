from unittest.mock import patch, call, ANY
import pungi.ml_agent as ml_agent


@patch("webbrowser.open_new")
@patch('pungi.environment.environment.step', side_effect=[
    (-1, [3, 3], False, {"score": 10}),
    (100, [2, 3], False, {"score": 11}),
    (-100, [1, 3], True, {"score": 11})
])
@patch('pungi.qlearning.next_move', return_value="left")
@patch('pungi.environment.environment.reset', return_value=("foobar", [4, 3]))
def test_play_game(reset, next_move, step, open_webbrowser):
    ml_agent.play_in_spectator_mode("trained_q_table")
    reset.assert_called_once()
    next_move.assert_has_calls([call("trained_q_table", [4, 3], ANY),
                                call("trained_q_table", [3, 3], ANY),
                                call("trained_q_table", [2, 3], ANY)])
    step.assert_has_calls([call("left", "foobar")] * 3)
    open_webbrowser.assert_called_once_with("foo bar/?spectate-game-id=foobar")


@patch('pungi.environment.environment.step', side_effect=[
    (-1, [3, 3], False, {"score": 10}),
    (100, [2, 3], False, {"score": 11}),
    (-100, [1, 3], True, {"score": 11})
])
@patch('pungi.qlearning.next_move', return_value="left")
@patch('pungi.environment.environment.reset', return_value=("foobar", [4, 3]))
def test_play_game(reset, next_move, step):
    reward_sum = ml_agent.play_in_background("trained_q_table")
    assert reward_sum == -1
    reset.assert_called_once()
    next_move.assert_has_calls([call("trained_q_table", [4, 3], ANY),
                                call("trained_q_table", [3, 3], ANY),
                                call("trained_q_table", [2, 3], ANY)])
    step.assert_has_calls([call("left", "foobar")] * 3)


