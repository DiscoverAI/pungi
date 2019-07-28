from unittest.mock import patch, call, ANY

from pungi.agents.qlearning import greedy_policy_agent as agent
import numpy as np
import gym


class MockEnvironment(gym.Env):
    def __init__(self):
        self.steps = 0
        self.current_game_id = 10

    def step(self, action):
        result = [
            (-1, np.array([3, 3]), False, {"score": 10}),
            (100, np.array([2, 3]), False, {"score": 11}),
            (-100, np.array([1, 3]), True, {"score": 11})
        ][self.steps]
        self.steps += 1
        return result

    def reset(self):
        return "foo bar", np.array([0, 0])


@patch("webbrowser.open_new")
@patch('pungi.agents.qlearning.qlearning.next_move', return_value="left")
def test_play_game(next_move, step, open_webbrowser):
    env = MockEnvironment()
    agent.play_in_spectator_mode("trained_q_table", env)
    next_move.assert_has_calls([call("trained_q_table", ANY, ANY),
                                call("trained_q_table", ANY, ANY),
                                call("trained_q_table", ANY, ANY)])
    step.assert_has_calls([call("left", "foobar")] * 3)
    open_webbrowser.assert_called_once_with("foo bar/?spectate-game-id=foobar")


@patch('pungi.agents.qlearning.qlearning.next_move', return_value="left")
def test_play_game(next_move):
    env = MockEnvironment()
    reward_sum = agent.play_in_background("trained_q_table", env)
    assert reward_sum == -1
    next_move.assert_has_calls([call("trained_q_table", ANY, ANY),
                                call("trained_q_table", ANY, ANY),
                                call("trained_q_table", ANY, ANY)])
