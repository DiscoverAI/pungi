import os
import re
from unittest.mock import patch, call

import pungi.metrics as metrics
import gym
import numpy as np


class MockEnvironment(gym.Env):
    def __init__(self):
        self.steps = 0

    def step(self, action):
        result = [
            (np.array([0, 1]), -1, False, {"score": 10}),
            (np.array([0, 2]), -1, False, {"score": 10}),
            (np.array([0, 3]), 100, True, {"score": 10})
        ][self.steps]
        self.steps += 1
        return result

    def reset(self):
        return np.array([0, 0])


@patch("pungi.agents.qlearning.greedy_policy_agent.play_in_background", side_effect=[
    10,
    22,
    -5
])
def test_get_average_cumulative_reward(play_in_background):
    mock = MockEnvironment()
    average_cumulative_reward = metrics.get_average_cumulative_reward(env=mock, episodes=3, q_table="q_table_mock")
    assert average_cumulative_reward == 9
    play_in_background.assert_has_calls([call("q_table_mock", mock),
                                         call("q_table_mock", mock),
                                         call("q_table_mock", mock)])


@patch("pungi.metrics.get_average_cumulative_reward", return_value=42)
def test_calculate_and_write_metrics(get_average_cumulative_reward):
    mock = MockEnvironment()
    test_json_file_path = "tests/resources/metrics.json"
    metrics.calculate_and_write_metrics(env=mock, episodes=3, q_table="q_table_mock", output_path=test_json_file_path)
    with open(test_json_file_path, "r") as metrics_file:
        assert re.sub(r"\s+", '', metrics_file.read()) == re.sub(r"\s+", '', """
        {
            "average_cumulative_reward": 42
        }""")
    os.remove(test_json_file_path)
