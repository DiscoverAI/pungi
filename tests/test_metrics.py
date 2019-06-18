import os
import re
from unittest.mock import patch, call

import pungi.metrics as metrics


@patch("pungi.ml_agent.play_in_background", side_effect=[
    10,
    22,
    -5
])
def test_get_average_cumulative_reward(play_in_background):
    average_cumulative_reward = metrics.get_average_cumulative_reward(episodes=3, q_table="q_table_mock")
    assert average_cumulative_reward == 9
    play_in_background.assert_has_calls([call("q_table_mock"),
                                         call("q_table_mock"),
                                         call("q_table_mock")])


@patch("pungi.metrics.get_average_cumulative_reward", return_value=42)
def test_calculate_and_write_metrics(get_average_cumulative_reward):
    test_json_file_path = "tests/resources/metrics.json"
    metrics.calculate_and_write_metrics(episodes=3, q_table="q_table_mock", output_path=test_json_file_path)
    with open(test_json_file_path, "r") as metrics_file:
        assert re.sub(r"\s+", '', metrics_file.read()) == re.sub(r"\s+", '', """
        {
            "average_cumulative_reward": 42
        }""")
    os.remove(test_json_file_path)
