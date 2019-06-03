from unittest.mock import patch, call, ANY

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
