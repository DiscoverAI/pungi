from unittest.mock import MagicMock
from unittest.mock import patch
import pungi.trainer as trainer
import pungi.environment.environment as environment

@patch('pungi.qlearning.update_q_value', return_value="updated_q_table")
@patch('pungi.qlearning.initialize_q_table', return_value="mock_q_table")
def test_run_episode_immediate_loss(initialize_q_table_mock, update_q_value_mock):
    pass
    # env.reset = MagicMock(return_value=[0, 0])
    # env.step = MagicMock(return_value=(-1, [0, 1], False, {"score": 10}))
    #
    # new_q_table = trainer.run_episode(q_table="mock_q_table", environment=env)
    # assert new_q_table == "updated_q_table"
    # env.reset.assert_called_once()
   # update_q_value_mock.asssert_called_once_with(q_table="mock_q_table",
    #                                             state=)
