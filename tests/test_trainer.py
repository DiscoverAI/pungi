from unittest.mock import MagicMock
from unittest.mock import patch
import pungi.trainer as trainer
import pungi.environment.environment as environment
from unittest.mock import call
import pungi.qlearning as qlearning


@patch('pungi.qlearning.update_q_value', return_value="updated_q_table")
@patch('pungi.qlearning.next_move', return_value="down")
@patch('pungi.environment.environment.reset', return_value=("foo bar", [0, 0]))
@patch('pungi.environment.environment.step', side_effect=[
    (-1, [0, 1], False, {"score": 10}),
    (-1, [0, 2], False, {"score": 10}),
    (-100, [0, 3], True, {"score": 10})
])
def test_run_episode_immediate_loss(step_mock, reset_mock, next_move_mock,
                                    update_q_value_mock):
    initial_q_table = "initial_q_table"

    trainer.run_episode(initial_q_table)
    reset_mock.assert_called_once()
    step_mock.assert_has_calls([call("down", "foo bar")] * 3)
    update_q_value_mock.assert_has_calls([
        call("initial_q_table", [0, 0], 'down', [0, 1], 0.9, 0.99, -1),
        call('updated_q_table', [0, 1], 'down', [0, 2], 0.9, 0.99, -1),
        call('updated_q_table', [0, 2], 'down', [0, 3], 0.9, 0.99, -100)])
    next_move_mock.assert_has_calls([call("initial_q_table", [0, 0], qlearning.max_policy),
                                     call("updated_q_table", [0, 1], qlearning.max_policy),
                                     call("updated_q_table", [0, 2], qlearning.max_policy)])


@patch('pungi.qlearning.initialize_q_table', return_value="initial_table")
@patch('pungi.trainer.run_episode', side_effect=["updated-table-1", "updated-table-2", "updated-table-3"])
def test_train(run_episode_mock, init_q_table_mock):
    training_result = trainer.train()
    assert training_result == "updated-table-3"
    init_q_table_mock.assert_called_once_with(initial_value=0)
    run_episode_mock.assert_has_calls([call("initial_table"),
                                       call("updated-table-1"),
                                       call("updated-table-2")])
