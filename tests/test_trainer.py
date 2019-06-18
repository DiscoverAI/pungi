from unittest.mock import patch, ANY, call
import pungi.trainer as trainer
import tests.mock_policies

@patch('pungi.agents.qlearning.update_q_value', return_value="updated_q_table")
@patch('pungi.agents.qlearning.next_move', return_value="down")
@patch('pungi.environment.environment.reset', return_value=("foo bar", [0, 0]))
@patch('pungi.environment.environment.step', side_effect=[
    (-1, [0, 1], False, {"score": 10}),
    (-1, [0, 2], False, {"score": 10}),
    (-100, [0, 3], True, {"score": 10})
])
def test_run_episode(step_mock, reset_mock, next_move_mock,
                     update_q_value_mock):
    initial_q_table = "initial_q_table"

    some_policy = lambda q_values: q_values["up"]

    trainer.run_episode(initial_q_table, some_policy)
    reset_mock.assert_called_once()
    step_mock.assert_has_calls([call("down", "foo bar")] * 3)
    update_q_value_mock.assert_has_calls([
        call("initial_q_table", [0, 0], 'down', [0, 1], 0.9, 0.99, -1),
        call('updated_q_table', [0, 1], 'down', [0, 2], 0.9, 0.99, -1),
        call('updated_q_table', [0, 2], 'down', [0, 3], 0.9, 0.99, -100)])
    next_move_mock.assert_has_calls([call("initial_q_table", [0, 0], some_policy),
                                     call("updated_q_table", [0, 1], some_policy),
                                     call("updated_q_table", [0, 2], some_policy)])


@patch('pungi.agents.qlearning.initialize_q_table', return_value="initial_table")
@patch('pungi.trainer.run_episode', side_effect=["updated-table-1", "updated-table-2", "updated-table-3"])
@patch('pungi.agents.policies.globals', return_value={"mock_policy": tests.mock_policies.mock_policy})
def test_train(_globals_mock, run_episode_mock, init_q_table_mock):
    training_result = trainer.train()
    assert training_result == "updated-table-3"
    init_q_table_mock.assert_called_once_with(initial_value=0)
    run_episode_mock.assert_has_calls([call("initial_table", ANY),
                                       call("updated-table-1", ANY),
                                       call("updated-table-2", ANY)])
