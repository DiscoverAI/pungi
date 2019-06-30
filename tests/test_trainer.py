from unittest.mock import patch, ANY, call
import pungi.trainer as trainer
import tests.mock_policies
import gym
import numpy as np


class MockEnvironment(gym.Env):
    def __init__(self):
        self.steps = 0

    def step(self, action):
        result = [
            (-1, np.array([0, 1]), False, {"score": 10}),
            (-1, np.array([0, 2]), False, {"score": 10}),
            (-100, np.array([0, 3]), True, {"score": 10})
        ][self.steps]
        self.steps += 1
        return result

    def reset(self):
        return "foo bar", np.array([0, 0])


@patch('pungi.agents.qlearning.qlearning.update_q_value', return_value="updated_q_table")
@patch('pungi.agents.qlearning.qlearning.next_move', return_value="down")
def test_run_episode(next_move_mock, update_q_value_mock):
    initial_q_table = "initial_q_table"

    some_policy = lambda q_values: q_values["up"]

    env = MockEnvironment()

    trainer.run_episode(initial_q_table, some_policy, env)
    update_q_value_mock.assert_has_calls([
        call("initial_q_table", ANY, 'down', ANY, 0.9, 0.99, -1),
        call('updated_q_table', ANY, 'down', ANY, 0.9, 0.99, -1),
        call('updated_q_table', ANY, 'down', ANY, 0.9, 0.99, -100)])
    next_move_mock.assert_has_calls([call("initial_q_table", ANY, some_policy),
                                     call("updated_q_table", ANY, some_policy),
                                     call("updated_q_table", ANY, some_policy)])


@patch('pungi.agents.qlearning.qlearning.initialize_q_table', return_value="initial_table")
@patch('pungi.trainer.run_episode', side_effect=["updated-table-1", "updated-table-2", "updated-table-3"])
@patch('pungi.agents.qlearning.policies.globals', return_value={"mock_policy": tests.mock_policies.mock_policy})
def test_train(_globals_mock, run_episode_mock, init_q_table_mock):
    env = MockEnvironment()
    training_result = trainer.train(env)
    assert training_result == "updated-table-3"
    init_q_table_mock.assert_called_once_with(initial_value=0)
    run_episode_mock.assert_has_calls([call("initial_table", ANY, env),
                                       call("updated-table-1", ANY, env),
                                       call("updated-table-2", ANY, env)])
