import pungi.agents.policies
import pungi.agents.qlearning.qlearning
from unittest.mock import patch

import tests.mock_policies


def test_max_policy_down():
    assert "down" == pungi.agents.policies.max_policy({"left": -1, "right": 10, "up": 42, "down": 43.1}, None)


def test_max_policy_up():
    assert "up" == pungi.agents.policies.max_policy({"left": -1, "right": 10, "up": 122, "down": 42}, None)


@patch('numpy.random.choice', lambda x, p: x[0])
def test_max_epsilon_greedy_down():
    assert "up" == pungi.agents.policies.epsilon_greedy_max_policy({"left": -1, "right": 10, "up": 122, "down": 42}, 0)


@patch('numpy.random.choice', lambda x, p: x[0])
def test_max_epsilon_greedy_up():
    assert "up" == pungi.agents.policies.epsilon_greedy_max_policy({"left": -1, "right": 10, "up": 122, "down": 42}, 0)


@patch('numpy.random.choice', lambda x, p: x[1])
@patch('random.choice', return_value="right")
def test_max_epsilon_greedy_down_explore(random_choice_mock):
    assert "right" == pungi.agents.policies.epsilon_greedy_max_policy({"left": -1, "right": 10, "up": 122, "down": 42}, 0)
    random_choice_mock.assert_called_once_with(pungi.agents.qlearning.qlearning.DIRECTIONS)


@patch('numpy.random.choice', lambda x, p: x[1])
@patch('random.choice', return_value="down")
def test_max_epsilon_greedy_up_explore(random_choice_mock):
    assert "down" == pungi.agents.policies.epsilon_greedy_max_policy({"left": -1, "right": 10, "up": 122, "down": 42}, 0)
    random_choice_mock.assert_called_once_with(pungi.agents.qlearning.qlearning.DIRECTIONS)


@patch('pungi.agents.policies.globals', return_value={"mock_policy": tests.mock_policies.mock_policy})
def test_load_policy(globals_mock):
    assert tests.mock_policies.mock_policy == pungi.agents.policies.get_policy("mock_policy")
