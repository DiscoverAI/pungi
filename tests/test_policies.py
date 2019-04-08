import pungi.policies
from unittest.mock import patch


def test_max_policy_down():
    assert "down" == pungi.policies.max_policy({"left": -1, "right": 10, "up": 42, "down": 43.1}, None)


def test_max_policy_up():
    assert "up" == pungi.policies.max_policy({"left": -1, "right": 10, "up": 122, "down": 42}, None)


@patch('numpy.random.choice', lambda x, p: x[0])
def test_max_epsilon_greedy_down():
    assert "up" == pungi.policies.epsilon_greedy_max_policy({"left": -1, "right": 10, "up": 122, "down": 42}, 0)


@patch('numpy.random.choice', lambda x, p: x[0])
def test_max_epsilon_greedy_up():
    assert "up" == pungi.policies.epsilon_greedy_max_policy({"left": -1, "right": 10, "up": 122, "down": 42}, 0)


@patch('numpy.random.choice', lambda x, p: x[1])
@patch('random.choice', return_value="right")
def test_max_epsilon_greedy_down_explore(np_choice_mock):
    assert "right" == pungi.policies.epsilon_greedy_max_policy({"left": -1, "right": 10, "up": 122, "down": 42}, 0)


@patch('numpy.random.choice', lambda x, p: x[1])
@patch('random.choice', return_value="down")
def test_max_epsilon_greedy_up_explore(np_choice_mock):
    assert "down" == pungi.policies.epsilon_greedy_max_policy({"left": -1, "right": 10, "up": 122, "down": 42}, 0)
