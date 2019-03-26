from collections import defaultdict
from unittest.mock import patch

import pungi.qlearning as qlearning


# qtable [(left,right,up,down), ...]

def test_get_left_as_next_move(mocker):
    q_table = {(0, 0, "left"): -1,
               (0, 0, "right"): 2,
               (0, 0, "up"): 3,
               (0, 0, "down"): 4}
    current_state = [0, 0]
    policy = mocker.Mock()
    policy.return_value = "left"
    assert "left" == qlearning.next_move(q_table, current_state, policy)
    policy.assert_called_once_with({"left": -1, "right": 2, "up": 3, "down": 4})


def test_next_move_based_on_maximum(mocker):
    q_table = {(0, 1, "left"): -1,
               (0, 1, "right"): 10,
               (0, 1, "up"): 42,
               (0, 1, "down"): -2}
    policy = mocker.Mock()
    policy.return_value = "up"
    current_state = [0, 1]
    assert "up" == qlearning.next_move(q_table, current_state, policy)
    policy.assert_called_once_with({"left": -1, "right": 10, "up": 42, "down": -2})


def test_max_policy_down():
    assert "down" == qlearning.max_policy({"left": -1, "right": 10, "up": 42, "down": 43.1})


def test_max_policy_up():
    assert "up" == qlearning.max_policy({"left": -1, "right": 10, "up": 122, "down": 42})


def test_reward():
    assert -1 == qlearning.get_reward(
        {"game-over": False, "score": 1, "board": [], "ate-food": False})
    assert 100 == qlearning.get_reward(
        {"game-over": False, "score": 1, "board": [], "ate-food": True})
    assert -100 == qlearning.get_reward(
        {"game-over": True, "score": 1, "board": [], "ate-food": False})


def test_initialize_q_table():
    sparse_data_structure = qlearning.initialize_q_table(initial_value=42)
    assert sparse_data_structure[(0, 0), "left"] == 42
    sparse_data_structure[(0, 0), "left"] = 0
    assert sparse_data_structure[(0, 0), "left"] == 0


def test_should_calculate_zero_for_zero_learning_rate():
    assert 0.0 == qlearning.q_value(old_value=0,
                                    next_state_q_value=0,
                                    learning_rate=0,
                                    discount_factor=42,
                                    reward=100)


def test_should_calculate_q_value_for_n_params():
    assert 1.07 == qlearning.q_value(old_value=1,
                                     next_state_q_value=3,
                                     learning_rate=0.1,
                                     discount_factor=0.9,
                                     reward=-1)


@patch('pungi.qlearning.q_value', return_value=42)
def test_update_q_value(q_value_patch):
    test_q_table = defaultdict(lambda: 0)
    test_q_table[(0, 1), "up"] = 1.0
    test_q_table[(0, 0), "up"] = 2.0
    test_q_table[(0, 0), "left"] = 0.0
    test_q_table[(0, 0), "right"] = 1.0
    test_q_table[(0, 0), "down"] = 3.0
    updated_q_table = qlearning.update_q_value(q_table=test_q_table,
                                               state=(0, 1),
                                               action="up",
                                               next_state=(0, 0),
                                               learning_rate=0.1,
                                               discount_factor=0.9,
                                               reward=-1)
    assert updated_q_table[(0, 1), "up"] == 42.0
    q_value_patch.assert_called_with(old_value=1.0,
                                     next_state_q_value=3.0,
                                     learning_rate=0.1,
                                     discount_factor=0.9,
                                     reward=-1)


@patch('pungi.qlearning.q_value', return_value=42)
def test_update_q_value_on_edge(q_value_patch):
    test_q_table = defaultdict(lambda: 0)
    test_q_table[(0, 0), "up"] = 1.0
    test_q_table[(0, 1), "up"] = 2.1
    test_q_table[(0, 1), "left"] = 0.1
    test_q_table[(0, 1), "right"] = -2.2
    test_q_table[(0, 1), "down"] = 4.0
    updated_q_table = qlearning.update_q_value(q_table=test_q_table,
                                               state=(0, 0),
                                               action="up",
                                               next_state=(0, 1),
                                               learning_rate=0.1,
                                               discount_factor=0.9,
                                               reward=-1)
    assert updated_q_table[(0, 0), "up"] == 42.0
    q_value_patch.assert_called_with(old_value=1.0,
                                     next_state_q_value=4.0,
                                     learning_rate=0.1,
                                     discount_factor=0.9,
                                     reward=-1)


def test_get_next_state():
    assert (0, 0) == qlearning.get_next_state((0, 1), "up", 2, 2)
    assert (0, 1) == qlearning.get_next_state((0, 0), "up", 2, 2)
    assert (0, 0) == qlearning.get_next_state((1, 0), "right", 2, 2)
    assert (19, 0) == qlearning.get_next_state((0, 0), "left", 20, 20)
    assert (0, 0) == qlearning.get_next_state((0, 19), "down", 20, 20)


def test_get_state_from_game_info():
    assert [0, 0] == qlearning.get_state_from_game_info({"board": [[1, 0], [0, 0]]})
    assert [0, 1] == qlearning.get_state_from_game_info({"board": [[0, 0], [1, 0]]})
    assert [2, 1] == qlearning.get_state_from_game_info({"board": [[0, 0, 3], [0, 0, 1]]})
    assert [1, 4] == qlearning.get_state_from_game_info({"board": [[0, 0, 3],
                                                                   [0, 0, 0],
                                                                   [0, 0, 0],
                                                                   [0, 0, 0],
                                                                   [0, 1, 0]]})
