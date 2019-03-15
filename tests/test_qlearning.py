import pungi.qlearning as qlearning
import numpy as np


# qtable [(left,right,up,down), ...]

def test_get_left_as_next_move(mocker):
    q_table = [[(-1, 2, 3, 4), (-1, 2, 3, 4)],
               [(-1, 2, 3, 4), (-1, 2, 3, 4)]]
    current_state = [0, 0]
    policy = mocker.Mock()
    policy.return_value = "left"
    assert "left" == qlearning.next_move(q_table, current_state, policy)
    policy.assert_called_once_with((-1, 2, 3, 4))


def test_next_move_based_on_maximum(mocker):
    q_table = [[(-1, 2, 3, 4), (-1, 2, 15, 4)],
               [(10, 2, 3, 4), (-1, 7, 3, 4)]]
    policy = mocker.Mock()
    policy.return_value = "up"
    current_state = [0, 1]
    assert "up" == qlearning.next_move(q_table, current_state, policy)
    policy.assert_called_once_with((-1, 2, 15, 4))


def test_max_policy_down():
    assert "down" == qlearning.max_policy((-1, 2, 3, 4))


def test_max_policy_up():
    assert "up" == qlearning.max_policy((-1, 2, 15, 4))


def test_reward():
    assert 0 == qlearning.get_reward({"game-over": False, "score": 1, "board": [], "ate-food": False})
    assert 100 == qlearning.get_reward({"game-over": False, "score": 1, "board": [], "ate-food": True})
    assert -100 == qlearning.get_reward({"game-over": True, "score": 1, "board": [], "ate-food": False})


def test_initialize_q_table():
    assert [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]] == qlearning.initialize_q_table(board_width=2,
                                                                                          board_height=2).tolist()


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


# TODO: Add more test cases, especially for the special case of hitting the border
def test_update_q_value(mocker):
    with mocker.patch('pungi.qlearning.q_value', return_value=42.0):
        test_q_table = np.array([
            [
                [0.0, 2.0, 1.0, 3.0], [0.0, 2.0, 1.0, 3.0]
            ],
            [
                [0.0, 2.0, 1.0, 3.0], [0.0, 2.0, 1.0, 3.0]
            ]
        ])
        assert [
                   [
                       [0.0, 2.0, 1.0, 3.0], [0.0, 2.0, 1.0, 3.0]
                   ],
                   [
                       [0.0, 2.0, 42.0, 3.0], [0.0, 2.0, 1.0, 3.0]
                   ]
               ] == qlearning.update_q_value(q_table=test_q_table,
                                             state=(1, 0),
                                             action="up",
                                             learning_rate=0.1,
                                             discount_factor=0.9,
                                             reward=-1).tolist()
