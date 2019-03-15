import pungi.qlearning as qlearning


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
