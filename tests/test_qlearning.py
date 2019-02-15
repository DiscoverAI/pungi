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
