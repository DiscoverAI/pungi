from unittest.mock import patch

import pungi
import pungi.environment.state as state


def test_get_state_from_game_info():
    assert (0, 0) == state.get_head_state_from_game_info({"board": [[1, 0], [0, 0]]})
    assert (0, 1) == state.get_head_state_from_game_info({"board": [[0, 0], [1, 0]]})
    assert (2, 1) == state.get_head_state_from_game_info({"board": [[0, 0, 3], [0, 0, 1]]})
    assert (1, 4) == state.get_head_state_from_game_info({"board": [[0, 0, 3],
                                                                    [0, 0, 0],
                                                                    [0, 0, 0],
                                                                    [0, 0, 0],
                                                                    [0, 1, 0]]})


def test_get_head_and_food_state_from_game_info():
    assert (0, 0, 1, 1) == state.get_head_and_food_state_from_game_info({"board": [[1, 0], [0, 3]]})
    assert (0, 1, 0, 0) == state.get_head_and_food_state_from_game_info({"board": [[3, 0], [1, 0]]})
    assert (2, 1, 2, 0) == state.get_head_and_food_state_from_game_info({"board": [[0, 0, 3], [0, 0, 1]]})
    assert (1, 4, 2, 0) == state.get_head_and_food_state_from_game_info({"board": [[0, 0, 3],
                                                                                   [0, 0, 0],
                                                                                   [0, 0, 0],
                                                                                   [0, 0, 0],
                                                                                   [0, 1, 0]]})
