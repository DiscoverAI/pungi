from unittest.mock import patch

from requests.models import Response

from pungi.environment import backend

mock_register_response = Response()
mock_register_response.json = lambda: {"gameId": "foobar"}

mock_game_info_response = Response()
mock_game_info_response.json = lambda: {"game-info-key": "game-info-value"}


@patch('requests.post', return_value=mock_register_response)
def test_register_game(post_mock):
    actual = backend.register_new_game(board_width=1, board_height=1, snake_length=1)
    expected = 'foobar'

    assert expected == actual
    post_mock.assert_called_with(
        'foo bar/games',
        headers={'Accept': 'application/json'},
        json={'height': 1, 'snakeLength': 1, 'width': 1}
    )


@patch('requests.put', return_value=mock_game_info_response)
def test_make_move(put_mock):
    actual = backend.make_move(direction="up", game_id="foobar")
    expected = {"game-info-key": "game-info-value"}

    assert expected == actual
    put_mock.assert_called_with(
        'foo bar/games/foobar/tokens/snake/direction',
        headers={'Accept': 'application/json'},
        json={"direction": "up"}
    )


@patch('requests.get', return_value=mock_game_info_response)
def test_get_game_info(put_mock):
    actual = backend.get_game_info(game_id="foobar")
    expected = {"game-info-key": "game-info-value"}

    assert expected == actual
    put_mock.assert_called_with(
        'foo bar/games/foobar',
        headers={'Accept': 'application/json'}
    )
