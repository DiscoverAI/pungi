from unittest.mock import patch

from pungi.snake_client import SnakeClient
from requests.models import Response

mock_register_response = Response()
mock_register_response.json = lambda: {"gameId": "foobar"}

mock_move_response = Response()
mock_move_response.json = lambda: {"game-info-key": "game-info-value"}


@patch('requests.post', return_value=mock_register_response)
def test_register_game(post_mock):
    client = SnakeClient("http://example.com")
    game_id = client.register_game(board_width=1, board_height=1, snake_length=1)
    assert 'foobar' == game_id
    post_mock.assert_called_with('http://example.com/games',
                                 headers={'Accept': 'application/json'},
                                 json={'height': 1, 'snakeLength': 1, 'width': 1})


@patch('requests.put', return_value=mock_move_response)
def test_register_game(put_mock):
    client = SnakeClient("http://example.com")
    game_info = client.make_move(direction="up", game_id="foobar")
    assert {"game-info-key": "game-info-value"} == game_info
    put_mock.assert_called_with('http://example.com/games/foobar/tokens/snake/direction',
                                headers={'Accept': 'application/json'},
                                json={"direction": "up"})
