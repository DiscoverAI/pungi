from pungi import snake_client


def test_register_game(mocker):
    requests_mock = mocker.patch('requests.post')
    requests_mock.returnValue = 'foo'
    game_id = snake_client.register_game('foobar', 1, 1, 1)

    assert 'foo' == game_id
    requests_mock.assert_called_with('foobar/games', data={'height': 1, 'snakeLength': 1, 'width': 1})
