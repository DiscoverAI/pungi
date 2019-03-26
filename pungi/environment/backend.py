import requests

from pungi import config


def register_new_game(board_width, board_height, snake_length):
    games_uri = config.CONF.get_value('backend') + '/games'
    response = requests.post(
        games_uri,
        json={'height': board_height, 'width': board_width, 'snakeLength': snake_length},
        headers={'Accept': 'application/json'}
    )
    return response.json()['gameId']


def make_move(direction, game_id):
    direction_uri = config.CONF.get_value('backend') + '/games/' + game_id + '/tokens/snake/direction'
    response = requests.put(
        direction_uri,
        json={'direction': direction},
        headers={'Accept': 'application/json'}
    )
    return response.json()


def get_game_info(game_id):
    game_uri = config.CONF.get_value('backend') + '/games/' + game_id
    response = requests.get(game_uri, headers={'Accept': 'application/json'})
    return response.json()
