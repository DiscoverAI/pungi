import requests


class SnakeClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def register_game(self, board_width, board_height, snake_length):
        return requests.post(self.base_url + '/games',
                             json={'height': board_height,
                                   'width': board_width,
                                   'snakeLength': snake_length},
                             headers={'Accept': 'application/json'}).json()['gameId']

    def make_move(self, direction, game_id):
        return requests.put(self.base_url + '/games/'
                            + game_id
                            + '/tokens/snake/direction',
                            json={'direction': direction},
                            headers={'Accept': 'application/json'}).json()
