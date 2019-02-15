import requests


def register_game(width, height, snakeLength):
    game_initialization = {
        'width': width,
        'height': height,
        'snakeLength': snakeLength
    }
    headers = {'Accept': 'application/json'}
    r = requests.post('http://localhost:8080/games',
                      json=game_initialization,
                      headers=headers)
    return r.json()['gameId']


def make_move(direction, game_id):
    payload = {'direction': direction}
    r = requests.put('http://localhost:8080/games/'
                     + game_id
                     + '/tokens/snake/direction', json=payload)
    return r.json()['board']
