import requests

def register_game(param, param1, param2, param3):
    requests.post(param + "/games", data={'height': 1, 'width': 1, 'snakeLength': 1})
    return "foo"