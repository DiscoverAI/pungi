from pungi.client.rest_client import *
import webbrowser
import time

registered_gameid = register_game(20, 20, 3)

webbrowser.open_new('http://localhost:8080/?spectate-game-id=' + registered_gameid)

while True:
    time.sleep(0.1)
    payload = {'direction': 'down'}
    r = requests.put('http://localhost:8080/games/'
                     + registered_gameid
                     + '/tokens/snake/direction', json=payload)
