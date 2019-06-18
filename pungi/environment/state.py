import numpy as np

CODE_SNAKE_HEAD = 1
CODE_SNAKE_TAIL = 2
CODE_FOOD = 3


def get_head_state_from_game_info(game_info):
    board = game_info["board"]
    board = np.array(board)
    itemindex_head = np.where(board == CODE_SNAKE_HEAD)
    return itemindex_head[1][0], itemindex_head[0][0]


def get_head_and_food_state_from_game_info(game_info):
    board = game_info["board"]
    board = np.array(board)
    itemindex_head = np.where(board == CODE_SNAKE_HEAD)
    itemindex_food = np.where(board == CODE_FOOD)
    return itemindex_head[1][0], itemindex_head[0][0], itemindex_food[1][0], itemindex_food[0][0]


def get_state_extractor(state_extractor_name):
    return globals()[state_extractor_name]
