import numpy as np

DIRECTION_ENCODING = {0: "left", 1: "right", 2: "up", 3: "down"}

DIRECTION_INDICES = {v: k for k, v in DIRECTION_ENCODING.items()}

DIRECTION_VECTORS = {"left": [0, -1], "right": [0, 1], "up": [-1, 0], "down": [1, 0]}


def next_move(q_table, current_state, policy):
    return policy(q_table[current_state[0]][current_state[1]])


def max_policy(q_values):
    return DIRECTION_ENCODING[q_values.index(max(q_values))]


def get_reward(game_state):
    if game_state["game-over"]:
        return -100
    elif game_state["ate-food"]:
        return 100
    else:
        return 0


def initialize_q_table(board_width, board_height):
    return np.zeros(shape=(board_width, board_height, 4))


def q_value(old_value, next_state_q_value, learning_rate, discount_factor, reward):
    return (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_state_q_value)


def update_q_value(q_table, state, action, learning_rate, discount_factor, reward):
    direction_change = DIRECTION_VECTORS[action]
    next_state = [state[0] + direction_change[0], state[1] + direction_change[1]]
    next_q_values = q_table[next_state[0]][next_state[1]]
    maximal_next_q_value = np.max(next_q_values)
    index_of_action = DIRECTION_INDICES[action]
    q_table[state[0]][state[1]][index_of_action] = q_value(q_table[state[0]][state[1]],
                                                           maximal_next_q_value,
                                                           learning_rate,
                                                           discount_factor, reward)
    return q_table
