from collections import defaultdict

DIRECTION_VECTORS = {"left": [-1, 0], "right": [1, 0], "up": [0, -1], "down": [0, 1]}
DIRECTIONS = DIRECTION_VECTORS.keys()


def next_move(q_table, current_state, policy):
    return policy({action: q_table[(*current_state, action)]
                   for action in DIRECTIONS})


def max_policy(q_values):
    return max(q_values, key=q_values.get)


def get_reward(game_state):
    if game_state["encoded-board"]["game-over"]:
        return -100
    elif game_state["encoded-board"]["ate-food"]:
        return 100
    else:
        return -1


def initialize_q_table(initial_value):
    return defaultdict(lambda: initial_value)


def q_value(old_value, next_state_q_value, learning_rate, discount_factor, reward):
    return (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_state_q_value)


def get_next_state(state, action, board_width, board_height):
    direction_change = DIRECTION_VECTORS[action]
    return (state[0] + direction_change[0]) % board_width, \
           (state[1] + direction_change[1]) % board_height


def get_state_from_game_info(game_info):
    pass


def update_q_value(q_table, state, action, next_state, learning_rate, discount_factor, reward):
    next_q_values = [q_table[next_state, direction]
                     for direction in DIRECTION_VECTORS.keys()]
    maximal_next_q_value = max(next_q_values)
    q_table[state, action] = \
        q_value(old_value=q_table[state, action],
                next_state_q_value=maximal_next_q_value,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                reward=reward)
    return q_table
