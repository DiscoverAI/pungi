DIRECTION_ENCODING = {0: "left", 1: "right", 2: "up", 3: "down"}


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
