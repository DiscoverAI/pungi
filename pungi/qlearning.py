from pungi import client


def next_move(q_table, current_state, policy):
    return policy(q_table[current_state[0]][current_state[1]])
