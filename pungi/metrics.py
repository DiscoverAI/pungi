import pungi.ml_agent

def get_average_cumulative_reward(episodes, q_table):
    sum_reward = 0
    for episode in range (episodes):
        reward = pungi.ml_agent.play_in_background(q_table)
        sum_reward = sum_reward + reward
    return sum_reward / episodes

