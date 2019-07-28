import json

from pungi.agents.qlearning import greedy_policy_agent as agent


def get_average_cumulative_reward(episodes, q_table):
    sum_reward = 0
    for episode in range(episodes):
        reward = agent.play_in_background(q_table)
        sum_reward = sum_reward + reward
    return sum_reward / episodes


def calculate_and_write_metrics(episodes, q_table, output_path):
    average_reward = get_average_cumulative_reward(episodes, q_table)
    metric_dict = {"average_cumulative_reward": average_reward}
    with open(output_path, "w") as metrics_file:
        metrics_file.write(json.dumps(metric_dict))
