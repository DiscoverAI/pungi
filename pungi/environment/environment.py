import pungi.qlearning as qlearning


class Environment:
    def __init__(self, client):
        self.client = client

    def reset(self):
        game_id = self.client.register_new_game()
        game_info = self.client.get_game_info(game_id)
        return qlearning.get_state_from_game_info(game_info)

    def step(self, action):
        next_state = self.client.make_move(action)
        reward = qlearning.get_reward(next_state)
        next_position = qlearning.get_state_from_game_info(next_state)
        game_over = next_state["game-over"]
        info = {"score": next_state["score"]}
        return reward, next_position, game_over, info
