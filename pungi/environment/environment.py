import pungi.qlearning as qlearning


class Environment:
    def __init__(self, state, client):
        self.state = state
        self.client = client

    def is_game_over(self):
        return self.state["encoded-board"]["game-over"]

    def step(self, action):
        next_state = self.client.make_move(action)
        reward = qlearning.get_reward(next_state)
        next_position = qlearning.get_state_from_game_info(next_state)
        game_over = self.client.is_game_over()
        info = {"score": next_state ["encoded-board"]["score"]}
        return reward, next_position, game_over, info
