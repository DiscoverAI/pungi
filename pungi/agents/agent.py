class Agent:
    def next_action(self, state, episode_number):
        raise NotImplementedError

    def update(self, state, action, next_state, reward, game_over):
        raise NotImplementedError

    def persist(self, path_to_model_file):
        raise NotImplementedError
