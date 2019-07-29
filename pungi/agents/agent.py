class Agent:
    def next_action(self, state, episode_number):
        raise NotImplementedError

    def update(self, state, action, next_state, reward):
        raise NotImplementedError
