class Agent:
    def next_action(self, state):
        raise NotImplementedError

    def update(self, state, action, next_state, reward):
        raise NotImplementedError
