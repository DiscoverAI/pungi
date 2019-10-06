from unittest.mock import patch, call, ANY

from pungi.agents.qlearning import greedy_policy_agent as agent
import numpy as np
import gym
import pungi.agents.agent as agent_class


class MockAgent(agent_class.Agent):

    def next_action(self, state, episode_number):
        return "left"

    def update(self, state, action, next_state, reward, game_over):
        pass

    def persist(self, path_to_model_file):
        pass


class MockEnvironment(gym.Env):
    def __init__(self):
        self.steps = 0
        self.current_game_id = 10

    def step(self, action):
        result = [
            (np.array([3, 3]), -1, False, {"score": 10}),
            (np.array([2, 3]), -1, False, {"score": 11}),
            (np.array([1, 3]), -100, True, {"score": 11})
        ][self.steps]
        self.steps += 1
        return result

    def reset(self):
        return "foo bar", np.array([0, 0])


@patch("webbrowser.open_new")
def test_play_game(open_webbrowser):
    env = MockEnvironment()
    agentMock = MockAgent()
    agent.play_in_spectator_mode(agentMock, env)
    open_webbrowser.assert_called_once_with("foo bar/?spectate-game-id=10")


def test_play_game_inbackground():
    env = MockEnvironment()
    agentMock = MockAgent()
    reward_sum = agent.play_in_background(agentMock, env)
    assert reward_sum == -1 - 1 - 100
