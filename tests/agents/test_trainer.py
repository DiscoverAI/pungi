from unittest.mock import patch, call

import gym
import numpy as np
import pungi.agents.agent as agent
import pungi.agents.trainer as trainer


class MockAgent(agent.Agent):
    def __init__(self):
        self.next_action_calls = []
        self.update_calls = []

    def next_action(self, state, episode_number):
        self.next_action_calls.append((tuple(state), episode_number))
        return "foo"

    def update(self, state, action, next_state, reward, game_over):
        self.update_calls.append((tuple(state), action, tuple(next_state), reward, game_over))

    def on_after_episode(self):
        pass


class MockEnvironment(gym.Env):
    def __init__(self):
        self.steps = 0

    def step(self, action):
        result = [
            (np.array([0, 1]), -1, False, {"score": 10}),
            (np.array([0, 2]), -1, False, {"score": 10}),
            (np.array([0, 3]), 100, True, {"score": 10})
        ][self.steps]
        self.steps += 1
        return result

    def reset(self):
        return np.array([0, 0])


def test_run_episode():
    env = MockEnvironment()
    agent = MockAgent()
    trainer.run_episode(agent, env, episode_number=42)
    assert agent.next_action_calls == \
           [((0, 0), 42), ((0, 1), 42), ((0, 2), 42)]
    assert agent.update_calls == \
           [((0, 0), 'foo', (0, 1), -1, False),
            ((0, 1), 'foo', (0, 2), -1, False),
            ((0, 2), 'foo', (0, 3), 100, True)]


@patch('pungi.agents.trainer.run_episode',
       side_effect=["updated-table-1", "updated-table-2", "updated-table-3"])
def test_train(run_episode_mock):
    env = MockEnvironment()
    agent = MockAgent()
    trainer.train(agent, env)
    run_episode_mock.assert_has_calls([call(agent, env, 0),
                                       call(agent, env, 1),
                                       call(agent, env, 2)])
