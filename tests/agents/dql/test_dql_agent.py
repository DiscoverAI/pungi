from unittest.mock import patch, ANY

import pytest

import pungi.agents.dql.dql_agent as dql_agent
import numpy as np

mock_config = {
    'replay_memory_limit': 10,
    'model_file_name': 'tests/resources/simple-dqn-architecture.json',
    'optimizer': 'adam',
    'batch_size': 2,
    'gamma': 0.95
}


class MockQNetwork:

    def __init__(self):
        self.mock_q_value_output = {"left": 0.0,
                                    "right": 2.0,
                                    "up": 0.0,
                                    "down": -1.0}

    def predict(self, _state):
        return [list(self.mock_q_value_output.values())]

    def fit(self, *args, **kwargs):
        pass


def test_agent_experience_replay_single_step():
    agent = dql_agent.DQLAgent(mock_config, MockQNetwork())
    state = np.array([0, 1])
    action = "left"
    game_over = False
    reward = -1
    next_state = np.array([1, 0])
    memory_entry = state, action, next_state, reward, game_over
    assert agent.update(*memory_entry) is True
    assert agent.sample_memory(1) == [memory_entry]


memory_entry1 = np.array([0, 1]), "left", np.array([1, 0]), -1, False
memory_entry2 = np.array([1, 0]), "right", np.array([0, 1]), 50, False


@patch("random.sample", return_value=[memory_entry1, memory_entry2])
def test_agent_experience_replay_two_steps(sample_mock):
    agent = dql_agent.DQLAgent(mock_config, MockQNetwork())
    agent.update(*memory_entry1)
    agent.update(*memory_entry2)
    assert agent.sample_memory(2) == [memory_entry1, memory_entry2]
    sample_mock.assert_called_with(ANY, 2)


@patch("random.sample", return_value=[memory_entry1, memory_entry2])
def test_agent_experience_replay_four_steps(sample_mock):
    agent = dql_agent.DQLAgent(mock_config, MockQNetwork())
    agent.update(*memory_entry1)
    agent.update(*memory_entry2)
    assert agent.sample_memory(64) == [memory_entry1, memory_entry2]
    sample_mock.assert_called_with(ANY, 2)


def test_should_create_agent_with_limitied_replay_memory(mocker):
    init_replay_memory_mock = mocker.patch('pungi.agents.dql.dql_agent.DQLAgent.init_replay_memory')
    agent = dql_agent.DQLAgent({'replay_memory_limit': 42, 'gamma': 0.9, 'batch_size': 42}, MockQNetwork())
    assert agent is not None
    init_replay_memory_mock.assert_called_with(42)


def test_next_action(mocker):
    epsilon_greedy_mock = mocker.patch('pungi.agents.policies.epsilon_greedy_max_policy')
    mock_prediction = [[1, 2, 3, 4]]

    class SimpleMockQNetwork:
        def predict(self, state):
            assert (state == np.array([[0, 1], [0, 1]]).reshape((1, 2, 2, 1))).all()
            self.called_predict = True
            return mock_prediction

    mock_network = SimpleMockQNetwork()
    agent = dql_agent.DQLAgent(mock_config, mock_network, epsilon_greedy_mock)
    agent.next_action(np.array([[0, 1], [0, 1]]), 42)
    assert mock_network.called_predict
    epsilon_greedy_mock.assert_called_with({'left': 1, 'right': 2, 'up': 3, 'down': 4}, 42)


def test_get_q_update():
    mock_network = MockQNetwork()
    agent = dql_agent.DQLAgent(mock_config, mock_network)
    assert 40 == agent.get_q_update(reward=40, game_over=True, next_state=np.array([[0, 1], [2, 2]]))
    assert 40 + 0.95 * 2.0 == pytest.approx(agent.get_q_update(reward=40,
                                                               game_over=False,
                                                               next_state=np.array([[0, 1], [2, 2]])),
                                            rel=1e-12)


def test_build_examples():
    mock_state = np.array([[0, 1], [2, 2]])
    mock_next_state = np.array([[0, 1], [2, 3]])
    mock_reward = 42
    mock_action = 1  # -> right, second item in prediction array
    mock_action_str = "right"
    game_over = False
    q_update = 1.234

    class MockQNetworkGradientStep:
        def predict(self, state):
            assert (state == mock_state.reshape((1, 2, 2, 1))).all()
            # return 2d array, because we are predicting a "batch" of 1 data item
            return [[1.0,
                     2.0,  # we picked this action
                     -1.0,
                     -1.5]]

        def fit(self, state, q_values, verbose):
            assert (state == mock_state).all()
            assert q_values[0][mock_action] == q_update
            assert q_values[0][0] == 1.0
            assert q_values[0][2] == -1.0
            assert q_values[0][3] == -1.5

    mock_network = MockQNetworkGradientStep()
    agent = dql_agent.DQLAgent(mock_config, mock_network)
    agent.get_q_update = lambda *args: q_update

    input, expected_output = agent.build_training_examples(
        [(mock_state, mock_action_str, mock_reward, mock_next_state, game_over)])
    assert input == [mock_state]
    assert expected_output == [[1.0,
                                q_update,  # we picked this action
                                -1.0,
                                -1.5]]


def test_memory_replay(mocker):
    mock_state = np.array([[0, 1], [1, 3]])
    action = 1
    action_str = "right"
    next_state = np.array([[1, 1], [0, 3]])
    reward = 10
    game_over = False
    q_update = 42

    class MockQNetworkGradientStep:
        def predict(self, state):
            assert (state == mock_state.reshape((1, 2, 2, 1))).all()
            return [[1.0, 2.0, -1.0, -1.5]]

        def fit(self, states, q_values, verbose=0):
            assert (states[0] == mock_state.reshape((1, 2, 2, 1))).all()
            assert q_values[0][action] == q_update
            assert q_values[0][0] == 1.0
            assert q_values[0][2] == -1.0
            assert q_values[0][3] == -1.5

    mock_network = MockQNetworkGradientStep()
    agent = dql_agent.DQLAgent(mock_config, mock_network)

    def sample_memory_mock(batch_size):
        assert batch_size == 2
        return [(mock_state, action_str, next_state, reward, game_over)]

    agent.sample_memory = sample_memory_mock
    agent.get_q_update = lambda *args: q_update

    agent.memory_replay()
