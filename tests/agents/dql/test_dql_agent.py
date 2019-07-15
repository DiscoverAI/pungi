from unittest.mock import patch, ANY
import pungi.agents.dql.dql_agent as dql_agent
import numpy as np


def test_agent_experience_replay_single_step():
    agent = dql_agent.DQLAgent()
    state = np.array([0, 1])
    action = "left"
    reward = -1
    next_state = np.array([1, 0])
    memory_entry = state, action, reward, next_state
    assert agent.remember(*memory_entry) is True
    assert agent.sample_memory(1) == [memory_entry]


memory_entry1 = np.array([0, 1]), "left", -1, np.array([1, 0])
memory_entry2 = np.array([1, 0]), "right", 50, np.array([0, 1])


@patch("random.sample", return_value=[memory_entry1, memory_entry2])
def test_agent_experience_replay_two_steps(sample_mock):
    agent = dql_agent.DQLAgent()
    agent.remember(*memory_entry1)
    agent.remember(*memory_entry2)
    assert agent.sample_memory(2) == [memory_entry1, memory_entry2]
    sample_mock.assert_called_with(ANY, 2)


@patch("random.sample", return_value=[memory_entry1, memory_entry2])
def test_agent_experience_replay_four_steps(sample_mock):
    agent = dql_agent.DQLAgent()
    agent.remember(*memory_entry1)
    agent.remember(*memory_entry2)
    assert agent.sample_memory(64) == [memory_entry1, memory_entry2]
    sample_mock.assert_called_with(ANY, 2)
