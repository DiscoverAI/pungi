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

