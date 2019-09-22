import pungi.agents.qlearning.agent as qlearning_agent
from unittest.mock import patch, ANY


@patch("pungi.agents.policies.get_policy", return_value="mock_policy")
@patch("pungi.agents.qlearning.qlearning.initialize_q_table", return_value="mock_initial_q")
@patch("pungi.agents.qlearning.qlearning.next_move", return_value="mock_move")
@patch("pungi.agents.qlearning.qlearning.update_q_value", return_value="mock_new_q_table")
def test_agent(update_q_table_mock, next_move_mock, init_q_table_mock, get_policy_mock):
    agent = qlearning_agent.QLearningAgent()
    init_q_table_mock.assert_called_once()
    get_policy_mock.assert_called_once_with(policy_name="mock_policy")
    assert agent.q_table == "mock_initial_q"
    assert agent.policy == "mock_policy"
    assert agent.learning_rate == 0.9
    assert agent.discount_factor == 0.99
    state = (0, 0)
    next_state = (0, 1)
    reward = -1
    next_action = agent.next_action(state, episode_number=42)
    assert next_action == "mock_move"
    next_move_mock.assert_called_once_with(agent.q_table, state, ANY)
    agent.update(state, next_action, next_state, reward, False)
    update_q_table_mock.assert_called_once_with("mock_initial_q",
                                                state,
                                                next_action,
                                                next_state,
                                                agent.learning_rate,
                                                agent.discount_factor,
                                                reward)

