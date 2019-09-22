from unittest.mock import patch

from pungi.main import run
import tests.mock_policies


def test_main_incorrect_mode():
    assert -1 == run(["", "fly"])


class MockAgent:

    def persist(self, path):
        pass


@patch('time.time', return_value=123456)
@patch('pungi.agents.trainer.train', return_value=MockAgent())
@patch('pungi.persistence.save_q_table')
@patch('pungi.agents.policies.globals', return_value={"mock_policy": tests.mock_policies.mock_policy})
def test_main_train_mode(_policy, save, train, _time):
    run(["", "train", "q_table"])
    train.assert_called_once()


@patch('pungi.agents.qlearning.greedy_policy_agent.play_in_spectator_mode')
@patch('pungi.persistence.load_q_table', return_value="mock_q_table")
def test_main_play_mode(load, play):
    run(["", "play", "my-path/model.pkl"])
    load.assert_called_once_with("my-path/model.pkl")
    play.assert_called_once()


def test_main_play_mode_invalid_path():
    try:
        run(["", "play"])
        assert False
    except FileNotFoundError:
        pass
