from unittest.mock import patch

from pungi.main import run
import tests.mock_policies

def test_main_incorrect_mode():
    assert -1 == run(["", "fly"])


@patch('time.time', return_value=123456)
@patch('pungi.agents.trainer.train', return_value="mock_q_table")
@patch('pungi.persistence.save')
@patch('pungi.agents.qlearning.policies.globals', return_value={"mock_policy": tests.mock_policies.mock_policy})
def test_main_train_mode(_policy, save, train, _time):
    run(["", "train"])
    train.assert_called_once()
    save.assert_called_once_with("mock_q_table", "./out/model-123456.pkl")


@patch('pungi.agents.qlearning.greedy_policy_agent.play_in_spectator_mode')
@patch('pungi.persistence.load', return_value="mock_q_table")
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
