from unittest.mock import patch

from pungi.main import run


def test_main_incorrect_mode():
    assert -1 == run(["", "fly"])


@patch('time.time', return_value=123456)
@patch('pungi.trainer.train', return_value="mock_q_table")
@patch('pungi.persistence.save')
def test_main_train_mode(save, train, _):
    run(["", "train"])
    train.assert_called_once()
    save.assert_called_once_with("mock_q_table", "./out/model-123456.pkl")


@patch('pungi.agents.qlearning.agent.play_in_spectator_mode')
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
