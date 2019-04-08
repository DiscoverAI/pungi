from pungi.config import CONF
import os
from contextlib import contextmanager


@contextmanager
def _setup_environ(name, new_value):
    original_value = os.getenv(name)
    try:
        os.environ[name] = new_value
        yield
    finally:
        if original_value:
            os.environ[name] = original_value
        else:
            del os.environ[name]


def test_load_config_from_file():
    assert "foo bar" == CONF.get_value("backend")


def test_load_config_from_environment_value():
    with _setup_environ('BACKEND', 'foobar'):
        assert "foobar" == CONF.get_value("backend")
