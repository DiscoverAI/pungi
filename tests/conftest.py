import os

original_config_file = ""
original_backend = ""


def pytest_sessionstart(session):
    """ before session.main() is called. """
    global original_config_file
    global original_backend
    original_config_file = str(os.getenv("CONFIG_FILE"))
    original_backend = str(os.getenv("BACKEND"))
    os.environ["CONFIG_FILE"] = "/tests/resources/test-config.json"
    os.environ["BACKEND"] = "foo bar"


def pytest_sessionfinish(session, exitstatus):
    """ whole test run finishes. """
    os.environ["CONFIG_FILE"] = original_config_file
    os.environ["BACKEND"] = original_backend
