import os

original_config_file = ""


def pytest_sessionstart(session):
    """ before session.main() is called. """
    global original_config_file
    original_config_file = str(os.getenv("CONFIG_FILE"))
    os.environ["CONFIG_FILE"] = "/tests/resources/test-config.json"


def pytest_sessionfinish(session, exitstatus):
    """ whole test run finishes. """
    os.environ["CONFIG_FILE"] = original_config_file
