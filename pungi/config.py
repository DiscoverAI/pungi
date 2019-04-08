import json
import os
import importlib


def _load(config_file="/resources/default-config.json"):
    config_file = os.getenv("CONFIG_FILE", config_file)
    with open(os.getcwd() + config_file, 'r') as opened_file:
        return json.load(opened_file)


class _Config(object):
    def __init__(self):
        self._state = _load()

    def get_value(self, name):
        value = os.getenv(self._to_environ_varname(name))
        if not value:
            value = self._state[name]
        return value

    def get_policy(self, policy_name):
        module = importlib.import_module(self.get_value("policy_module"))
        return getattr(module, policy_name)

    @staticmethod
    def _to_environ_varname(string):
        return string.upper().replace('-', '_')


CONF = _Config()
