import pickle
from collections import defaultdict


def save_q_table(test_q_table, path):
    with open(path, "wb") as file_pointer:
        test_q_table["default_value"] = test_q_table.default_factory()
        pickle.dump(dict(test_q_table), file_pointer)


def load_q_table(path):
    with open(path, "rb") as file_pointer:
        q_table = pickle.load(file_pointer)
        default_value = q_table["default_value"]
        with_default_value = defaultdict(lambda: default_value)
        with_default_value.update(q_table)
        return with_default_value
