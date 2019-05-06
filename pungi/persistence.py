import pickle


def save(test_q_table, path):
    with open(path, "wb") as file_pointer:
        pickle.dump(dict(test_q_table), file_pointer)


def load(path):
    with open(path, "rb") as file_pointer:
        return pickle.load(file_pointer)
