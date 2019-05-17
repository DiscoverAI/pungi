import pungi.persistence as persistence
from collections import defaultdict
import os
import shutil


def test_save_simple_q_table():
    test_q_table = defaultdict(lambda: 5)

    test_q_table[0] = (4, 5, 6, 7, 8)
    test_q_table[(1, 2)] = (7, 5, 4, 3, 2)

    test_dir = "./tests/out/"
    filename = "model.pkl"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    persistence.save(test_q_table, test_dir + filename)
    loaded = persistence.load(test_dir + filename)
    assert test_q_table == loaded
    assert 5 == loaded["non_existing_key"]
    shutil.rmtree(test_dir)
