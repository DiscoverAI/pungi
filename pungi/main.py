import logging
import sys
import time

import pungi.metrics as metrics
import pungi.persistence as persistence
import pungi.trainer as trainer
from pungi.agents.qlearning import agent

logger = logging.getLogger(__name__)


def load_q_table_from_args(argv):
    if len(argv) <= 2:
        raise FileNotFoundError("Please provide a model path that the agent should use to play.")
    q_table_file = argv[2]
    q_table = persistence.load(q_table_file)
    return q_table


def run(argv):
    mode = argv[1]
    if mode == "train":
        q_table = trainer.train()
        persistence.save(q_table, "./out/model-" + str(int(time.time())) + ".pkl")
    elif mode == "test":
        logging.info("Test mode not implemented yet")
        # Here we should make some test runs and write metrics such as average total reward.
        # Maybe we should do that in JSON format to integrate with dvc metrics tracking?
        # https://github.com/iterative/dvc.org/blob/master/static/docs/get-started/compare-experiments.md
    elif mode == "play":
        q_table = load_q_table_from_args(argv)
        agent.play_in_spectator_mode(q_table)
    elif mode == "eval":
        q_table = load_q_table_from_args(argv)
        metrics.calculate_and_write_metrics(episodes=10,
                                            q_table=q_table,
                                            output_path="./out/metrics-" + str(int(time.time())) + ".json")
    else:
        logging.warning("First argument must be either train, test or play.")
        return -1


if __name__ == '__main__':
    run(sys.argv)
