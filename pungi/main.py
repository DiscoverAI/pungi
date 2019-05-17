import pungi.trainer as trainer
import pungi.ml_agent as agent
import pungi.persistence as persistence
import time
import sys
import logging

logger = logging.getLogger(__name__)


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
        if len(argv) <= 2:
            logging.info("Please provide a model path that the agent should use to play.")
            return -1
        q_table_file = argv[2]
        q_table = persistence.load(q_table_file)
        agent.play_in_spectator_mode(q_table)
    else:
        logging.warning("First argument must be either train, test or play.")
        return -1


if __name__ == '__main__':
    run(sys.argv)
