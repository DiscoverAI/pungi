import pungi.trainer as trainer
from pungi.ml_agent import play_in_spectator_mode
import pungi.persistence as persistence
import time

if __name__ == '__main__':
    q_table = trainer.train()
    persistence.save(q_table, "./out/model-" + str(int(time.time())) + ".pkl")
    for i in range(5):  # play some example games
        play_in_spectator_mode(q_table)
