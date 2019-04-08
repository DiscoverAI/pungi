import pungi.qlearning as qlearning
import pungi.environment.environment as environment
import pungi.config as conf


def run_episode(q_table):
    learning_rate = float(conf.CONF.get_value("learning_rate"))
    discount_factor = float(conf.CONF.get_value("discount_factor"))
    game_id, state = environment.reset()
    reward = None
    while reward != -100:
        next_action = qlearning.next_move(q_table, state, qlearning.max_policy)
        reward, next_state, game_over, info = environment.step(next_action, game_id)
        q_table = qlearning.update_q_value(q_table,
                                           state,
                                           next_action,
                                           next_state,
                                           learning_rate,
                                           discount_factor,
                                           reward)
        state = next_state
    return q_table


def train():
    episodes = 0
    q_table_initial_value = float(conf.CONF.get_value("q_table_initial_value"))
    q_table = qlearning.initialize_q_table(initial_value=q_table_initial_value)
    while episodes < int(conf.CONF.get_value("episodes")):
        q_table = run_episode(q_table)
        episodes += 1
    return q_table
