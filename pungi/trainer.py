import pungi.qlearning as qlearning
import pungi.environment.environment as environment
import pungi.config as conf
import pungi.policies as policies


def run_episode(q_table, policy):
    learning_rate = float(conf.CONF.get_value("learning_rate"))
    discount_factor = float(conf.CONF.get_value("discount_factor"))
    game_id, state = environment.reset()
    game_over = False
    while not game_over:
        next_action = qlearning.next_move(q_table, state, policy)
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
    policy = policies.get_policy(policy_name=conf.CONF.get_value("policy"))
    while episodes < int(conf.CONF.get_value("episodes")):
        q_table = run_episode(q_table, lambda q_values: policy(q_values, episodes))
        episodes += 1
    return q_table
