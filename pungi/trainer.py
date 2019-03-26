import pungi.qlearning as qlearning

# Just a sketch for todays meetup!
# will be TDD'ed from scratch!!

episodes = 0

q_table = qlearning.initialize_q_table(0)

learning_rate = 0.9
discount_factor = 0.99

while episodes < 100:
    game_id = client.register_new_game(board_width=5, board_height=5, snake_length=1)

    game_info = client.get_game_info(game_id)
    current_state = game_info["tokens"]["snake"]["position"][0]
    reward = None
    while reward != -100:
        action = qlearning.next_move(q_table=q_table, current_state=current_state, policy=qlearning.max_policy)
        game_info = client.make_move(direction=action, game_id=game_id)
        next_state = qlearning.get_state_from_game_info(game_info)
        reward = qlearning.get_reward(game_info)
        qlearning.update_q_value(q_table=q_table,
                                 state=current_state,
                                 next_state=next_state,
                                 action=action,
                                 reward=reward,
                                 learning_rate=learning_rate,
                                 discount_factor=discount_factor)
        current_state = next_state
    episodes += 1
