from gym.envs.registration import register

register(
    id='snake-v1',
    entry_point='pungi.environment.environment:SnakeEnv',
)

register(
    id='snake-partial-info-v1',
    entry_point='pungi.environment.environment:PartialInformationSnakeEnv'
)
