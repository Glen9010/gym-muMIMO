from gym.envs.registration import register

register(
    id='muMIMO-v0',
    entry_point='gym_muMIMO.envs:muMIMOEnv'
)