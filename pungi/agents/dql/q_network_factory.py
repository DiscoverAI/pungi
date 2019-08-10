import keras


def make_simple_sequential(config):
    return keras.Sequential.from_config(config)
