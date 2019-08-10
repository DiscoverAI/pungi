import json
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D
import pungi.agents.dql.q_network_factory as factory


def test_load_simple_model():
    with open("tests/resources/simple-dqn-architecture.json", "r") as config_file:
        config = json.load(config_file)
        model = factory.make_simple_sequential(config)
        print(model.summary())
        # assert model.count_params() == ?
        assert len(model.layers) == 2
        dense1, dense2 = model.layers
        assert dense1.use_bias
        assert dense2.use_bias
        assert dense1.input_shape == (None, 5)
        # assert dense1.count_params() == ?


def test_load_convolutional_model():
    with open("tests/resources/conv-dqn-architecture.json", "r") as config_file:
        config = json.load(config_file)
        model = factory.make_simple_sequential(config)
        print(model.summary())
        # assert model.count_params() == ?
        assert len(model.layers) == 3
        conv, flatten, dense = model.layers
        assert conv.use_bias
        assert dense.use_bias
        assert conv.input_shape == (None, 10, 10, 1)
        # assert conv.output_shape == ?
        # assert conv.count_params() == ?
        # assert dense.count_params() == ?
        # assert model.count_params() == ?
