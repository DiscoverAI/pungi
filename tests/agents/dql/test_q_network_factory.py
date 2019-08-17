import json
import pungi.agents.q_network_factory as factory


def test_load_simple_model():
    with open("tests/resources/simple-dqn-architecture.json", "r") as config_file:
        config = json.load(config_file)
        model = factory.make_simple_sequential(config)
        assert len(model.layers) == 2
        dense1, dense2 = model.layers
        assert dense1.use_bias
        assert dense2.use_bias
        assert dense1.input_shape == (None, 5)
        assert model.count_params() == 22
        assert dense1.count_params() == 18


def test_load_convolutional_model():
    with open("tests/resources/conv-dqn-architecture.json", "r") as config_file:
        config = json.load(config_file)
        model = factory.make_simple_sequential(config)
        assert len(model.layers) == 3
        conv, flatten, dense = model.layers
        assert conv.use_bias
        assert dense.use_bias
        assert conv.input_shape == (None, 10, 10, 1)
        assert conv.output_shape == (None, 5, 5, 8)
        assert conv.count_params() == 4 * 8 + 8
        assert dense.count_params() == (5 * 5 * 8) * 4 +4
        assert model.count_params() == conv.count_params() + dense.count_params()
