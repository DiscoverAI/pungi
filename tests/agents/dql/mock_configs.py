three_layer_dense = {
    "graph": {
        "layers": [
            {"type": "input", "config": {"shape": [20, 20]}},
            {
                "type": "dense",
                "config": {
                    "num_neurons": 100,
                }
            },
            {
                "activation": "relu"
            },
            {
                "type": "dense",
                "config": {
                    "num_neurons": 100,
                }
            },
            {
                "activation": "relu"
            },
            {
                "type": "dense",
                "config": {
                    "num_neurons": 4,
                }
            },
            {
                "activation": "linear"
            }
        ]
    },
    "optimizer": {"type": "adam"}
}
