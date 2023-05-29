from torch import nn
from typing import List


def create_mlp(input_size: int, hidden_sizes: List, output_size: int):
    """Create a multi-layer perceptron (MLP) with ReLU activations.
    Parameters
    ----------
    input_size : int
        Input size.
    hidden_sizes : List
        Hidden sizes.
    output_size : int
        Output size.
    Returns
    -------
    nn.Sequential
    A multi-layer perceptron.
    """
    layers = []
    layer_sizes = [input_size] + hidden_sizes
    for layer_index in range(1, len(layer_sizes)):
        layers += [
            nn.Linear(layer_sizes[layer_index - 1], layer_sizes[layer_index]),
            nn.ReLU(),
        ]
    layers += [nn.Linear(layer_sizes[-1], output_size)]
    return nn.Sequential(*layers)
