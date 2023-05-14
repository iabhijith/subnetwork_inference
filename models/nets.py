from torch import nn
from typing import List


def create_mlp(input_size: int, hidden_sizes: List, output_size: int):
    layers = []
    layer_sizes = [input_size] + hidden_sizes
    for layer_index in range(1, len(layer_sizes)):
        layers += [
            nn.Linear(layer_sizes[layer_index - 1], layer_sizes[layer_index]),
            nn.ReLU(),
        ]
    layers += [nn.Linear(layer_sizes[-1], output_size)]
    return nn.Sequential(*layers)
