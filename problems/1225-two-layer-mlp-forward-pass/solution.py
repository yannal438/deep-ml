import torch
import torch.nn as nn


def two_layer_mlp_forward(x, w1, b1, w2, b2):
    """Build a 2-layer MLP, set fixed weights, return scalar output.

    Args:
        x (torch.Tensor): Input of shape (1, 2).
        w1 (torch.Tensor): First Linear weight, shape (2, 2).
        b1 (torch.Tensor): First Linear bias, shape (2,).
        w2 (torch.Tensor): Second Linear weight, shape (1, 2).
        b2 (torch.Tensor): Second Linear bias, shape (1,).

    Returns:
        float: Scalar network output.
    """
    # TODO
    neuron1 = nn.Linear(2, 2)
    neuron2 = nn.Linear(2, 1)

    with torch.no_grad():
        neuron1.weight.copy_(w1)
        neuron1.bias.copy_(b1)
        neuron2.weight.copy_(w2)
        neuron2.bias.copy_(b2)
    hidden_output = torch.relu(neuron1(x))
    final_output = neuron2(hidden_output)
    return float(final_output.item())
