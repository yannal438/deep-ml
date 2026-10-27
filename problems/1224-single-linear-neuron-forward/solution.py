import torch
import torch.nn as nn


def single_neuron_forward(x):
    """Forward pass of one fixed linear neuron.

    Args:
        x: torch.Tensor of shape (1, 3).

    Returns:
        Python float, the neuron output.
    """
    # TODO: build nn.Linear(3, 1), set fixed weight/bias under no_grad, return float output
    neuron = nn.Linear(3, 1)
    with torch.no_grad():
        fixed_weight = torch.tensor([[0.5, -0.2, 0.3]])
        fixed_bias = torch.tensor([0.1])
        neuron.weight.copy_(fixed_weight)
        neuron.bias.copy_(fixed_bias)
    output = neuron(x)
    return float(output.item())
