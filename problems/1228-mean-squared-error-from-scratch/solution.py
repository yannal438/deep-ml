import torch

def mse(pred, target):
    """
    Compute mean squared error between pred and target.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values (same shape as pred).

    Returns:
        float: Mean of squared differences.
    """
    # TODO
    ecart = (pred - target)**2
    mean = torch.mean(ecart)
    return float(mean.item())
