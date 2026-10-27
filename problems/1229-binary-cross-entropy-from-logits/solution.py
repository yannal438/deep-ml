import torch

def bce_with_logits(logits, targets):
    """Mean BCE-with-logits loss, numerically stable, rounded to 4 decimals.

    Args:
        logits (torch.Tensor): 1-D raw logits.
        targets (torch.Tensor): 1-D binary targets in {0, 1}, same shape.

    Returns:
        float: mean loss rounded to 4 decimal places.
    """
    # TODO: stable BCE-with-logits, mean, round to 4 decimals
    max_logits = torch.clamp(logits, min=0.0)
    abs_logits = torch.abs(logits)
    logits_log = torch.log1p(torch.exp(-abs_logits))
    L = max_logits - (logits * targets) + logits_log
    mean_logits = torch.mean(L)
    rounded = torch.round(mean_logits, decimals=4)
    return float(rounded.item())
