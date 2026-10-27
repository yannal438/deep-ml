import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    t_stable = t - torch.max(t, dim=dim, keepdim=True)[0]
    soft_num = torch.exp((t_stable))
    soft_de = torch.sum(soft_num, dim=dim, keepdim=True)
    softma = soft_num / soft_de
    return softma

