import math
import warnings

import numpy as np
import torch

MAX_QUANTILE_TENSOR_SIZE = 2 ** 24


def sigmoid(x):
    """Classic squashing function"""
    return 1 / (1 + math.exp(-x))


def robust_quantile(x, q, *args, **kwargs):
    """Wrapper for torch.quantile to handle images that are too big for it to compute"""
    if x.shape[0] > MAX_QUANTILE_TENSOR_SIZE:
        warnings.warn("Image is too big for torch.quantile (> 2^24 ~=17 million pixels). Taking "
                      "a sample of points to check sharpness instead of entire image.",
                      RuntimeWarning,
                      stacklevel=2)
        x = tensor_choice(x, MAX_QUANTILE_TENSOR_SIZE)
    return torch.quantile(x, q=q, *args, **kwargs)


def tensor_choice(tens: torch.Tensor, size: int):
    """Randomly selects 'size' entries from a torch tensor. Uses numpy due to size constraints for torch."""
    idx = np.random.default_rng().choice(tens.view(-1).shape[0], size, replace=False)
    return tens[idx]


def weighted_quantile_mean(x: torch.Tensor, weight: torch.Tensor | None = None, p: float = 4,
                           quantile: float = 0.995) -> float:
    """
    Finds the weighted pth-power mean of entries among the largest tensor-entries

    Note: used as a robust estimate of the maximum when p and quantile are large.

    :param x: The tensor we're taking the mean/maximum of
    :param weight: A mask indicating the region of the tensor we're considering (eg: an object in an image)
    :param p: The exponent in the mean
    :param quantile: The fraction of entries that we ignore because they're too small
    :return: The quantity of the weighted quantile mean
    """
    if weight is None:
        weight = torch.ones_like(x)

    if quantile:
        # If quantile > 0 we only consider the pixels greater than the quantile, hopefully corresponding to transition areas/edges
        x_flat = x[weight >= 0.5].view(-1)
        quantile_value = robust_quantile(x_flat, q=quantile)
        x, weight = x[x >= quantile_value], weight[x >= quantile_value]

    quantile_mean = ((weight * x ** p).sum() / weight.sum()) ** (1 / p)

    return float(quantile_mean)


def weighted_central_root_moment(x: torch.Tensor, weight: torch.Tensor | None = None, p: float = 4,
                                    quantile: float = 0.995):
    if weight is None:
        weight = torch.ones_like(x)

    weighted_mean = (x * weight).sum() / weight.sum()

    if quantile:
        # If quantile > 0 we only consider the pixels greater than the quantile, hopefully corresponding to transition areas/edges
        x_flat = x[weight >= 0.5].view(-1)
        quantile_value = robust_quantile(x_flat, q=quantile)
        x, weight = x[x >= quantile_value], weight[x >= quantile_value]

    return float((((x - weighted_mean) ** p * weight).sum() / weight.sum()) ** (1 / p))
