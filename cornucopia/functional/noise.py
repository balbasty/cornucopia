__all__ = [
    "noisify_gaussian",
    "noisify_gamma",
    "noisify_chi",
]
from typing import Optional

from ..baseutils import prepare_output
from ..utils import smart_math as math
from ._utils import Tensor, Value, Output, _unsqz_spatial
from .random import random_field_gaussian_like


def noisify_gaussian(
    input: Tensor,
    std: Value = 0.1,
    gfactor: Optional[Tensor] = None,
    **kwargs
) -> Output:
    """
    Apply additive Gaussian noise

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    std : float | ([C],) tensor
        Standard deviation
    gfactor : ([C], *spatial) tensor
        Gfactor map that scales noise locally.

    Other Parameters
    ----------------
    returns : {"output", "input", "noise"}

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor

    """
    noise = random_field_gaussian_like(input, std=std)
    if gfactor is not None:
        noise = math.mul_(noise, gfactor)
    output = math.add_(noise, input)
    return prepare_output(
        {"input": input, "output": output, "noise": noise, "gfactor": gfactor},
        kwargs.pop("returns", "output")
    )()


def noisify_gamma(
    input: Tensor,
    std: Value = 0.1,
    gfactor: Optional[Tensor] = None,
    **kwargs
) -> Output:
    """
    Apply multiplicative Gamma noise

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    std : float | ([C],) tensor
        Standard deviation
    gfactor : ([C], *spatial) tensor
        Gfactor map that scales noise locally.

    Other Parameters
    ----------------
    returns : {"output", "input", "noise"}

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor

    """
    noise = random_field_gaussian_like(input, std=std)
    if gfactor is not None:
        noise = math.mul_(noise, gfactor)
    output = math.mul_(noise, input)
    return prepare_output(
        {"input": input, "output": output, "noise": noise},
        kwargs.pop("returns", "output")
    )()


def noisify_chi(
    input: Tensor,
    std: Value = 0.1,
    df: int = 2,
    gfactor: Optional[Tensor] = None,
    **kwargs
) -> Output:
    """
    Apply non-central Chi noise

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    std : float | ([C],) tensor
        Standard deviation.
    df : int
        Number of independant noise sources.
    gfactor : ([C], *spatial) tensor
        Gfactor map that scales noise locally.

    Other Parameters
    ----------------
    returns : {"output", "input", "noise"}

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor

    """
    # generate Chi-squared noise
    noise = 0
    for _ in range(df):
        noise += random_field_gaussian_like(input).square_()

    # scale to reach target variance
    mu = math.sqrt(2) * math.gamma((df+1)/2) / math.gamma(df/2)
    scale = (std * std) / (df - mu*mu)
    noise = math.mul_(noise, _unsqz_spatial(scale, input.ndim-1))

    # gfactor scaling (squared because noise is squared)
    if gfactor is not None:
        noise = math.mul_(math.mul_(noise, gfactor), gfactor)

    # apply noise
    output = math.sqrt_(math.add_(input.square(), noise))

    # sqrt to get Chi noise
    noise = math.sqrt_(noise)

    return prepare_output(
        {"input": input, "output": output, "noise": noise},
        kwargs.pop("returns", "output")
    )()
