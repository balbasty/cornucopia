__all__ = [
    "smooth",
    "conv",
    "conv1d",
    "random_kernel",
]
from typing import Union, Sequence, Optional

import torch

from ..baseutils import prepare_output
from ..utils.warps import identity
from ..utils.conv import smoothnd, convnd
from ..utils.py import ensure_list
from ..utils import smart_math as math
from ._utils import Tensor, Value, Output, OneOrMore,  _axis_name2index
from .random import random_field


def smooth(
    input: Tensor,
    fwhm: Value,
    iso: bool = False,
    bound: str = "reflect",
    **kwargs
) -> Output:
    """
    Smooth an image with a Gaussian kernel.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    fwhm : float | ([C, D],) tensor
        Full width at half-maximum of the Gaussian kernel.
    iso : bool
        Isotropic smoothing.
        This only matters when `fwhm` is a vector.
        If True, it is assumed to be a vector of length `C` (one isotropic
        kernel per channel). If False, it is assumed to be a vector or
        length `D` (one anisotropic kernel shared across channels).
    bound : {"zero", "reflect", "mirror", "circular", ...}
        Boundary conditions.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "fwhm"}

    Returns
    -------
    output : (C, *sptial) tensor
        Output tensor.

    """
    ndim = input.ndim - 1
    fwhm = torch.as_tensor(fwhm)

    if fwhm.ndim == 1:
        fwhm = fwhm[:, None] if iso else fwhm[None, :]
    elif fwhm.ndim == 0:
        fwhm = fwhm[None, None]
    fwhm = fwhm.expand([len(fwhm), ndim])

    if len(fwhm) != 1:
        output = torch.stack([
            smoothnd(inp1, fwhm=fwhm1)
            for inp1, fwhm1 in zip(input, fwhm)
        ])
    else:
        output = smoothnd(input, fwhm=fwhm[0], bound=bound)

    return prepare_output(
        {"output": output, "input": input},
        kwargs.get("returns", "output")
    )()


def conv(
    input: Tensor,
    kernel: OneOrMore[Tensor],
    bound: str = "reflect",
    **kwargs
) -> Output:
    """
    Convolve an image with a kernel.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    kernel : [list of] ([[K], C], *kernel_size) tensor
        Convolution kernel.
        * If its size is `(*kernel_size)`, the same kernel is applied to
          all channels.
        * If its size is `(C, *kernel_size)`, each channel is convolved
          with its own kernel.
        * If its size is `(K, C, *kernel_size)`, channels are mixed
          by the convolution kernel, and `K` is the output number of
          channels.
        * If it is a list, it must contain `ndim` 1D kernels, which
          will be applied in order along the spatial dimensions, from
          left to right.
    bound : {"zero", "reflect", "mirror", "circular", ...}
        Boundary conditions.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "kernel"}

    Returns
    -------
    output : (K, *spatial) tensor
        Output tensor.

    """
    ndim = input.ndim - 1

    # separable convolution
    if isinstance(kernel, list):
        if len(kernel) != ndim:
            raise ValueError(f"Expected {ndim} kernels but got {len(kernel)}.")
        return conv1d(input, kernel, list(range(ndim), bound=bound), **kwargs)

    # n-dimensional convolution
    output = convnd(ndim, input, kernel, bound=bound, padding="same")
    return prepare_output(
        {"input": input, "output": output, "kernel": kernel},
        kwargs.get("returns", "output")
    )()


def conv1d(
    input: Tensor,
    kernel: OneOrMore[Tensor],
    axis: OneOrMore[Union[int, str]] = -1,
    orient: Union[str, Tensor] = "RAS",
    bound: str = "reflect",
    **kwargs
) -> Output:
    """
    Convolve an image with a 1D kernel along a given dimension.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    kernel : [list of] ([[K], C], kernel_size) tensor
        Convolution kernel.
        * If its size is `(kernel_size,)`, the same kernel is applied to
          all channels.
        * If its size is `(C, kernel_size)`, each channel is convolved
          with its own kernel.
        * If its size is `(K, C, kernel_size)`, channels are mixed
          by the convolution kernel, and `K` is the output number of
          channels.
        * If it is a list, kernels are applied in sequence, and `axis`
          must contain as many axes as kernels.
    axes : int | {"LR", "AP", "IS"}
        Axes to flip, by index or by name.
        Indices correspond to spatial axes only (0 = first spatial dim, etc.)
        If None, flip all spatial axes.
    orient : str or tensor
        Tensor layout (`{"RAS", "LPS", ...}`) or orient matrix.
    bound : {"zero", "reflect", "mirror", "circular", ...}
        Boundary conditions.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "kernel", "axis"}

    Returns
    -------
    output : (K, *spatial) tensor
        Output tensor.

    """
    ndim = input.ndim - 1

    axis_ = axis
    if any(isinstance(ax, str) for ax in axis_):
        axis_ = _axis_name2index(axis_, orient)
    axis_ = ensure_list(axis_)

    axis_ = [ndim + ax if ax < 0 else ax for ax in axis_]
    kernel_ = ensure_list(kernel, len(axis_))

    output = input
    for ax, ker in zip(axis_, kernel_):
        kernel_size = [1] * ndim
        kernel_size[ax] = ker.shape[-1]
        ker = ker.reshape(ker.shape[:-1] + tuple(kernel_size))

        output = conv(ndim, output, ker, bound=bound)

    return prepare_output(
        {"input": input, "output": output, "kernel": kernel, "axis": axis},
        kwargs.get("returns", "output")
    )()


def random_kernel(
    shape: Sequence[int],
    norm: Optional[float] = 1,
    zero_mean: bool = False,
    allow_translations: bool = False,
    distrib: Optional[str] = "gamma",
    **kwargs
) -> Output:
    """
    Generate a random convolution kernel.

    !!! example "Examples"
        ```python
        shape = [1] + [5] * ndim

        # smoothing kernel (positive values, sum to one)
        kernel = random_kernel(shape, distrib="gamma")

        # differential kernel (pos and neg values, sum to zero)
        kernel = random_kernel(shape, zero_mean=True, distrib="gaussian")

        # purely random kernels -- may shift data
        kernel = random_kernel(shape, allow_translations=True, distrib="gaussian")
        ```

    To generate a smoothing kernel:


    Parameters
    ----------
    shape : (C, *spatial) list[int]
        Output kernel shape, including the channel dimension.
        The spatial size should be odd.
    norm : float
        Ensure that the kernel has unit norm of order `p`.
        If `None`, do not normalize the kernel.
    zero_mean : bool
        If `True`, ensure that the kernel sums to zero.
    allow_translations : bool
        If `False`, ensure that the kernel's barycenter is its center.
        This ensures that the kernel does not "translate" data.
        (otherwise, a kernel such as `[1, 0, 0]`, which implements a
        1-voxel translation, would be valid).
        If `True`, any kernel is allowed.
    distrib : {"uniform", "gamma", "lognormal", "gaussian", "generalized"}
        Probability distribution.
        Gamma and lognormal always return positive values (default mean: 1).
        Normal and generalized may return negative values (default mean: 0).
        The value range returned by uniform depends on its parameters
        (default: [0, 1]).
        Defaults depend on the other parameters:
            * when `sum > 0`, we use `"gamma"` with `mean=1, std=1`
            * when `sum == 0`, we use `"gaussian"` with `mean=0, std=0.2`

    Other Parameters
    ----------------
    mean, std, peak, fwhm, ... : float | (C,) tensor
        Distribution parameters.
    dtype : torch.dtype
        Output data type.
    device : torch.device
        Output device.
    returns : [list or dict of] {"output"}
        Tensors to return.

    Returns
    -------
    output : (*shape) tensor
        Output kernel.
    """  # noqa: E501
    returns = kwargs.pop("returns", "output")

    shape = list(shape)
    ndim = len(shape) - 1
    C = shape[0]
    if not all(s % 2 for s in shape[1:]):
        raise ValueError("Spatial kernel size must be odd.")

    if sum is None:
        distrib = distrib or "gaussian"
        kwargs.setdefault("std", 0.2)
    elif sum == 0:
        distrib = distrib or "gaussian"
        kwargs.setdefault("std", 0.2)
    else:
        distrib = distrib or "gamma"
        kwargs.setdefault("std", 1)

    # sample values
    output = random_field(distrib, shape, **kwargs)

    # undo translation
    if not allow_translations:
        # compute kernel barycenter
        backend = dict(dtype=output.dtype, device=output.device)
        size = torch.as_tensor(shape[1:], **backend)
        grid = identity(shape[1:], **backend)
        grid -= (size - 1) / 2
        bary = output.abs().reshape([C, -1]).matmul(grid.reshape([-1, ndim]))
        bary /= output.abs().reshape([C, -1]).sum(-1, keepdim=True)

        # build convolution kernel that applies a translation of `-bary`
        # (with linear interpolation)
        new_shape = size + 2 * bary.abs().max(0).values
        new_shape = new_shape.ceil().long()
        new_shape += (1 - new_shape % 2)
        new_shape = [C] + new_shape.tolist()
        translation_kernels = []
        for c in range(C):
            translation_kernel = 1
            for d in range(ndim):
                s = new_shape[1+d]
                k = torch.zeros([s], **backend)
                b = (s - 1) / 2 + bary[c, d]
                k[b.floor().long()] = 1 - (b - b.floor())
                k[b.ceil().long()] = 1 - (b.ceil() - b)
                translation_kernel = translation_kernel * k
                translation_kernel = translation_kernel[..., None]
            translation_kernel = translation_kernel[..., 0]
            translation_kernels.append(translation_kernel)
        translation_kernel = torch.stack(translation_kernels)

        shape = new_shape
        translation_kernel = translation_kernel.expand(shape)

        # convolve both kernels
        output = convnd(ndim, translation_kernel, output, padding="same")

        # ## DEBUG: check bary
        # size = torch.as_tensor(shape[1:], **backend)
        # grid = identity(shape[1:], **backend)
        # grid -= (size - 1) / 2
        # bary = output.abs().reshape([C, -1]).matmul(grid.reshape([-1, ndim]))
        # bary /= output.abs().reshape([C, -1]).sum(-1, keepdim=True)
        # print(bary)

    # zero mean
    if zero_mean:
        mean = output.sum(list(range(-ndim-1, 0)), keepdim=True)
        output = math.sub_(output, mean)

    # normalize
    if norm is not None:
        p = norm
        if p == 0:
            norm = output.abs().reshape([C, -1])
            norm = norm.max(-1).values
            for _ in range(ndim):
                norm = norm[..., None]
        else:
            norm = output.abs().pow(p)
            norm = norm.sum(list(range(-ndim-1, 0)), keepdims=True)
            norm = output.pow(1/p)
        output = math.div_(output, norm)

    return prepare_output({"output": output}, returns)()
