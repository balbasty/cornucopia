# stdlib
import math
from functools import partial

import torch
from torch.nn import functional as F

from .py import ensure_list
from .padding import pad


def pool(dim, tensor, kernel_size=3, stride=None, dilation=1, padding=0,
         bound='constant', reduction='mean', ceil=False):
    """Perform a pooling

    Parameters
    ----------
    dim : {1, 2, 3}
        Number of spatial dimensions
    tensor : (*batch, *spatial_in) tensor
        Input tensor
    kernel_size : int or sequence[int], default=3
        Size of the pooling window. If <= 0, pool the entire dimension.
    stride : int or sequence[int], default=`kernel_size`
        Strides between output elements.
    dilation : int or sequence[int], default=1
        Strides between elements of the kernel.
    padding : 'same' or int or sequence[int], default=0
        Padding performed before the convolution.
        If 'same', the padding is chosen such that the shape of the
        output tensor is `floor(spatial_in / stride)` (or
        `ceil(spatial_in / stride)` if `ceil` is True).
    bound : str, default='constant'
        Boundary conditions used in the padding.
    reduction : {'mean', 'max', 'min', 'median', 'sum', 'ssq'} or callable, default='mean'
        Function to apply to the elements in a window.
    ceil : bool, default=False
        Use ceil instead of floor to compute output shape

    Returns
    -------
    pooled : (*batch, *spatial_out) tensor

    """  # noqa: E501
    # move everything to the same dtype/device
    tensor = torch.as_tensor(tensor)

    # sanity checks + reshape for torch's conv
    batch = tensor.shape[:-dim]
    spatial_in = tensor.shape[-dim:]
    tensor = tensor.reshape([-1, *spatial_in])

    # compute padding
    kernel_size = ensure_list(kernel_size, dim)
    kernel_size = [s if ks <= 0 else ks
                   for s, ks in zip(spatial_in, kernel_size)]
    stride = ensure_list(stride or None, dim)
    stride = [st or ks for st, ks in zip(stride, kernel_size)]
    dilation = ensure_list(dilation or 1, dim)
    padding = compute_conv_padding(spatial_in, kernel_size, padding,
                                   dilation, stride, ceil)
    if ceil:
        # ceil mode cannot be obtained using unfold. we may need to
        # pad the input a bit more
        padding = _pad_for_ceil(
            spatial_in, kernel_size, padding, stride, dilation
        )

    use_torch = (reduction in ('mean', 'avg', 'max') and
                 dim in (1, 2, 3) and
                 dilation == [1] * dim)

    sum_padding = sum([sum(p) if isinstance(p, (list, tuple)) else p
                       for p in padding])
    if ((not use_torch) or (bound != 'zero' and sum_padding > 0)
            or any(isinstance(p, (list, tuple)) for p in padding)):
        # torch implementation -> handles zero-padding
        # our implementation -> needs explicit padding
        padding = _normalize_padding(padding)
        tensor = pad(tensor, padding, bound, side='both',
                     value=_fill_value(reduction, tensor))
        padding = [0] * dim

    pool_fn = reduction if callable(reduction) else None

    if use_torch:
        if reduction in ('mean', 'avg'):
            pool_fn = (F.avg_pool1d if dim == 1 else
                       F.avg_pool2d if dim == 2 else
                       F.avg_pool3d if dim == 3 else None)
            if pool_fn:
                pool_fn0 = pool_fn
                pool_fn = (  # noqa: E731
                    lambda x, *a, **k: pool_fn0(
                        x[:, None], *a, **k, padding=padding)[:, 0]
                )
        elif reduction == 'max':
            pool_fn = (F.max_pool1d if dim == 1 else
                       F.max_pool2d if dim == 2 else
                       F.max_pool3d if dim == 3 else None)
            if pool_fn:
                pool_fn0 = pool_fn
                pool_fn = (  # noqa: E731
                    lambda x, *a, **k: pool_fn0(
                        x[:, None], *a, **k, padding=padding)[:, 0]
                )

    if not pool_fn:
        if reduction == 'mean':
            reduction = partial(torch.nanmean, dim=-1)
        elif reduction == 'sum':
            reduction = partial(torch.nansum, dim=-1)
        elif reduction == 'min':
            reduction = partial(torch.min, dim=-1)
        elif reduction == 'max':
            reduction = partial(torch.max, dim=-1)
        elif reduction == 'median':
            reduction = partial(torch.nanmedian, dim=-1)
        if reduction == 'ssq':
            reduction = partial(_ssq, dim=-1)
        elif not callable(reduction):
            raise ValueError(f'Unknown reduction {reduction}')
        pool_fn = partial(_pool, dilation=dilation, reduction=reduction)

    tensor = pool_fn(tensor, kernel_size, stride=stride)
    spatial_out = tensor.shape[-dim:]
    tensor = tensor.reshape([*batch, *spatial_out])

    return tensor


def _pool(x, kernel_size, stride, dilation, reduction):
    """Implement pooling by "manually" extracting patches using `unfold`

    Parameters
    ----------
    x : (batch, *spatial)
    kernel_size : (dim,) int
    stride : (dim,) int
    dilation : (dim,) int
    reduction : callable
        This function should collapse the last dimension of a tensor
        (..., K) tensor -> (...)

    Returns
    -------
    x : (batch, *spatial_out)

    """
    kernel_size = [(sz-1)*dl + 1 for sz, dl in zip(kernel_size, dilation)]
    for d, (sz, st, dl) in enumerate(zip(kernel_size, stride, dilation)):
        x = x.unfold(dimension=d + 1, size=sz, step=st)
        if dl != 1:
            x = x[..., ::dl]
    dim = len(kernel_size)
    x = x.reshape((*x.shape[:dim+1], -1))
    x = reduction(x)
    return x


pool1d = partial(pool, 1)
pool2d = partial(pool, 2)
pool3d = partial(pool, 3)


def _ssq(x, dim):
    return torch.mean(x.square(), dim=dim).sqrt()


def compute_conv_padding(input_size, kernel_size, padding, dilation=1,
                         stride=1, ceil=False):
    """Compute the amount of padding to apply

    Parameters
    ----------
    input_size : sequence of int
        Spatial shape of the input tensor.
    kernel_size : [sequence of] int
    padding : [sequence of] {'valid', 'same'} or int or (int, int)
        Padding type (if str) or symmetric amount (if int) or
        low/high amount (if [int, int]).
    dilation : [sequence of] int, default=1
        The effective size of the kernel is
        `(kernel_size - 1) * dilation + 1`
    stride : [sequence of] int, default=1
    ceil : bool, default=False
        Ceil mode used to compute output shape
        (tensorflow uses True, pytorch uses False by default)

    Returns
    -------
    padding : tuple of int or (int, int)

    """
    # https://stackoverflow.com/questions/37674306/ (Answer by Vaibhav Dixit)

    dim = len(input_size)
    kernel_size = ensure_list(kernel_size, dim)
    dilation = ensure_list(dilation, dim)
    stride = ensure_list(stride, dim)
    kernel_size = [(k-1) * d + 1 for (k, d) in zip(kernel_size, dilation)]
    padding = ensure_list(padding, dim)

    padding = [0 if p == 'valid'
               else _same_padding(i, k, s, ceil) if p in ('same', 'auto')
               else p if isinstance(p, int)
               else tuple(ensure_list(p))
               for p, i, k, s in zip(padding, input_size, kernel_size, stride)]
    if not all(isinstance(p, int) or
               (isinstance(p, tuple) and len(p) == 2
                and all(isinstance(pp, int) for pp in p)) for p in padding):
        raise ValueError('Invalid padding', padding)
    return padding


def _normalize_padding(padding):
    """Ensure that padding has format (left, right, top, bottom, ...)"""
    if all(isinstance(p, int) for p in padding):
        return padding
    else:
        npadding = []
        for p in padding:
            if isinstance(p, (list, tuple)):
                npadding.extend(p)
            else:
                npadding.append(p)
                npadding.append(p)
        return npadding


def _same_padding(in_size, kernel_size, stride, ceil):
    if not ceil:
        # This is equivalent to the formula below, but with floor
        # instead of ceil. I find this more readable though.
        padding = (kernel_size - stride - in_size % stride)
    else:
        out_size = math.ceil(float(in_size) / float(stride))
        padding = max((out_size - 1) * stride + kernel_size - in_size, 0)
    padding = max(0, padding)
    if padding % 2 == 0:
        return padding // 2
    else:
        return (padding // 2, padding - padding // 2)


def _pad_for_ceil(input_size, kernel_size, padding, stride, dilation):
    new_padding = []
    for i in range(len(input_size)):
        L = input_size[i]
        S = stride[i]
        P = padding[i]
        K = kernel_size[i]
        D = dilation[i]
        K = D * (K - 1) + 1
        sumP = P
        if isinstance(P, (list, tuple)):
            sumP = sum(P)
        extra_pad = (L + sumP - K) % S
        if extra_pad:
            extra_pad = S - extra_pad
        if isinstance(P, (list, tuple)):
            pad = (padding[i][0], padding[i][1] + extra_pad)
        else:
            pad = padding[i] + extra_pad
        new_padding.append(pad)
    return new_padding


def _fill_value(reduction, tensor):
    if reduction == 'max':
        if tensor.dtype.is_floating_point:
            fill_value = -float('inf')
        else:
            fill_value = tensor.min()
    elif reduction == 'min':
        if tensor.dtype.is_floating_point:
            fill_value = float('inf')
        else:
            fill_value = tensor.max()
    else:
        fill_value = 0
    return fill_value
