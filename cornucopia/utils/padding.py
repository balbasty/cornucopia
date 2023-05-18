import torch
from .py import prod
from .jit import sub2ind_list, meshgrid_list_ij
from . import bounds


def ensure_shape(inp, shape, mode='constant', value=0, side='post'):
    """Pad/crop a tensor so that it has a given shape

    Parameters
    ----------
    inp : tensor
        Input tensor
    shape : sequence
        Output shape
    mode : 'constant', 'replicate', 'reflect1', 'reflect2', 'circular'
        default='constant'
    value : scalar, default=0
        Value for mode 'constant'
    side : {'pre', 'post', 'both'}, default='post'
        Side to crop/pad

    Returns
    -------
    out : tensor
        Padded tensor with shape `shape`

    """
    shape = list(shape)
    shape = shape + [1] * max(0, inp.dim() - len(shape))
    if inp.dim() < len(shape):
        inp = inp.reshape(inp.shape + (1,) * max(0, len(shape) - inp.dim()))
    inshape = inp.shape
    shape = [inshape[d] if shape[d] is None else shape[d]
             for d in range(len(shape))]
    ndim = len(shape)

    # crop
    if side == 'both':
        crop = [max(0, inshape[d] - shape[d]) for d in range(ndim)]
        index = tuple(slice(c//2, (c//2 - c) or None) for c in crop)
    elif side == 'pre':
        crop = [max(0, inshape[d] - shape[d]) for d in range(ndim)]
        index = tuple(slice(-c or None) for c in crop)
    else:  # side == 'post'
        index = tuple(slice(min(shape[d], inshape[d])) for d in range(ndim))
    inp = inp[index]

    # pad
    pad_size = [max(0, shape[d] - inshape[d]) for d in range(ndim)]
    if side == 'both':
        pad_size = [[p//2, p-p//2] for p in pad_size]
        pad_size = [q for p in pad_size for q in p]
        side = None
    inp = pad(inp, tuple(pad_size), mode=mode, value=value, side=side)

    return inp


def pad(inp, padsize, mode='constant', value=0, side=None):
    """Pad a tensor.

    This function is a bit more generic than torch's native pad, but probably
    a bit slower:
        - works with any input type
        - works with arbitrarily large padding size
        - crops the tensor for negative padding values
        - implements additional padding modes
    When used with defaults parameters (side=None), it behaves
    exactly like `torch.nn.functional.pad`

    Boundary modes are:
        - 'circular' or 'dft'
        - 'reflect' or 'reflect1' or 'dct1'
        - 'reflect2' or 'dct2'
        - 'replicate'
        - 'constant'

    Side modes are 'pre', 'post', 'both' or None. If side is not None,
    inp.dim() values (or less) should be provided. If side is None,
    twice as many values should be provided, indicating different padding sizes
    for the 'pre' and 'post' sides. If the number of padding values is less
    than the dimension of the input tensor, zeros are prepended.

    Parameters
    ----------
    inp : tensor_like
        Input tensor
    padsize : [sequence of] int
        Amount of padding in each dimension.
    mode : {'constant', 'replicate', 'reflect1', 'reflect2', 'circular'}, default='constant'
        Padding mode
    value : scalar, default=0
        Value to pad with in mode 'constant'.
    side : {'left', 'right', 'both', None}, default=None
        Use padsize to pad on left side ('pre'), right side ('post') or
        both sides ('both'). If None, the padding side for the left and
        right sides should be provided in alternate order.

    Returns
    -------
    tensor
        Padded tensor.

    """
    # Argument checking
    if mode not in bounds.all_bounds:
        raise ValueError(f'Padding mode should be one of {bounds.all_bounds}. '
                         f'Got {mode}.')
    padsize = tuple(padsize)
    if not side:
        if len(padsize) % 2:
            raise ValueError('Padding length must be divisible by 2')
        padpre = padsize[::2]
        padpost = padsize[1::2]
    else:
        side = side.lower()
        if side == 'both':
            padpre = padsize
            padpost = padsize
        elif side in ('pre', 'left'):
            padpre = padsize
            padpost = (0,) * len(padpre)
        elif side in ('post', 'right'):
            padpost = padsize
            padpre = (0,) * len(padpost)
        else:
            raise ValueError(f'Unknown side `{side}`')
    padpre = (0,) * max(0, inp.dim()-len(padpre)) + padpre
    padpost = (0,) * max(0, inp.dim()-len(padpost)) + padpost
    if inp.dim() != len(padpre) or inp.dim() != len(padpost):
        raise ValueError('Padding length too large')

    # Pad
    mode = bounds.to_nitorch(mode)
    if mode == 'zero':
        return _pad_constant(inp, padpre, padpost, value)
    else:
        bound = getattr(bounds, mode)
        return _pad_bound(inp, padpre, padpost, bound)


def _pad_constant(inp, padpre, padpost, value):
    new_shape = [s + pre + post
                 for s, pre, post in zip(inp.shape, padpre, padpost)]
    out = inp.new_full(new_shape, value)
    slicer = [slice(pre, pre + s) for pre, s in zip(padpre, inp.shape)]
    out[tuple(slicer)] = inp
    return out


def _pad_bound(inp, padpre, padpost, bound):
    begin = list(map(lambda x: -x, padpre))
    end = tuple(d+p for d, p in zip(inp.shape, padpost))

    grid = [torch.arange(b, e, device=inp.device) for (b, e) in zip(begin, end)]
    mult = [None] * inp.dim()
    for d, n in enumerate(inp.shape):
        grid[d], mult[d] = bound(grid[d], n)
    grid = list(meshgrid_list_ij(grid))
    if any(map(torch.is_tensor, mult)):
        for d in range(len(mult)):
            if not torch.is_tensor(mult[d]):
                continue
            for _ in range(d+1, len(mult)):
                mult[d].unsqueeze_(-1)
    mult = prod(mult)
    grid = sub2ind_list(grid, inp.shape)

    out = inp.flatten()[grid]
    if torch.is_tensor(mult) or mult != 1:
        out *= mult
    return out
