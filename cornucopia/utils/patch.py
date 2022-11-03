import torch
import itertools
from .py import ensure_list, movedims, ind2sub
from .padding import pad as pad_fn, ensure_shape
from ..intensity import QuantileTransform


_minvalue = {
    torch.bool: False,
    torch.uint8: 0,
    torch.int8: -128,
    torch.int16: -32768,
    torch.int32: -2147483648,
    torch.int64: -9223372036854775808,
    torch.half: -float('inf'),
    torch.float: -float('inf'),
    torch.double: -float('inf'),
}

_maxvalue = {
    torch.bool: True,
    torch.uint8: 255,
    torch.int8: 127,
    torch.int16: 32767,
    torch.int32: 2147483647,
    torch.int64: 9223372036854775807,
    torch.half: float('inf'),
    torch.float: float('inf'),
    torch.double: float('inf'),
}


def unfold(inp, kernel_size, stride=None, collapse=False):
    """Extract patches from a tensor.

    Parameters
    ----------
    inp : (..., *spatial) tensor
        Input tensor.
    kernel_size : [sequence of] int
        Patch shape.
    stride : [sequence of] int, default=`kernel_size`
        Stride.
    collapse : bool or 'view', default=False
        Collapse the original spatial dimensions.
        If 'view', forces collapsing to use the view mechanism, which ensures
        that no data copy is triggered. This can fail if the tensor's
        strides do not allow these dimensions to be collapsed.

    Returns
    -------
    out : (..., *spatial_out, *kernel_size) tensor
        Output tensor of patches.
        If `collapse`, the output spatial dimensions (`spatial_out`)
        are flattened.

    """
    kernel_size = ensure_list(kernel_size)
    dim = len(kernel_size)
    batch_dim = inp.dim() - dim
    stride = ensure_list(stride, dim)
    stride = [st or sz for st, sz in zip(stride, kernel_size)]
    for d, (sz, st) in enumerate(zip(kernel_size, stride)):
        inp = inp.unfold(dimension=batch_dim+d, size=sz, step=st)
    if collapse:
        batch_shape = inp.shape[:-dim*2]
        if collapse == 'view':
            inp = inp.view([*batch_shape, -1, *kernel_size])
        else:
            inp = inp.reshape([*batch_shape, -1, *kernel_size])
    return inp


def fold(inp, dim=None, stride=None, shape=None, collapsed=False,
         reduction='mean'):
    """Reconstruct a tensor from patches.

    .. warning: This function only works if `kernel_size <= 2*stride`.

    Parameters
    ----------
    inp : (..., *spatial, *kernel_size) tensor
        Input tensor of patches
    dim : int
        Length of `kernel_size`.
    stride : [sequence of] int, default=`kernel_size`
        Stride.
    shape : sequence of int, optional
        Output shape. By default, it is computed from `spatial`,
        `stride` and `kernel_size`. If the output shape is larger than
        the computed shape, zero-padding is used.
        This parameter is mandatory if `collapsed = True`.
    collapsed : 'view' or bool, default=False
        Whether the spatial dimensions are collapsed in the input tensor.
        If 'view', use `view` instead of `reshape`, which will raise an
        error instead of triggering a copy when dimensions cannot be
        collapsed in a contiguous way.
    reduction : {'mean', 'sum', 'min', 'max'}, default='mean'
        Method to use to merge overlapping patches.

    Returns
    -------
    out : (..., *shape) tensor
        Folded tensor

    """
    def recon(x, stride):
        dim = len(stride)
        inshape = x.shape[-2*dim:-dim]
        batch_shape = x.shape[:-2*dim]
        indim = list(reversed(range(-1, -2 * dim - 1, -1)))
        outdim = (list(reversed(range(-2, -2 * dim - 1, -2))) +
                  list(reversed(range(-1, -2 * dim - 1, -2))))
        x = movedims(x, indim, outdim)
        outshape = [i * k for i, k in zip(inshape, stride)]
        x = x.reshape([*batch_shape, *outshape])
        return x

    if torch.is_tensor(shape):
        shape = shape.tolist()
    dim = dim or (len(shape) if shape else None)
    if not dim:
        raise ValueError('Cannot guess dim from inputs')
    kernel_size = inp.shape[-dim:]
    stride = ensure_list(stride, len(kernel_size))
    stride = [st or sz for st, sz in zip(stride, kernel_size)]
    if any(sz > 2*st for st, sz in zip(stride, kernel_size)):
        # I only support overlapping of two patches (along a given dim).
        # If the kernel  is too large, more than two patches can overlap
        # and this function fails.
        raise ValueError('This function only works if kernel_size <= 2*stride')
    if not shape:
        if collapsed:
            raise ValueError('`shape` is mandatory when `collapsed=True`')
        inshape = inp.shape[-dim*2:-dim]
        shape = [(i-1)*st + sz
                 for i, st, sz in zip(inshape, stride, kernel_size)]
    else:
        inshape = [(o - sz) // st + 1
                   for o, st, sz in zip(shape, stride, kernel_size)]

    if collapsed:
        batch_shape = inp.shape[:-dim-1]
        inp = inp.reshape([*batch_shape, *inshape, *kernel_size])
    batch_shape = inp.shape[:-2*dim]

    # When the stride is equal to the kernel size, folding is easy
    # (it is obtained by shuffling dimensions and reshaping)
    # However, in the more general case, patches can overlap or,
    # conversely, have gaps between them. In the first case,
    # overlapping values must be reduced somehow. In the second case,
    # patches must be padded.

    # 1) padding (stride > kernel_size)
    padding = [max(0, st - sz) for st, sz in zip(stride, kernel_size)]
    padding = [0] * (inp.dim() - dim) + padding
    inp = pad_fn(inp, padding, side='right')
    stride = [(st if st < sz else sz) for st, sz in zip(stride, kernel_size)]
    kernel_size = inp.shape[-dim:]

    # 2) merge overlaps
    overlap = [max(0, sz - st) for st, sz in zip(stride, kernel_size)]
    if any(o != 0 for o in overlap):
        slicer = [slice(None)] * (inp.dim() - dim)
        slicer += [slice(k) for k in stride]
        out = inp[tuple(slicer)].clone()
        if reduction == 'mean':
            count = inp.new_ones([*inshape, *stride], dtype=torch.int)
            fn = 'sum'
        else:
            count = None
            fn = reduction

        # ! a bit of padding to save the last values
        padding = [1 if o else 0 for o in overlap] + [0] * dim
        if count is not None:
            count = pad_fn(count, padding, side='right')
        padding = [0] * (out.dim() - 2*dim) + padding
        value = (_minvalue[inp.dtype] if fn == 'max' else
                 _maxvalue[inp.dtype] if fn == 'min' else 0)
        out = pad_fn(out, padding, value=value, side='right')

        slicer1 = [slice(-1 if o else None) for o in overlap]
        slicer2 = [slice(None)] * dim
        slicer1 += [slice(st) for st in stride]
        slicer2 += [slice(st) for st in stride]

        overlaps = itertools.product(*[[0, 1] if o else [0] for o in overlap])
        for overlap in overlaps:
            front_slicer = list(slicer1)
            back_slicer = list(slicer2)
            for d, o in enumerate(overlap):
                if o == 0:
                    continue
                front_slicer[-dim+d] = slice(o)
                front_slicer[-2*dim+d] = slice(1, None)
                back_slicer[-dim+d] = slice(-o, None)
                back_slicer[-2*dim+d] = slice(None)
            if count is not None:
                count[tuple(front_slicer)] += 1
            front_slicer = (Ellipsis, *front_slicer)
            back_slicer = (Ellipsis, *back_slicer)

            if fn == 'sum':
                out[front_slicer] += inp[back_slicer]
            elif fn == 'max':
                out[front_slicer] = torch.max(out[front_slicer], inp[back_slicer])
            elif fn == 'min':
                out[front_slicer] = torch.min(out[front_slicer], inp[back_slicer])
            else:
                raise ValueError(f'Unknown reduction {reduction}')
        if count is not None:
            out /= count
    else:
        out = inp.clone()

    # end) reshape
    out = recon(out, stride)
    out = ensure_shape(out, [*batch_shape, *shape], side='right')

    return out


def patch_apply(net, inp, patch, pad=None, preproc=QuantileTransform(),
                preproc_patchwise=True):
    """
    Apply a network to a large image patch-wise

    Parameters
    ----------
    net : nn.Module
        Network to apply.
        Should take a [B, C, *spatial] tensor as input and return
        a [B, K, *spatial] tensor as output.
    inp : (C, *spatial) tensor
        Input imag,e without a batch dimensions
    patch : int or list[int]
        Patch size
    pad : int or list[int], default=patch//2
        Apply network to patches of size `patch + 2*pad`, but only use
        the `patch` center from the output.
    preproc : transform
        Transform
    preproc_patchwise : bool
        Apply the preprocessing transform to each patch individually

    Returns
    -------
    out : (K, *spatial) tensor
        Output

    """
    ndim = inp.dim() - 1
    patch = ensure_list(patch, ndim)
    pad = ensure_list(pad, ndim)
    pad = [s//2 if p is None else p for s, p in zip(patch, pad)]
    fullpatch = [s+2*p for s, p in zip(patch, pad)]
    inshape = inp.shape[1:]

    if preproc and not preproc_patchwise:
        inp = preproc(inp)

    patchslicer = None
    if any(pad):
        inp = pad_fn(inp, [0] + pad, mode='dct2', side='both')
        patchslicer = tuple(slice(p, -p) if p else slice(None) for p in pad)
        patchslicer = (slice(None), *patchslicer)

    foldinp = unfold(inp, fullpatch, patch)
    out = None

    backend = dict(dtype=list(net.parameters())[0].dtype,
                   device=list(net.parameters())[0].device)

    nloop = foldinp.shape[-2*ndim:-ndim].numel()
    for n in range(nloop):

        # extract input patch
        index = ind2sub(n, foldinp.shape[-2*ndim:-ndim])
        slicer = (slice(None), *index, Ellipsis)
        patchinp = foldinp[slicer].to(**backend)

        # apply preprocessing if needed
        if preproc and preproc_patchwise:
            patchinp = preproc(patchinp)

        # apply network and get center of output patch
        patchout = net(patchinp[None])[0].to(inp.device)
        if patchslicer:
            patchout = patchout[patchslicer]

        # allocate full output tensor
        if out is None:
            out = torch.zeros([len(patchout), *inshape],
                              dtype=patchout.dtype, device=inp.device)

        # assign output patch
        slicer = tuple(slice(i*p, (i+1)*p) for i, p in zip(index, patch))
        slicer = (slice(None), *slicer)
        out[slicer] = patchout

    return out
