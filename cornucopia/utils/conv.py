import torch
from torch.nn import functional as F
from .py import ensure_list
from .padding import pad
from .kernels import smoothing_kernel


def convnd(ndim, tensor, kernel, bias=None, stride=1, padding=0, bound='zero',
           dilation=1, groups=1):
    """Perform a convolution

    Parameters
    ----------
    ndim : {1, 2, 3}
        Number of spatial dimensions
    tensor : (*batch, [channel_in,] *spatial_in) tensor
        Input tensor
    kernel : ([channel_in, channel_out,] *kernel_size) tensor
        Convolution kernel
    bias : ([channel_out,]) tensor, optional
        Bias tensor
    stride : int or sequence[int], default=1
        Strides between output elements,
    padding : 'same' or int or sequence[int], default=0
        Padding performed before the convolution.
        If 'same', the padding is chosen such that the shape of the
        output tensor is `spatial_in // stride`.
    bound : str, default='zero'
        Boundary conditions used in the padding.
    dilation : int or sequence[int], default=1
        Dilation of the kernel.
    groups : int, default=1

    Returns
    -------
    convolved : (*batch, [channel_out], *spatial_out)

    """
    # move everything to the same dtype/device
    kernel = kernel.to(tensor)
    if bias is not None:
        bias = bias.to(tensor)

    # sanity checks + reshape for torch's conv
    if kernel.dim() not in (ndim, ndim + 2):
        raise ValueError('Kernel shape should be (*kernel_size) or '
                         '(channel_in, channel_out, *kernel_size) but '
                         'got {}'.format(kernel.shape))
    has_channels = kernel.dim() == ndim + 2
    channels_in = kernel.shape[0] if has_channels else 1
    channels_out = kernel.shape[1] if has_channels else 1
    kernel_size = kernel.shape[(2*has_channels):]
    kernel = kernel.reshape([channels_in, channels_out, *kernel_size])
    batch = tensor.shape[:-(ndim+has_channels)]
    spatial_in = tensor.shape[(-ndim):]
    if has_channels and tensor.shape[-(ndim+has_channels)] != channels_in:
        raise ValueError('Number of input channels not consistent: '
                         'Got {} (kernel) and {} (tensor).' .format(
                         channels_in, tensor.shape[-(ndim+has_channels)]))
    tensor = tensor.reshape([-1, channels_in, *spatial_in])
    if bias:
        bias = bias.flatten()
        if bias.numel() == 1:
            bias = bias.expand(channels_out)
        elif bias.numel() != channels_out:
            raise ValueError('Number of output channels not consistent: '
                             'Got {} (kernel) and {} (bias).' .format(
                             channels_out, bias.numel()))

    # Perform padding
    dilation = ensure_list(dilation, ndim)
    padding = ensure_list(padding, ndim)
    padding = [0 if p == 'valid' else 'same' if p == 'auto' else p
               for p in padding]
    for i in range(ndim):
        if isinstance(padding[i], str):
            assert padding[i].lower() == 'same'
            if kernel_size[i] % 2 == 0:
                raise ValueError('Cannot compute "same" padding '
                                 'for even-sized kernels.')
            padding[i] = dilation[i] * (kernel_size[i] // 2)
    if bound != 'zero' and sum(padding) > 0:
        tensor = pad(tensor, padding, bound, side='both')
        padding = 0

    conv_fn = (F.conv1d if ndim == 1 else
               F.conv2d if ndim == 2 else
               F.conv3d if ndim == 3 else None)
    if not conv_fn:
        raise NotImplementedError('Convolution is only implemented in '
                                  'dimension 1, 2 or 3.')
    tensor = conv_fn(tensor, kernel, bias, stride=stride, padding=padding,
                     dilation=dilation, groups=groups)
    spatial_out = tensor.shape[(-ndim):]
    channels_out = [channels_out] if has_channels else []
    tensor = tensor.reshape([*batch, *channels_out, *spatial_out])
    return tensor


def conv1d(input, kernel, dim=-1,  padding='same',
           bound='dct2', stride=1, dilation=1):
    """Perform a 1d convolution along a given dimension

    Parameters
    ----------
    input : tensor
        Input tensor
    kernel : (length,) tensor
        1D kernel
    dim : int, default=-1
        Dimension to convolve
    padding : 'same' or int, default='same'
        Padding performed before the convolution.
        If 'same', the padding is chosen such that the shape of the
        output tensor is `spatial_in // stride`.
    bound : str, default='dct2'
        Boundary conditions used in the padding.
    stride : int, default=1
        Strides between output elements,
    dilation : int or sequence[int], default=1
        Dilation of the kernel.

    Returns
    -------
    convolved : tensor

    """
    # move everything to the same dtype/device
    kernel = kernel.to(input)

    # sanity checks + reshape for torch's conv
    if kernel.dim() > 1:
        raise ValueError(f'Kernel should be a vector but got {kernel.shape}')

    # make sure input shape is [B, C, X, Y, Z]
    indim = input.dim()
    input = input.movedim(dim, -1)
    while input.dim() < 5:
        input = input.unsqueeze(max(-input.dim()-1, -4))
    batch = input.shape[:1]
    if input.dim() > 5:
        batch = input.shape[:-4]
        input = input.reshape([-1, *input.shape[-4:]])

    # Perform padding
    if padding == 'valid':
        padding = 0
    elif padding == 'same':
        if len(kernel) % 2 == 0:
            raise ValueError('Cannot compute "same" padding '
                             'for even-sized kernels.')
        padding = dilation * (len(kernel) // 2)
    if bound != 'zero' and padding > 0:
        input = pad(input, [padding], bound, side='both')
        padding = 0

    # reshape kernel
    C = input.shape[1]
    kernel = kernel.reshape([1, 1, 1, 1, -1])
    kernel = kernel.expand([C, 1, 1, 1, -1])

    # convolve
    output = F.conv3d(input, kernel, None, stride=stride, padding=padding,
                      dilation=dilation, groups=C)

    # reshape
    output = output.reshape([*batch, *output.shape[1:]])
    while output.dim() > indim:
        output = output.squeeze(max(-output.dim(), -4))
    output = output.movedim(-1, dim)
    return output


def smooth1d(input, type='gauss', fwhm=1, basis=1, dim=-1, bound='dct2',
             padding='same', stride=1):
    """Smooth a tensor along a given dimension

    Parameters
    ----------
    input : (..., *spatial) tensor
        Input tensor. Its last `dim` dimensions
        will be smoothed.
    type : {'gauss', 'tri', 'rect'}, default='gauss'
        Smoothing function:
        - 'gauss' : Gaussian
        - 'tri'   : Triangular
        - 'rect'  : Rectangular
    fwhm : float, default=1
        Full-Width at Half-Maximum of the smoothing function.
    basis : {0, 1}, default=1
        Encoding basis.
        The smoothing kernel is the convolution of a smoothing
        function and an encoding basis. This ensures that the
        resulting kernel integrates to one.
    dim : int, default=-1
        Dimension to convolve
    bound : str, default='dct2'
        Boundary condition.
    padding : [sequence of] int or 'same', default='same'
        Amount of padding applied to the input volume.
        'same' ensures that the output dimensions are the same as the
        input dimensions.
    stride : [sequence of] int, default=1
        Stride between output elements.

    Returns
    -------
    output : (..., *spatial) tensor
        The resulting tensor has the same shape as the input tensor.
        This differs from the behaviour of torch's `conv*d`.
    """
    backend = dict(dtype=input.dtype, device=input.device)
    kernel = smoothing_kernel(type, fwhm, basis, **backend)
    return conv1d(input, kernel, dim, padding, bound, stride)


def smoothnd(input, type='gauss', fwhm=1, basis=1, bound='dct2',
             padding='same', stride=1, kernel=None):
    """Smooth a tensor along the last n dimensions

    Parameters
    ----------
    input : (..., *spatial) tensor
        Input tensor. Its last `dim` dimensions
        will be smoothed.
    type : {'gauss', 'tri', 'rect'}, default='gauss'
        Smoothing function:
        - 'gauss' : Gaussian
        - 'tri'   : Triangular
        - 'rect'  : Rectangular
    fwhm : float or sequence[float], default=1
        Full-Width at Half-Maximum of the smoothing function.
    basis : {0, 1}, default=1
        Encoding basis.
        The smoothing kernel is the convolution of a smoothing
        function and an encoding basis. This ensures that the
        resulting kernel integrates to one.
    bound : str, default='dct2'
        Boundary condition.
    padding : [sequence of] int or 'same', default='same'
        Amount of padding applied to the input volume.
        'same' ensures that the output dimensions are the same as the
        input dimensions.
    stride : [sequence of] int, default=1
        Stride between output elements.

    Returns
    -------
    output : (..., *spatial) tensor
        The resulting tensor has the same shape as the input tensor.
        This differs from the behaviour of torch's `conv*d`.
    """
    backend = dict(dtype=input.dtype, device=input.device)
    fwhm = ensure_list(fwhm)
    if kernel is None or len(kernel) == 0:
        kernel = smoothing_kernel(type, fwhm, basis, **backend)

    ndim = len(fwhm)
    stride = ensure_list(stride, ndim)
    padding = ensure_list(padding, ndim)
    bound = ensure_list(bound, ndim)
    kernel = ensure_list(kernel, ndim)

    for d in range(ndim):
        input = conv1d(input, kernel[d].flatten(), dim=-(ndim-d),
                       padding=padding[d], bound=bound[d], stride=stride[d])
    return input


def smooth2d(input, type='gauss', fwhm=1, *args, **kwargs):
    fwhm = ensure_list(fwhm, 2)[:2]
    return smoothnd(input, type, fwhm, *args, **kwargs)


def smooth3d(input, type='gauss', fwhm=1, *args, **kwargs):
    fwhm = ensure_list(fwhm, 3)[:3]
    return smoothnd(input, type, fwhm, *args, **kwargs)
