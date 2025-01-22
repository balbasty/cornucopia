# NOTE: copied from https://github.com/balbasty/nitorch
#
# TODO:
# [ ] Implement Sinc kernel
import torch
import math


def make_separable(ker, channels):
    """Transform a single-channel kernel into a multi-channel separable kernel.

    Args:
        ker (torch.tensor): Single-channel kernel (1, 1, D, H, W).
        channels (int): Number of input/output channels.

    Returns:
        ker (torch.tensor): Multi-channel group kernel (1, 1, D, H, W).

    """
    ndim = ker.ndim
    repetitions = (channels,) + (1,)*(ndim-1)
    ker = ker.repeat(repetitions)
    return ker


def _integrate_poly(L, H, *args):
    """Integrate a polynomial on an interval.

    k = _integrate_poly(l, h, a, b, c, ...)
    integrates the polynomial a+b*x+c*x^2+... on [l,h]

    All inputs should be `torch.Tensor`
    """
    # NOTE: operations are not performed inplace (+=, *=) so that autograd
    # can backpropagate.
    # TODO: (maybe) use inplace if gradients not required
    K = 0
    HH = H
    LL = L
    for i in range(len(args)):
        if torch.any(args[i] != 0):
            K += (args[i] / (i+1)) * (HH - LL)
        HH = HH * H
        LL = LL * L
    return K


def _dirac1d(fwhm, basis, x):
    if x is None:
        x = torch.ones(1, dtype=fwhm.dtype, device=fwhm.device)
    return (x == 1).to(x.dtype), x


def _gauss1d(fwhm, basis, x):
    if basis:
        return _gauss1d1(fwhm, x)
    else:
        return _gauss1d0(fwhm, x)


def _rect1d(fwhm, basis, x):
    if basis:
        return _rect1d1(fwhm, x)
    else:
        return _rect1d0(fwhm, x)


def _triangle1d(fwhm, basis, x):
    if basis:
        return _triangle1d1(fwhm, x)
    else:
        return _triangle1d0(fwhm, x)


def _gauss1d0(W, X):
    LOG2 = math.log(2)
    SQRT2 = math.sqrt(2)
    S = W / math.sqrt(8 * LOG2) + 1E-7  # standard deviation
    if X is None:
        L = torch.floor(4*S + 0.5).int()
        X = torch.arange(-L, L+1).to(W)
    W1 = 1. / (SQRT2*S)
    K = 0.5 * (
        (W1 * (X + 0.5)).erf() -
        (W1 * (X - 0.5)).erf()
    )
    K = K.clamp_min_(0)
    return K, X


def _gauss1d1(W, X):
    LOG2 = math.log(2)
    SQRT2 = math.sqrt(2)
    SQRTPI = math.sqrt(math.pi)
    S = W / math.sqrt(8*LOG2) + 1E-7  # standard deviation
    if X is None:
        L = torch.floor(4*S + 1).int()
        X = torch.arange(-L, L+1).to(W)
    W1 = 0.5 * SQRT2 / S
    W2 = -0.5 / (S * S)
    W3 = S / (SQRT2 * SQRTPI)
    K = (
        0.5 * (
            (W1 * (X + 1)).erf() * (X + 1) +
            (W1 * (X - 1)).erf() * (X - 1) -
            (W1 * X).erf() * X * 2
        ) +
        W3 * (
            (W2 * (X + 1).square()).exp() +
            (W2 * (X - 1).square()).exp() -
            (W2 * X.square()).exp() * 2
        )
    )
    K = K.clamp_min_(0)
    return K, X


def _rect1d0(W, X):
    if X is None:
        L = torch.floor((W+1)/2).int()
        X = torch.arange(-L, L+1).to(W)
    K = (
        torch.min(X + 0.5,  W / 2) -
        torch.max(X - 0.5, -W / 2)
    ).clamp_min_(0)
    K = K / W
    return K, X


def _rect1d1(W, X):
    if X is None:
        L = torch.floor((W+2)/2).int()
        X = torch.arange(-L, L+1).to(W)
    neg_low = torch.clamp(X - W/2, -1, 0)
    neg_upp = torch.clamp(X + W/2, -1, 0)
    pos_low = torch.clamp(X - W/2,  0, 1)
    pos_upp = torch.clamp(X + W/2,  0, 1)
    K = (
        _integrate_poly(neg_low, neg_upp, 1,  1) +
        _integrate_poly(pos_low, pos_upp, 1, -1)
    )
    K = K/W
    return K, X


def _triangle1d0(W, X):
    if X is None:
        L = torch.floor((2*W+1)/2).int()
        X = torch.arange(-L, L+1).to(W)
    neg_low = torch.clamp(X - 0.5, -W, 0)
    neg_upp = torch.clamp(X + 0.5, -W, 0)
    pos_low = torch.clamp(X - 0.5,  0, W)
    pos_upp = torch.clamp(X + 0.5,  0, W)
    K = (
        _integrate_poly(neg_low, neg_upp, 1,  1/W) +
        _integrate_poly(pos_low, pos_upp, 1, -1/W)
    )
    K = K / W
    return K, X


def _triangle1d1(W, X):
    if X is None:
        L = torch.floor((2*W+2)/2).int()
        X = torch.arange(-L, L+1).to(W)
    neg_neg_low = torch.clamp(X,     -1, 0)
    neg_neg_upp = torch.clamp(X + W, -1, 0)
    neg_pos_low = torch.clamp(X,      0, 1)
    neg_pos_upp = torch.clamp(X + W,  0, 1)
    pos_neg_low = torch.clamp(X - W, -1, 0)
    pos_neg_upp = torch.clamp(X,     -1, 0)
    pos_pos_low = torch.clamp(X - W,  0, 1)
    pos_pos_upp = torch.clamp(X,      0, 1)
    K = (
        _integrate_poly(neg_neg_low, neg_neg_upp, 1+X/W,  1+X/W-1/W, -1/W) +
        _integrate_poly(neg_pos_low, neg_pos_upp, 1+X/W, -1-X/W-1/W,  1/W) +
        _integrate_poly(pos_neg_low, pos_neg_upp, 1-X/W,  1-X/W+1/W,  1/W) +
        _integrate_poly(pos_pos_low, pos_pos_upp, 1-X/W, -1+X/W+1/W, -1/W)
    )
    K = K/W
    return K, X


_smooth_switcher = {
    'dirac': _dirac1d,
    'gauss': _gauss1d,
    'rect': _rect1d,
    'triangle': _triangle1d,
    -1: _dirac1d,
    0: _rect1d,
    1: _triangle1d,
    2: _gauss1d,
    }


def smoothing_kernel(types='gauss', fwhm=1, basis=1, x=None, sep=True,
                     dtype=None, device=None):
    """Create a smoothing kernel.

    Creates a (separable) smoothing kernel with fixed (i.e., not learned)
    weights. These weights are obtained by analytically convolving a
    smoothing function (e.g., Gaussian) with a basis function that encodes
    the underlying image (e.g., trilinear).
    Note that `smooth` is fully differentiable with respect to `fwhm`.
    If the kernel is evaluated at all integer coordinates from its support,
    its elements are ensured to sum to one.
    The returned kernel is a `torch.Tensor`.

    The returned kernel is intended for volumes ordered as (B, C, D, H, W).
    However, the fwhm elements should be ordered as (W, H, D).
    For more information about ordering conventions in nitorch, see
    `nitorch.spatial?`.

    Parameters
    ----------
    types : str or int or sequence[str or int]
        Smoothing function (integrates to one).
        - -1 or 'dirac' : Dirac function
        -  0 or 'rect'  : Rectangular function (0th order B-spline)
        -  1 or 'tri'   : Triangular function (1st order B-spline)
        -  2 or 'gauss' : Gaussian
    fwhm : int or sequence[int], default=1
        Full-width at half-maximum of the smoothing function
        (in voxels), in each dimension.
    basis : int, default=1
        Image encoding basis (B-spline order)
    x : tuple or vector_like, optional
        Coordinates at which to evaluate the kernel.
        If None, evaluate at all integer coordinates from its support
        (truncated support for 'gauss').
    sep : bool, default=True
        Return separable 1D kernels.
        If False, the 1D kernels are combined to form an N-D kernel.
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    tuple or (channel_in, channel_out, *kernel_size) tensor
        If `sep is False` or all input parameters are scalar,
        a single kernel is returned.
        Else, a tuple of kernels is returned.


    """
    # Convert to tensors
    fwhm = torch.as_tensor(fwhm, dtype=dtype, device=device).flatten()
    if not fwhm.is_floating_point():
        fwhm = fwhm.float()
    dtype = fwhm.dtype
    device = fwhm.device
    return_tuple = True
    if not isinstance(x, tuple):
        return_tuple = (len(fwhm.shape) > 0)
        x = (x,)
    x = tuple(torch.as_tensor(x1, dtype=dtype, device=device).flatten()
              if x1 is not None else None for x1 in x)
    if type(types) not in (list, tuple):
        types = [types]
    types = list(types)

    # Ensure all sizes are consistant
    nker = max(fwhm.numel(), len(x), len(types))
    fwhm = torch.cat((fwhm, fwhm[-1].repeat(max(0, nker-fwhm.numel()))))
    x = x + (x[-1],) * max(0, nker-len(x))
    types += (types[-1],) * max(0, nker-len(types))

    # Loop over dimensions
    ker = tuple()
    x = list(x)
    for d in range(nker):
        ker1, x[d] = _smooth_switcher[types[d]](fwhm[d], basis, x[d])
        shape = [1, ] * nker
        shape[d] = ker1.numel()
        ker1 = ker1.reshape(shape)
        ker1 = ker1[None, None, ...]  # Cout = 1, Cin = 1
        ker += (ker1, )

    # Make N-D kernel
    if not sep:
        ker1 = ker
        ker = ker1[0]
        for d in range(1, nker):
            ker = ker * ker1[d]
    elif not return_tuple:
        ker = ker[0]

    return ker
