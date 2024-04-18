# NOTE: copied from https://github.com/balbasty/nitorch
#
# TODO:
# [ ] Implement Sinc kernel
import torch


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


def _integrate_poly(l, h, *args):
    """Integrate a polynomial on an interval.

    k = _integrate_poly(l, h, a, b, c, ...)
    integrates the polynomial a+b*x+c*x^2+... on [l,h]

    All inputs should be `torch.Tensor`
    """
    # NOTE: operations are not performed inplace (+=, *=) so that autograd
    # can backpropagate.
    # TODO: (maybe) use inplace if gradients not required
    zero = torch.zeros(tuple(), dtype=torch.bool)
    k = torch.zeros(l.shape, dtype=l.dtype, device=l.device)
    hh = h
    ll = l
    for i in range(len(args)):
        if torch.any(args[i] != zero):
            k = k + (args[i]/(i+1))*(hh-ll)
        hh = hh * h
        ll = ll * l
    return k


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


def _gauss1d0(w, x):
    logtwo = torch.tensor(2., dtype=w.dtype, device=w.device).log()
    sqrttwo = torch.tensor(2., dtype=w.dtype, device=w.device).sqrt()
    s = w/(8.*logtwo).sqrt() + 1E-7  # standard deviation
    if x is None:
        lim = torch.floor(4*s+0.5).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    w1 = 1./(sqrttwo*s)
    ker = 0.5*((w1*(x+0.5)).erf() - (w1*(x-0.5)).erf())
    ker = ker.clamp(min=0)
    return ker, x


def _gauss1d1(w, x):
    import math
    logtwo = torch.tensor(2., dtype=w.dtype, device=w.device).log()
    sqrttwo = torch.tensor(2., dtype=w.dtype, device=w.device).sqrt()
    sqrtpi = torch.tensor(math.pi, dtype=w.dtype, device=w.device).sqrt()
    s = w/(8.*logtwo).sqrt() + 1E-7  # standard deviation
    if x is None:
        lim = torch.floor(4*s+1).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    w1 = 0.5*sqrttwo/s
    w2 = -0.5/s.pow(2)
    w3 = s/(sqrttwo*sqrtpi)
    ker = 0.5*((w1*(x+1)).erf()*(x+1)
               + (w1*(x-1)).erf()*(x-1)
               - 2*(w1*x).erf()*x) \
        + w3*((w2*(x+1).pow(2)).exp()
              + (w2*(x-1).pow(2)).exp()
              - 2*(w2*x.pow(2)).exp())
    ker = ker.clamp(min=0)
    return ker, x


def _rect1d0(w, x):
    if x is None:
        lim = torch.floor((w+1)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    ker = torch.max(torch.min(x+0.5, w/2) - torch.max(x-0.5, -w/2), zero)
    ker = ker/w
    return ker, x


def _rect1d1(w, x):
    if x is None:
        lim = torch.floor((w+2)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    one = torch.ones(tuple(), dtype=w.dtype, device=w.device)
    neg_low = torch.min(torch.max(x-w/2, -one),   zero)
    neg_upp = torch.max(torch.min(x+w/2,  zero), -one)
    pos_low = torch.min(torch.max(x-w/2,  zero),  one)
    pos_upp = torch.max(torch.min(x+w/2,  one),   zero)
    ker = _integrate_poly(neg_low, neg_upp, one,  one) \
        + _integrate_poly(pos_low, pos_upp, one, -one)
    ker = ker/w
    return ker, x


def _triangle1d0(w, x):
    if x is None:
        lim = torch.floor((2*w+1)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    one = torch.ones(tuple(), dtype=w.dtype, device=w.device)
    neg_low = torch.min(torch.max(x-0.5, -w),     zero)
    neg_upp = torch.max(torch.min(x+0.5,  zero), -w)
    pos_low = torch.min(torch.max(x-0.5,  zero),  w)
    pos_upp = torch.max(torch.min(x+0.5,  w),     zero)
    ker = _integrate_poly(neg_low, neg_upp, one,  1/w) \
        + _integrate_poly(pos_low, pos_upp, one, -1/w)
    ker = ker/w
    return ker, x


def _triangle1d1(w, x):
    if x is None:
        lim = torch.floor((2*w+2)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    one = torch.ones(tuple(), dtype=w.dtype, device=w.device)
    neg_neg_low = torch.min(torch.max(x,   -one),   zero)
    neg_neg_upp = torch.max(torch.min(x+w,  zero), -one)
    neg_pos_low = torch.min(torch.max(x,    zero),  one)
    neg_pos_upp = torch.max(torch.min(x+w,  one),   zero)
    pos_neg_low = torch.min(torch.max(x-w, -one),   zero)
    pos_neg_upp = torch.max(torch.min(x,    zero), -one)
    pos_pos_low = torch.min(torch.max(x-w,  zero),  one)
    pos_pos_upp = torch.max(torch.min(x,    one),   zero)
    ker = _integrate_poly(neg_neg_low, neg_neg_upp, 1+x/w,  1+x/w-1/w, -1/w) \
        + _integrate_poly(neg_pos_low, neg_pos_upp, 1+x/w, -1-x/w-1/w,  1/w) \
        + _integrate_poly(pos_neg_low, pos_neg_upp, 1-x/w,  1-x/w+1/w,  1/w) \
        + _integrate_poly(pos_pos_low, pos_pos_upp, 1-x/w, -1+x/w+1/w, -1/w)
    ker = ker/w
    return ker, x


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
    x = x + (x[-1],)*max(0, nker-len(x))
    types += (types[-1],)*max(0, nker-len(types))

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

