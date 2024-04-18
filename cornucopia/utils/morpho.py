import torch
from .conv import convnd
import itertools


def connectivity_kernel(dim, conn=1, **backend):
    """Build a connectivity kernel

    Parameters
    ----------
    dim : int
        Number of spatial dimensions
    conn : int, default=1
        Order of the connectivity
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    kernel : (*spatial) tensor

    """
    kernel = torch.zeros((3,)*dim, **backend)
    for coord in itertools.product([0, 1], repeat=dim):
        if sum(coord) > conn:
            continue
        for sgn in itertools.product([-1, 1], repeat=dim):
            coord1 = [1 + c*s for c, s in zip(coord, sgn)]
            kernel[tuple(coord1)] = 1
    return kernel


# @torch.jit.script
def xor(x, y):
    return (x + y) == 1


def _dist1(x, ix, dist, conn, n_iter):
    dim = conn.dim()
    if dist is not None:
        ox, oix = x, ix
    if x is not None:
        x = convnd(dim, x, conn, padding='same').clamp_max_(1)
    if ix is not None:
        ix = convnd(dim, ix, conn, padding='same').clamp_max_(1)
    if dist is not None:
        # NOTE: this function used to be inlined in the `_morpho` loop,
        # but  the two following lines caused massive memory leaks
        # (of RAM -- not VRAM, even when tensor live in the GPU).
        # I haven't found the cause of it, but moving the loop body in a
        # separate function seems to solve the problem.
        dist = dist.masked_fill_(xor(x, ox), n_iter)
        dist = dist.masked_fill_(xor(ix, oix), -n_iter)
    return x, ix, dist


def _morpho(mode, x, conn, nb_iter, dim):
    """Common worker for binary operations

    Notes
    -----
    .. Adapted from Neurite-Sandbox (author: B Fischl)

    Parameters
    ----------
    mode : {'dilate', 'erode', 'dist'}
    x : (..., *spatial) tensor
    conn : tensor or int
    nb_iter : int
    dim : int

    Returns
    -------
    y : (..., *spatial) tensor

    """
    in_dtype = x.dtype
    if in_dtype is not torch.bool:
        x = x > 0
    x = x.to(torch.float32 if x.is_cuda else torch.uint8)
    backend = dict(dtype=x.dtype, device=x.device)

    dim = dim or x.dim()
    if isinstance(conn, int):
        conn = connectivity_kernel(dim, conn, **backend)
    else:
        conn = conn.to(**backend)

    ix = dist = None
    if mode == 'dist':
        dist = torch.full_like(x, nb_iter+1, dtype=torch.int)
        dist.masked_fill_(x > 0, -(nb_iter+1))
        ix = 1 - x
    if mode == 'erode':
        ix = 1 - x
        x = None

    for n_iter in range(1, nb_iter+1):
        x, ix, dist = _dist1(x, ix, dist, conn, n_iter)

    if mode == 'dilate':
        if x.dtype.is_floating_point:
            x = x.round()
        return x.to(in_dtype)
    if mode == 'erode':
        if ix.dtype.is_floating_point:
            ix = ix.round()
        return ix.neg_().add_(1).to(in_dtype)
    return dist


def _soft_morpho(mode, x, conn, nb_iter, dim):
    """Common worker for soft operations

    Parameters
    ----------
    mode : {'dilate', 'erode'}
    x : (..., *spatial) tensor
    conn : tensor or int
    nb_iter : int
    dim : int

    Returns
    -------
    y : (..., *spatial) tensor

    """
    backend = dict(dtype=x.dtype, device=x.device)

    dim = dim or x.dim()
    if isinstance(conn, int):
        conn = connectivity_kernel(dim, conn, **backend)
    else:
        conn = conn.to(**backend)

    x = x.clone()
    if mode == 'dilate':
        x.neg_().add_(1)
    x = x.clamp_(0.001, 0.999).log_()

    for n_iter in range(1, nb_iter+1):
        x = convnd(dim, x, conn, padding='same')

    x = x.exp_()
    if mode == 'dilate':
        x.neg_().add_(1)
    return x


def erode(x, conn=1, nb_iter=1, dim=None, soft=False):
    """Binary erosion

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor (will be binarized, if not `soft`)
    conn : int or tensor, default=1
        If a tensor, the connectivity kernel to use.
        If an int, the connectivity order
            1 : 4-connectivity (2D) //  6-connectivity (3D)
            2 : 8-connectivity (2D) // 18-connectivity (3D)
            3 :                     // 26-connectivity (3D)
    nb_iter : int, default=1
        Number of iterations
    dim : int, default=`x.dim()`
        Number of spatial dimensions
    soft : bool, default=False
        Assume input are probabilities and use a soft operator.

    Returns
    -------
    y : (..., *spatial) tensor
        Eroded tensor

    """
    fn = _soft_morpho if soft else _morpho
    return fn('erode', x, conn, nb_iter, dim)


def dilate(x, conn=1, nb_iter=1, dim=None, soft=False):
    """Binary dilation

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor (will be binarized if not `soft`)
    conn : int or tensor, default=1
        If a tensor, the connectivity kernel to use.
        If an int, the connectivity order
            1 : 4-connectivity (2D) //  6-connectivity (3D)
            2 : 8-connectivity (2D) // 18-connectivity (3D)
            3 :                     // 26-connectivity (3D)
    nb_iter : int, default=1
        Number of iterations
    dim : int, default=`x.dim()`
        Number of spatial dimensions
    soft : bool, default=False
        Assume input are probabilities and use a soft operator.

    Returns
    -------
    y : (..., *spatial) tensor
        Dilated tensor

    """
    fn = _soft_morpho if soft else _morpho
    return fn('dilate', x, conn, nb_iter, dim)


def bounded_distance(x, conn=1, nb_iter=1, dim=None):
    """Bounded signed (city block) distance to a binary object.

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor (will be binarized)
    conn : int or tensor, default=1
        If a tensor, the connectivity kernel to use.
        If an int, the connectivity order
            1 : 4-connectivity (2D) //  6-connectivity (3D)
            2 : 8-connectivity (2D) // 18-connectivity (3D)
            3 :                     // 26-connectivity (3D)
    nb_iter : int, default=1
        Number of iterations. All voxels farther from the object than
        `nb_iter` will be given the distance `nb_iter + 1`
    dim : int, default=`x.dim()`
        Number of spatial dimensions

    Returns
    -------
    y : (..., *spatial) tensor
        Dilated tensor

    """
    return _morpho('dist', x, conn, nb_iter, dim)
