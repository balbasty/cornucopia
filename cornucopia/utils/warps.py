import torch
from torch.nn import functional as F
from .py import ensure_list, cartesian_grid, meshgrid_ij


def add_identity_(flow):
    """Adds the identity grid to a displacement field, inplace.

    Parameters
    ----------
    flow : (..., *shape, dim) tensor
        Displacement field

    Returns
    -------
    flow : (..., *shape, dim) tensor
        Transformation field

    """
    dim = flow.shape[-1]
    spatial = flow.shape[-dim-1:-1]
    grid = cartesian_grid(spatial, dtype=flow.dtype, device=flow.device)
    flow = flow.movedim(-1, 0)
    for i, grid1 in enumerate(grid):
        flow[i].add_(grid1)
    flow = flow.movedim(0, -1)
    return flow


def sub_identity_(flow):
    """Subtracts the identity grid from a transformation field, inplace.

    Parameters
    ----------
    flow : (..., *shape, dim) tensor
        Transformation field

    Returns
    -------
    flow : (..., *shape, dim) tensor
        Displacement field

    """
    dim = flow.shape[-1]
    spatial = flow.shape[-dim-1:-1]
    grid = cartesian_grid(spatial, dtype=flow.dtype, device=flow.device)
    flow = flow.movedim(-1, 0)
    for i, grid1 in enumerate(grid):
        flow[i].sub_(grid1)
    flow = flow.movedim(0, -1)
    return flow


def add_identity(flow):
    """Adds the identity grid to a displacement field.

    Parameters
    ----------
    flow : (..., *shape, dim) tensor
        Displacement field

    Returns
    -------
    flow : (..., *shape, dim) tensor
        Transformation field

    """
    return add_identity_(flow.clone())


def sub_identity(flow):
    """Subtracts the identity grid from a transformation field.

    Parameters
    ----------
    flow : (..., *shape, dim) tensor
        Transformation field

    Returns
    -------
    flow : (..., *shape, dim) tensor
        Displacement field

    """
    return sub_identity_(flow.clone())


def identity(shape, **backend):
    """Returns an identity transformation field.

    Parameters
    ----------
    shape : (dim,) sequence of int
        Spatial dimension of the field.

    Returns
    -------
    grid : (*shape, dim) tensor
        Transformation field

    """
    backend.setdefault('dtype', torch.get_default_dtype())
    return torch.stack(cartesian_grid(shape, **backend), dim=-1)


def affine_flow(affine, shape):
    """Generate an affine flow field

    Parameters
    ----------
    affine : ([B], D+1, D+1) tensor
        Affine matrix
    shape : (D,) list[int]
        Lattice size

    Returns
    -------
    flow : ([B], *shape, D) tensor, Affine flow

    """
    ndim = len(shape)
    backend = dict(dtype=affine.dtype, device=affine.device)

    # add spatial dimensions so that we can use batch matmul
    for _ in range(ndim):
        affine = affine.unsqueeze(-3)
    lin, trl = affine[..., :ndim, :ndim], affine[..., :ndim, -1]

    # create affine transform
    flow = identity(shape, **backend)
    flow = lin.matmul(flow.unsqueeze(-1)).squeeze(-1)
    flow = flow.add_(trl)

    # subtract identity to get a flow
    flow = sub_identity_(flow)

    return flow


def flow_to_torch(flow, shape, align_corners=True, has_identity=False):
    """Convert a voxel displacement field to a torch sampling grid

    Parameters
    ----------
    flow : (..., *shape, D) tensor
        Displacement field
    shape : list[int] tensor
        Spatial shape of the input image
    align_corners : bool, default=True
        Torch's grid mode
    has_identity : bool, default=False
        If False, `flow` is contains relative displacement.
        If False, `flow` contains absolute coordinates.

    Returns
    -------
    grid : (..., *shape, D) tensor
        Sampling grid to be used with torch's `grid_sample`

    """
    backend = dict(dtype=flow.dtype, device=flow.device)
    # 1) reverse last dimension
    flow = torch.flip(flow, [-1])
    # 2) add identity grid
    if not has_identity:
        grid = cartesian_grid(shape, **backend)
        grid = list(reversed(grid))
        for d, g in enumerate(grid):
            flow[..., d].add_(g)
    shape = list(reversed(shape))
    # 3) convert coordinates
    for d, s in enumerate(shape):
        if align_corners:
            # (0, N-1) -> (-1, 1)
            flow[..., d].mul_(2/(s-1)).add_(-1)
        else:
            # (-0.5, N-0.5) -> (-1, 1)
            flow[..., d].mul_(2/s).add_(1/s-1)
    return flow


def apply_flow(image, flow, has_identity=False, **kwargs):
    """Warp an image according to a (voxel) displacement field.

    Parameters
    ----------
    image : (B, C, *shape_in) tensor
        Input image.
        If input dtype is integer, assumes labels: each unique labels
        gets warped using linear interpolation, and the label map gets
        reconstructed by argmax.
    flow : ([B], *shape_out, D) tensor
        Displacement field, in voxels.
        Note that the order of the last dimension is inverse of what's
        usually expected in torch's grid_sample.
    has_identity : bool, default=False
        - If False, `flow` is contains relative displacement.
        - If True, `flow` contains absolute coordinates.

    Returns
    -------
    warped : (B, C, *shape_out) tensor
        Warped image

    """
    kwargs.setdefault('align_corners', True)
    B, C, *shape_in = image.shape
    D = flow.shape[-1]
    if flow.dim() == D+1:
        flow = flow[None]
    shape_out = flow.shape[1:-1]
    flow = flow_to_torch(flow, shape_in,
                         align_corners=kwargs['align_corners'],
                         has_identity=has_identity)
    B = max(len(flow), len(image))
    if len(flow) != B:
        flow = flow.expand([B, *flow.shape[1:]])
    if len(image) != B:
        image = image.expand([B, *image.shape[1:]])
    if not image.dtype.is_floating_point:
        vmax = flow.new_full([B, C, *shape_out], -float('inf'))
        warped = image.new_zeros([B, C, *shape_out])
        for label in image.unique():
            w = F.grid_sample((image == label).to(flow), flow, **kwargs)
            warped[w > vmax] = label
            vmax = torch.maximum(vmax, w)
        return warped
    else:
        return F.grid_sample(image, flow, **kwargs)


def downsample(image, factor=None, shape=None, anchor='center'):
    """Downsample using centers or edges of the corner voxels as anchors.

    Parameters
    ----------
    image : (B, C, *shape_in) tensor
    factor OR shape : int or list[int]
    anchor : {'center', 'edge'} tensor

    Returns
    -------
    image : (B, C, *shape_out)

    """
    if shape and factor:
        raise ValueError('Only one of `shape` and `factor` should be used.')
    ndim = image.dim() - 2
    mode = 'linear' if ndim == 1 else 'bilinear' if ndim == 2 else 'trilinear'
    align_corners = (anchor[0].lower() == 'c')
    recompute_scale_factor = factor is not None
    if factor:
        if isinstance(factor, (list, tuple)):
            factor = [1/f for f in factor]
        else:
            factor = 1/factor
    image = F.interpolate(image, size=shape, scale_factor=factor,
                          mode=mode, align_corners=align_corners,
                          recompute_scale_factor=recompute_scale_factor)
    return image


def upsample(image, factor=None, shape=None, anchor='center'):
    """Upsample using centers or edges of the corner voxels as anchors.

    Parameters
    ----------
    image : (B, C, *shape_in) tensor
    factor OR shape : int or list[int]
    anchor : {'center', 'edge'}

    Returns
    -------
    image : (B, C, *shape_out) tensor

    """
    if shape and factor:
        raise ValueError('Only one of `shape` and `factor` should be used.')
    ndim = image.dim() - 2
    mode = 'linear' if ndim == 1 else 'bilinear' if ndim == 2 else 'trilinear'
    align_corners = (anchor[0].lower() == 'c')
    recompute_scale_factor = factor is not None
    image = F.interpolate(image, size=shape, scale_factor=factor,
                          mode=mode, align_corners=align_corners,
                          recompute_scale_factor=recompute_scale_factor)
    return image


def downsample_flow(flow, factor=None, shape=None, anchor='center'):
    """Downsample a flow field  using centers or edges of the corner
    voxels as anchors.

    Parameters
    ----------
    flow : (B, *shape_in, D) tensor
    factor OR shape : int or list[int]
    anchor : {'center', 'edge'}

    Returns
    -------
    flow : (B, *shape_out, D) tensor

    """
    shape_in = flow.shape[1:-1]

    # downsample flow
    flow = flow.movedim(-1, 1)
    flow = downsample(flow, factor, shape, anchor)
    flow = flow.movedim(1, -1)

    # compute scale
    shape_out = flow.shape[1:-1]
    if anchor[0] == 'c':
        factor = [(fout - 1) / (fin - 1)
                  for fout, fin in zip(shape_out, shape_in)]
    else:
        factor = [fout / fin
                  for fout, fin in zip(shape_out, shape_in)]

    # rescale displacement
    ndim = flow.dim() - 2
    for d in range(ndim):
        flow[..., d] /= factor[d]

    return flow


def upsample_flow(flow, factor=None, shape=None, anchor='center'):
    """Upsample a flow field  using centers or edges of the corner
    voxels as anchors.

    Parameters
    ----------
    flow : (B, *shape_in, D) tensor
    factor OR shape : int or list[int]
    anchor : {'center', 'edge'}

    Returns
    -------
    flow : (B, *shape_out, D) tensor

    """
    shape_in = flow.shape[1:-1]

    # upsample flow
    flow = flow.movedim(-1, 1)
    flow = upsample(flow, factor, shape, anchor)
    flow = flow.movedim(1, -1)

    # compute scale
    shape_out = flow.shape[1:-1]
    if anchor[0] == 'c':
        factor = [(fout - 1) / (fin - 1)
                  for fout, fin in zip(shape_out, shape_in)]
    else:
        factor = [fout / fin
                  for fout, fin in zip(shape_out, shape_in)]

    # rescale displacement
    ndim = flow.dim() - 2
    for d in range(ndim):
        flow[..., d] /= factor[d]

    return flow


def downsample_convlike(image, kernel_size, stride, padding=0):
    """Downsample using the same alignment pattern as a strided convolution

    Parameters
    ----------
    image : (B, C, *shape_in) tensor
    kernel_size : int or list[int]
    stride : int or list[int]
    padding : int or list[int]

    Returns
    -------
    image : (B, C, *shape_out) tensor

    """
    shape_in = image.shape[2:]
    ndim = image.dim() - 2
    kernel_size = ensure_list(kernel_size, ndim)
    stride = ensure_list(stride, ndim)
    padding = ensure_list(padding, ndim)

    # create sampling grid
    backend = dict(dtype=image.dtype, device=image.device)
    shape_out = [(l + 2 * p - k)//s + 1 for l, k, s, p
                 in zip(shape_in, kernel_size, stride, padding)]

    flow = [torch.arange(s, **backend) for s in shape_out]
    for f, k, s, p in zip(flow, kernel_size, stride, padding):
        f.mul_(s).add_((k-1-p)/2)
    flow = torch.stack(meshgrid_ij(*flow), -1)

    # interpolate
    return apply_flow(image, flow[None], has_identity=True)


def downsample_flow_convlike(flow, kernel_size, stride, padding=0):
    """Downsample a flow field using the same alignment pattern as a
    strided convolution

    Parameters
    ----------
    flow : (B, *shape_in, D) tensor
        Input image
    kernel_size : int or list[int]
        Kernel size of the equivalent convolution
    stride : int or list[int]
        Stride of the equivalent convolution

    Returns
    -------
    flow : (B, *shape_out, D) tensor

    """
    # downsample flow
    flow = flow.movedim(-1, 1)
    flow = downsample_convlike(flow, kernel_size, stride, padding)
    flow = flow.movedim(1, -1)

    # rescale displacement
    ndim = flow.dim() - 2
    stride = ensure_list(stride, ndim)
    for d in range(ndim):
        flow[..., d] /= stride[d]

    return flow


def upsample_convlike(image, kernel_size, stride, padding=0, shape=None):
    """Upsample using the same alignment pattern as a transposed convolution

    Parameters
    ----------
    image : (B, C, *shape_in) tensor
    kernel_size : int or list[int]
    stride : int or list[int]
    shape : int or list[int]

    Returns
    -------
    image : (B, C, *shape_out) tensor

    """
    shape_in = image.shape[2:]
    ndim = image.dim() - 2
    kernel_size = ensure_list(kernel_size, ndim)
    stride = ensure_list(stride, ndim)
    padding = ensure_list(padding, ndim)
    if shape:
        shape = ensure_list(shape, ndim)

    # create sampling grid
    backend = dict(dtype=image.dtype, device=image.device)
    if not shape:
        shape = [(l - 1) * s - 2 * p + k for l, k, s, p
                 in zip(shape_in, kernel_size, stride, padding)]

    flow = [torch.arange(s, **backend) for s in shape]
    for f, k, s, p in zip(flow, kernel_size, stride, padding):
        f.sub_((k-1-p)/2).div_(s)
    flow = torch.stack(meshgrid_ij(*flow), -1)

    # interpolate
    return apply_flow(image, flow[None], has_identity=True)


def upsample_flow_convlike(flow, kernel_size, stride, padding=0, shape=None):
    """Upsample a flow field using the same alignment pattern as a
    transposed convolution

    Parameters
    ----------
    flow : (B, *shape_in, D) tensor
        Input image
    kernel_size : int or list[int]
        Kernel size of the equivalent convolution
    stride : int or list[int]
        Stride of the equivalent convolution

    Returns
    -------
    flow : (B, *shape_out, D) tensor

    """
    # upsample flow
    flow = flow.movedim(-1, 1)
    flow = upsample_convlike(flow, kernel_size, stride, padding, shape)
    flow = flow.movedim(1, -1)

    # rescale displacement
    ndim = flow.dim() - 2
    stride = ensure_list(stride, ndim)
    for d, s in enumerate(stride):
        flow[..., d].mul_(s)

    return flow


def compose_flows(flow_left, flow_right, has_identity=False):
    """Compute flow_left o flow_right

    Parameters
    ----------
    flow_left : (B, *shape, D) tensor
    flow_right : (B, *shape, D) tensor
    has_identity : bool, default=False

    Returns
    -------
    flow : (B, *shape, D) tensor

    """
    if has_identity:
        flow_left = sub_identity(flow_left)
    flow_left = flow_left.movedim(-1, 1)
    flow = apply_flow(flow_left, flow_right, has_identity=has_identity)
    flow = flow.movedim(1, -1)
    flow += flow_right
    return flow


def bracket(vel_left, vel_right):
    """Compute the Lie bracket of two SVFs

    Parameters
    ----------
    vel_left : (B, *shape, D) tensor
    vel_right : (B, *shape, D) tensor

    Returns
    -------
    bkt : (B, *shape, D) tensor

    """
    return (compose_flows(vel_left, vel_right) -
            compose_flows(vel_right, vel_left))


def exp_velocity(vel, steps=8):
    """Exponentiate a stationary velocity field by scaling and squaring

    Parameters
    ----------
    vel : (B, *shape, D) tensor
        Stationary velocity
    steps : int, default=8
        Number of scaling and squaring steps

    Returns
    -------
    flow : (B, *shape, D) tensor
        Displacement field

    """
    vel = vel / (2**steps)
    for i in range(steps):
        vel = vel + apply_flow(vel.movedim(-1, 1), vel).movedim(1, -1)
    return vel


def compose_velocities(vel_left, vel_right, order=2):
    """Find v such that exp(v) = exp(u) o exp(w) using the
    (truncated) Baker–Campbell–Hausdorff formula.

    https://en.wikipedia.org/wiki/BCH_formula

    Parameters
    ----------
    vel_left : (B, *shape, D) tensor
    vel_right : (B, *shape, D) tensor
    order : 1..4, default=2
        Truncation order.

    Returns
    -------
    vel : (B, *shape, D) tensor

    """
    vel = vel_left + vel_right
    if order > 1:
        b1 = bracket(vel_left, vel_right)
        vel.add_(b1, alpha=1/2)
        if order > 2:
            b2_left = bracket(vel_left, b1)
            vel.add_(b2_left, alpha=1/12)
            b2_right = bracket(vel_right, b1)
            vel.add_(b2_right, alpha=-1/12)
            if order > 3:
                b3 = bracket(vel_right, b2_left)
                vel.add_(b3, alpha=-1/24)
                if order > 4:
                    raise ValueError('BCH only implemented up to order 4')
    return vel

