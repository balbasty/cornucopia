__all__ = [
    'fieldmap_to_shift',
    'labels_to_chi',
    'ppm_to_hz',
    'chi_to_fieldmap',
    'susceptibility_phantom',
    'spherical_harmonics',
    'yield_spherical_harmonics',
]
import torch
from torch import fft
import math
import itertools
from .warps import identity as identity_grid, cartesian_grid
from .py import prod


r"""
Absolute MR susceptibility values.

!!! warning
    the `chi_to_fieldmap` function takes *delta* susceptibility values, with
    respect to the air susceptibility. The susceptibility of the air should
    therefore be subtracted from these values before being passed to
    `mrfield`.

!!! note
    All values are expressed in ppm (parts per million).
    They get multiplied by 1e-6 in `mrfield`

References
----------
1.  "Perturbation Method for Magnetic Field Calculations of
       Nonconductive Objects"
      Mark Jenkinson, James L. Wilson, and Peter Jezzard
      MRM, 2004
      https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.20194
2.  "Susceptibility mapping of air, bone, and calcium in the head"
      Sagar Buch, Saifeng Liu, Yongquan Ye, Yu-Chung Norman Cheng,
      Jaladhar Neelavalli, and E. Mark Haacke
      MRM, 2014
3.  "Whole-brain susceptibility mapping at high field: A comparison
       of multiple- and single-orientation methods"
      Sam Wharton, and Richard Bowtell
      NeuroImage, 2010
4.  "Quantitative susceptibility mapping of human brain reflects
       spatial variation in tissue composition"
      Wei Li, Bing Wua, and Chunlei Liu
      NeuroImage 2011
5.  "Human brain atlas for automated region of interest selection in
       quantitative susceptibility mapping: Application to determine iron
       content in deep gray matter structures"
      Issel Anne L.Lim, Andreia V. Faria, Xu Li, Johnny T.C.Hsu,
      Raag D.Airan, Susumu Mori, Peter C.M. van Zijl
      NeuroImage, 2013
"""
mr_chi = {
    'air': 0.4,         # Jenkinson (Buch: 0.35)
    'water': -9.1,      # Jenkinson (Buch: -9.05)
    'bone': -11.3,      # Buch
    'teeth': -12.5,     # Buch
}


# TODO: FFTs could be a bit more efficient by using rfft/hfft when possible
#       e.g., the momentum is real -> rfft
#             the convolution kernel is real + symmetric -> rfft or hfft


def fieldmap_to_shift(delta, bandwidth=140):
    """Convert fieldmap to voxel shift map

    Parameters
    ----------
    delta : tensor
        Fieldmap (Hz)
    bandwidth : float, default=140
        Bandwidth (Hz/pixel)

    Returns
    -------
    shift : tensor
        Displacement map (pixel)
    """
    return delta / bandwidth


def labels_to_chi(label_map, label_dict=None,
                  reference='air', dtype=None):
    """Synthesize a susceptibility map from labels

    Parameters
    ----------
    label_map : tensor[int]
        Input label map
    label_dict : dict[int, float or str], default={0: 'air', 1: 'water'}
        Dictionary mapping labels to susceptibility values or region names
    reference : float or str, default='air'
        Reference susceptibility
    dtype : torch.dtype
        Data type

    Returns
    -------
    delta_chi : tensor
        Delta susceptibility map, with respect to the reference (ppm)

    """
    if not label_dict:
        label_dict = {0: 'air', 1: 'water'}
    unique_labels = label_map.unique().long().tolist()

    dtype = dtype or torch.get_default_dtype()
    delta = torch.empty_like(label_map, dtype=dtype)

    for label in unique_labels:
        susceptibility = label_dict.get(label, 'water')

        if susceptibility in mr_chi:
            susceptibility = mr_chi[susceptibility]
        elif susceptibility in jenkinson_chi:
            susceptibility = jenkinson_chi[susceptibility]
        elif susceptibility in buch_chi_delta_water:
            susceptibility = buch_chi_delta_water[susceptibility] \
                           + mr_chi['water']
        elif susceptibility in li_chi_delta_water:
            susceptibility = li_chi_delta_water[susceptibility] \
                           + mr_chi['water']
        elif susceptibility in wharton_chi_delta_water:
            susceptibility = wharton_chi_delta_water[susceptibility] \
                           + mr_chi['water']
        else:
            susceptibility = mr_chi['water']

        delta[label_map == label] = susceptibility

    if isinstance(reference, str):
        reference = mr_chi[reference]
    delta -= reference
    return delta


def ppm_to_hz(fmap, b0=3, freq=42.576E6):
    """Convert part-per-million to Hz

    Parameters
    ----------
    fmap : tensor
        Fieldmap in ppm
    b0 : float, default=3
        Field strength (Tesla)
    freq : float, default=42.576E6
        Larmor frequency (Hz)

    Returns
    -------
    fmap : tensor
        Fieldmap in Hz

    """
    return fmap * (b0 * freq)


def chi_to_fieldmap(
        ds, zdim=-1, dim=None, s0=mr_chi['air'],
        s1=mr_chi['water'] - mr_chi['air'], vx=1):
    """Generate a MR fieldmap from a MR susceptibility map.

    Parameters
    ----------
    ds : (..., *spatial) tensor
        Susceptibility delta map (delta from air susceptibility) in ppm.
        If bool, `s1` will be used to set the value inside the mask.
        If float, should contain quantitative delta values in ppm.
    zdim : int, default=-1
        Dimension of the main magnetic field.
    dim : int, default=ds.dim()
        Number of spatial dimensions.
    s0 : float, default=0.4
        Susceptibility of air (ppm)
    s1 : float, default=-9.5
        Susceptibility of tissue minus susceptiblity of air (ppm)
        (only used if `ds` is a boolean mask)
    vx : [sequence of] float
        Voxel size

    Returns
    -------
    delta_b0 : tensor
        MR field map (ppm).

    References
    ----------
    1. "Perturbation Method for Magnetic Field Calculations of
           Nonconductive Objects"
          Mark Jenkinson, James L. Wilson, and Peter Jezzard
          MRM, 2004
          https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.20194

    """
    ds = torch.as_tensor(ds)
    backend = dict(dtype=ds.dtype, device=ds.device)
    if ds.dtype is torch.bool:
        backend['dtype'] = torch.get_default_dtype()

    dim = dim or ds.dim()
    shape = ds.shape[-dim:]
    zdim = zdim - ds.dim() if zdim >= 0 else zdim
    # ^ number from end so that we can apply to vx
    vx = torch.as_tensor(vx).flatten().tolist()
    vx += vx[-1:] * max(0, dim - len(vx))

    if ds.dtype is torch.bool:
        ds = ds.to(**backend)
    else:
        s1 = ds.abs().max()
        ds = ds / s1
    s1 = s1 * 1e-6
    s0 = s0 * 1e-6

    # Analytical implementation following Jenkinson et al.
    g = greens(shape, zdim, voxel_size=vx, **backend)
    f = greens_apply(ds, g)

    # apply rest of the equation
    out = ds * ((1. + s0) / (3. + s0))
    out -= f
    out *= s1 / (1 + s0)

    return out


def greens(shape, zdim=-1, voxel_size=1, dtype=None, device=None):
    """Semi-analytical second derivative of the Greens kernel.

    This function implements exactly the solution from Jenkinson et al.
    (Same as in the FSL source code), with the assumption that
    no gradients are played and the main field is constant and has
    no orthogonal components (Bz = B0, Bx = By = 0).

    The Greens kernel and its second derivatives are derived analytically
    and integrated numerically over a voxel.

    The returned tensor has already been Fourier transformed and could
    be cached if multiple field simulations with the same lattice size
    must be performed in a row.

    Parameters
    ----------
    shape : sequence of int
        Lattice shape
    zdim : int, defualt=-1
        Dimension of the main magnetic field
    voxel_size : [sequence of] int
        Voxel size
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    kernel : (*shape) tensor
        Fourier transform of the (second derivatives of the) Greens kernel.

    """
    def atan(num, den):
        """Robust atan"""
        return torch.where(den.abs() > 1e-8,
                           torch.atan_(num / den),
                           torch.atan2(num, den))

    dim = len(list(shape))
    dims = list(range(-dim, 0))
    voxel_size = torch.as_tensor(voxel_size).flatten().tolist()
    voxel_size += voxel_size[-1:] * max(0, dim - len(voxel_size))
    zdim = -dim + zdim if zdim >= 0 else zdim  # use negative indexing

    if dim not in (2, 3):
        raise ValueError('Invalid dim', dim)
    if zdim not in range(-dim, 0):
        raise ValueError('Invalid zdim', zdim)

    if dim == 3:
        odims = ([-3, -2] if zdim == -1 else
                 [-3, -1] if zdim == -2 else
                 [-2, -1])
    elif dim == 2:
        odim = -2 if zdim == -1 else -1
    else:
        raise NotImplementedError

    # make zero-centered meshgrid
    g0 = identity_grid(shape, dtype=dtype, device=device)
    for g1, s in zip(g0.unbind(-1), shape):
        # g1 -= int(math.ceil(s / 2))
        g1 -= (s-1) / 2

    def shift_and_scale_grid(grid, shift):
        g = grid.clone()
        for g1, v, t in zip(g.unbind(-1), voxel_size, shift):
            g1 += t                         # apply shift
            g1 *= v                         # convert to mm
        return g

    g = 0
    for shift in itertools.product([-0.5, 0.5], repeat=dim):
        # Integrate across a voxel
        g1 = shift_and_scale_grid(g0, shift)
        # Compute \int G" dx dy dz
        # where G is the Green function of the 2D or 3D Laplacian and
        # G" is the second derivative of G wrt z.
        z = g1[..., zdim]
        if dim == 3:
            r = g1.square().sum(-1).sqrt_()
            x = g1[..., odims[0]]
            y = g1[..., odims[1]]
            g1 = atan(x * y, z * r)
        else:
            y = g1[..., odim]
            g1 = atan(y, z)
        if prod(shift) < 0:
            g -= g1
        else:
            g += g1

    g /= (2 ** (dim-1)) * math.pi
    g = fft.ifftshift(g, dims)  # move center voxel to first voxel

    # fourier transform
    g = fft.fftn(g, dim=dims).real
    return g


def greens_apply(mom, greens):
    """Apply the Greens function to a momentum field.

    Parameters
    ----------
    mom : (..., *spatial) tensor
        Momentum
    greens : (*spatial) tensor
        Greens function

    Returns
    -------
    field : (..., *spatial) tensor
        Field

    """
    greens = greens.to(mom)
    dim = greens.dim()
    dims = list(range(-dim, 0))

    # fourier transform
    mom = fft.fftn(mom, dim=dims)

    # voxel wise multiplication
    mom *= greens

    # inverse fourier transform
    mom = fft.ifftn(mom, dim=dims).real

    return mom


def susceptibility_phantom(shape, radius=None, dtype=None, device=None):
    """Generate a circle/sphere susceptibility phantom

    Parameters
    ----------
    shape : sequence of int
    radius : default=shape/4
    dtype : optional
    backend : optional

    Returns
    -------
    f : (*shape) tensor[bool]
        susceptibility delta map

    """
    radius = radius or (min(shape) / 4.)
    f = identity_grid(shape, dtype=dtype, device=device)
    for comp, s in zip(f.unbind(-1), shape):
        comp -= s / 2
    f = f.square().sum(-1).sqrt() <= radius
    return f


def diff(x, ndim):
    """Forward finite differences"""
    g = x.new_zeros([ndim, *x.shape])
    for dim, g1 in enumerate(g):
        dim = dim - ndim
        g1 = g1.transpose(dim, 0)
        x = x.transpose(dim, 0)
        torch.sub(x[1:], x[:-1], out=g1[:-1])
        x = x.transpose(dim, 0)
    g = g.movedim(0, -1)
    return g


def shim(fmap, max_order=2, mask=None, isocenter=None, dim=None,
         lam_abs=1, lam_grad=10, returns='corrected'):
    """Subtract a linear combination of spherical harmonics that
    minimize absolute values and gradients

    Parameters
    ----------
    fmap : (..., *spatial) tensor
        Field map
    max_order : int, default=2
        Maximum order of the spherical harmonics
    mask : tensor, optional
        Mask of voxels to include (typically brain mask)
    isocenter : [sequence of] float, default=shape/2
        Coordinate of isocenter, in voxels
    dim : int, default=fmap.dim()
        Number of spatial dimensions
    lam_abs : float, default=1
        Penalty on absolute values
    lam_abs : float, default=10
        Penalty on gradients
    returns : combination of {'corrected', 'correction', 'parameters'}
        Components to return

    Returns
    -------
    corrected : (..., *spatial) tensor, if 'corrected' in `returns`
        Corrected field map (with spherical harmonics subtracted)
    correction : (..., *spatial) tensor, if 'correction' in `returns`
        Linear combination of spherical harmonics.
    parameters : (..., k) tensor, if 'parameters' in `returns`
        Parameters of the linear combination

    """
    fmap = torch.as_tensor(fmap)
    dim = dim or fmap.dim()
    shape = fmap.shape[-dim:]
    batch = fmap.shape[:-dim]
    backend = dict(dtype=fmap.dtype, device=fmap.device)

    if mask is not None:
        mask = ~mask  # make it a mask of background voxels
        mask = mask.expand(fmap.shape)

    # compute gradients
    gmap = diff(fmap, dim)
    gmap = torch.cat([gmap, fmap.unsqueeze(-1)], -1)
    gmap[..., :-1] *= lam_grad
    gmap[..., -1] *= lam_abs
    if mask is not None:
        gmap[mask, :] = 0
    gmap = gmap.reshape([*batch, -1])   # (*batch, k)

    # compute basis of spherical harmonics
    basis = []
    for i in range(1, max_order + 1):
        b0 = spherical_harmonics(shape, i, isocenter, **backend)
        b0 = torch.movedim(b0, -1, 0)
        b = diff(b0, dim)
        b = torch.cat([b, b0.unsqueeze(-1)], -1)
        del b0
        b[..., :-1] *= lam_grad
        b[..., -1] *= lam_abs
        if mask is not None:
            b[mask.expand(b.shape[:-1]), :] = 0
        b = b.reshape([b.shape[0], *batch, -1])
        basis.append(b)
    basis = torch.cat(basis, 0)
    basis = torch.movedim(basis, 0, -1)  # (*batch, vox*dim, k)

    # solve system
    prm = basis.pinverse().matmul(gmap.unsqueeze(-1)).squeeze(-1)
    # > (*batch, k)

    # rebuild basis (without taking gradients)
    basis = []
    for i in range(1, max_order + 1):
        b = spherical_harmonics(shape, i, isocenter, **backend)
        b = torch.movedim(b, -1, 0)
        b = b.reshape([b.shape[0], *batch, -1])
        basis.append(b)
    basis = torch.cat(basis, 0)
    basis = torch.movedim(basis, 0, -1)  # (*batch, vox*dim, k)

    comb = basis.matmul(prm.unsqueeze(-1)).squeeze(0)
    comb = comb.reshape([*batch, *shape])
    fmap = fmap - comb

    returns = returns.split('+')
    out = []
    for ret in returns:
        if ret == 'corrected':
            out.append(fmap)
        elif ret == 'correction':
            out.append(comb)
        elif ret[0] == 'p':
            out.append(prm)
    return out[0] if len(out) == 1 else tuple(out)


def spherical_harmonics(shape, order=2, isocenter=None, **backend):
    """Generate a basis of spherical harmonics on a lattice

    !!! note
        - This should be checked!
        - Only orders 1 and 2 implemented
        - I tried to implement some sort of "circular" harmonics in
         dimension 2 but I don't know what I am doing.
        - The basis is not orthogonal

    Parameters
    ----------
    shape : sequence of int
    order : {1, 2}, default=2
    isocenter : [sequence of] int, default=shape/2
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    b : (*shape, 2*order + 1) tensor
        Basis

    """
    shape = list(shape)
    dim = len(shape)
    if dim not in (2, 3):
        raise ValueError('Dimension must be 2 or 3')
    if order not in (1, 2):
        raise ValueError('Order must be 1 or 2')

    if isocenter is None:
        isocenter = [s / 2 for s in shape]
    isocenter = list(isocenter)
    isocenter += isocenter[-1:] * max(0, dim-len(isocenter))

    ramps = identity_grid(shape, **backend)
    for i, ramp in enumerate(ramps.unbind(-1)):
        ramp -= isocenter[i]
        ramp /= shape[i] / 2

    if order == 1:
        return ramps
    # order == 2
    if dim == 3:
        basis = [ramps[..., 0] * ramps[..., 1],
                 ramps[..., 0] * ramps[..., 2],
                 ramps[..., 1] * ramps[..., 2],
                 ramps[..., 0].square() - ramps[..., 1].square(),
                 ramps[..., 0].square() - ramps[..., 2].square()]
        return torch.stack(basis, -1)
    else:  # basis == 2
        basis = [ramps[..., 0] * ramps[..., 1],
                 ramps[..., 0].square() - ramps[..., 1].square()]
        return torch.stack(basis, -1)


def yield_spherical_harmonics(shape, order=2, isocenter=None, **backend):
    r"""Generate a basis of spherical harmonics on a lattice

    !!! note
        - This should be checked!
        - Only orders 1 and 2 implemented
        - The basis is not orthogonal

    Parameters
    ----------
    shape : sequence of int
    order : {1, 2}, default=2
    isocenter : [sequence of] int, default=shape/2
    dtype : torch.dtype, optional
    device : torch.device, optional

    Yields
    ------
    b : (*shape) tensor
        Basis
        (2d: `2*order - 1` bases, 3D: `2*order + 1` bases)

    """
    backend.setdefault('dtype', torch.get_default_dtype())
    shape = list(shape)
    dim = len(shape)
    if dim not in (2, 3):
        raise ValueError('Dimension must be 2 or 3')
    if order not in (1, 2):
        raise ValueError('Order must be 1 or 2')

    # zero-th order (global shift)
    yield torch.ones(shape, **backend)
    if order == 0:
        return

    # 1st order (linear ramps)
    if isocenter is None:
        isocenter = [s / 2 for s in shape]
    isocenter = list(isocenter)
    isocenter += isocenter[-1:] * max(0, dim-len(isocenter))

    ramps = []
    for i, ramp in enumerate(cartesian_grid(shape, **backend)):
        ramp = ramp.clone()
        ramp -= isocenter[i]
        ramp /= shape[i] / 2
        ramps.append(ramp)
        yield ramp

    if order == 1:
        return

    # 2nd order (quadratics)
    if dim == 3:
        yield 2 * ramps[0] * ramps[1]
        yield 2 * ramps[0] * ramps[2]
        yield 2 * ramps[1] * ramps[2]
        yield ramps[0].square() - ramps[1].square()
        yield ramps[0].square() - ramps[2].square()
    else:
        assert dim == 2
        yield 2 * ramps[0] * ramps[1]
        yield ramps[0].square() - ramps[1].square()


# --- values from the literature

# Wharton and Bowtell, NI, 2010
# These are relative susceptibility with respect to surrounding tissue
# (which I assume we can consider as white matter?)
wharton_chi_delta_water = {
    'sn': 0.17,         # substantia nigra
    'rn': 0.14,         # red nucleus
    'ic': -0.01,        # internal capsule
    'gp': 0.19,         # globus pallidus
    'pu': 0.10,         # putamen
    'cn': 0.09,         # caudate nucleus
    'th': 0.045,        # thalamus
    'gm_ppl': 0.043,    # posterior parietal lobe
    'gm_apl': 0.053,    # anterior parietal lobe
    'gm_fl': 0.04,      # frontal lobe
    'gm': (0.053 + 0.043 + 0.04)/3,
}

# Li et al, NI, 2011
# These are susceptibility relative to CSF
li_chi_delta_water = {
    'sn': 0.053,    # substantia nigra
    'rn': 0.032,    # red nucleus
    'ic': -0.068,   # internal capsule
    'gp': 0.087,    # globus pallidus
    'pu': 0.043,    # putamen
    'cn': 0.019,    # caudate nucleus
    'dn': 0.064,    # dentate nucleus
    'gcc': -0.033,  # genu of corpus callosum
    'scc': -0.038,  # splenium of corpus collosum
    'ss': -0.075,   # sagittal stratum
}

# Buch et al, MRM, 2014
buch_chi_delta_water = {
    'air': 9.2,
    'bone': -2.1,
    'teeth': -3.3,
}

# Jenkinson et al, MRM, 2004
# Absolute susceptibilities
jenkinson_chi = {
    'air': 0.4,
    'parenchyma': -9.1,  # (== water)
}

# Acosta-Cabronero et al., Radiology, 2018
# "not referenced"
acosta_chi = {
    'cortex': 0.0173,       # motor cortex
    'sn': 0.113,            # substantia nigra
    'gp': 0.075,            # globus pallidus
    'rn': 0.098,            # red nucleus
    'cst': -0.043,          # cerebrospinal tract
}
