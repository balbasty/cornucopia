__all__ = [
    'fieldmap_to_shift',
    'labels_to_chi',
    'ppm_to_hz',
    'chi_to_fieldmap',
    'susceptibility_phantom',
    'spherical_harmonics',
    'yield_spherical_harmonics',
]
# stdlib
import math
import itertools

# external
import torch
from torch import fft

# internal
from .warps import identity as identity_grid, cartesian_grid
from .py import prod, make_vector
from .smart_inplace import add_, mul_, div_, square_, sub_


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
    https://doi.org/10.1002/mrm.20194
2.  "Application of a Fourier-Based Method for Rapid Calculation of
     Field Inhomogeneity Due to Spatial Variation of Magnetic Susceptibility"
    Jose P. Marques, and Richard Bowtell
    CMR B, 2005
    https://doi.org/10.1002/cmr.b.20034
3.  "Susceptibility mapping of air, bone, and calcium in the head"
    Sagar Buch, Saifeng Liu, Yongquan Ye, Yu-Chung Norman Cheng,
    Jaladhar Neelavalli, and E. Mark Haacke
    MRM, 2014
4.  "Whole-brain susceptibility mapping at high field: A comparison
     of multiple- and single-orientation methods"
    Sam Wharton, and Richard Bowtell
    NeuroImage, 2010
5.  "Quantitative susceptibility mapping of human brain reflects
     spatial variation in tissue composition"
    Wei Li, Bing Wua, and Chunlei Liu
    NeuroImage 2011
6.  "Human brain atlas for automated region of interest selection in
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


def fieldmap_to_shift(delta, bandwidth=30):
    """Convert fieldmap to voxel shift map

    Parameters
    ----------
    delta : tensor
        Fieldmap (Hz)
    bandwidth : float, default=30
        Bandwidth (Hz/pixel)

    Returns
    -------
    shift : tensor
        Displacement map (pixel)
    """
    return delta / bandwidth


def _label2chi(label):
    if label in mr_chi:
        return mr_chi[label]
    elif label in jenkinson_chi:
        return jenkinson_chi[label]
    elif label in buch_chi_delta_water:
        return mr_chi['water'] + buch_chi_delta_water[label]
    elif label in li_chi_delta_water:
        return mr_chi['water'] + li_chi_delta_water[label]
    elif label in wharton_chi_delta_water:
        return mr_chi['water'] + wharton_chi_delta_water[label]
    return mr_chi['water']


def labels_to_chi(label_map, label_dict=None, dtype=None):
    """Synthesize a susceptibility map from labels

    Parameters
    ----------
    label_map : tensor[int]
        Input label map
    label_dict : dict[int, float or str], default={0: 'air', 1: 'water'}
        Dictionary mapping labels to susceptibility values or region names
    dtype : torch.dtype
        Data type

    Returns
    -------
    chi : tensor
        Susceptibility map (ppm)

    """
    if not label_dict:
        label_dict = {0: 'air', 1: 'water'}

    unique_labels = label_map.unique().long().tolist()
    label_range = max(unique_labels) - min(unique_labels) + 1

    dtype = dtype or torch.get_default_dtype()

    susceptibility = {
        label: _label2chi(label_dict.get(label, 'water'))
        for label in unique_labels
    }

    if label_range < 2**16:
        # Use a fast lookup table if the label range is small enough
        # (Otherwise, the lookup table would be too large)

        conversion = torch.full(
            [label_range], mr_chi['water'],
            dtype=dtype, device=label_map.device
        )
        for src, dst in susceptibility.items():
            conversion[src] = dst

        if min(unique_labels) != 0:
            label_map = label_map.to(torch.long, copy=True)
            label_map -= min(unique_labels)
        else:
            label_map = label_map.to(torch.long)

        chi = conversion[label_map]

    else:
        # Loop through labels

        chi = torch.full_like(label_map, mr_chi['water'], dtype=dtype)
        for src, dst in susceptibility.items():
            chi[label_map == src] = dst

    return chi


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
    chi,
    zaxis=-1,
    ndim=None,
    chi0=mr_chi['air'],
    delta=False,
    vx=1,
    mode='marques',
):
    """Generate a MR fieldmap from a MR susceptibility map.

    Parameters
    ----------
    chi : (..., *spatial) tensor
        Magnetic susceptibility map (ppm).
        Each voxel should contain the magnetic susceptibility of the
        tissue in this voxel. Voxels that contain air should have the
        susceptibility of air (e.g., 0.4 ppm).
    zaxis : int | sequence[float], default=-1
        Direction of the main magnetic field.
        If a vector or floats, it is unit vector that encodes the direction
        of the main magnetic field in the "scaled voxel" coordinate systems.
        If an int, the main magnetic field is assumed to be aligned
        with one of the dimensions of the voxel grid, and `zaxis` is the
        index of this dimension. I.e., `0` is equivalent to `[1, 0, 0]`.
    ndim : int, default=`ds.ndim`
        Number of spatial dimensions.
    chi0 : float, default=0.4
        Susceptibility of air (ppm)
    delta : bool | float, default=False
        * If a bool, the input `chi` is relative to the susceptibility of air
          (no need to subtract `chi0`).
          I.e., `true_chi = chi + chi0`.
        * If a float, the input `chi` is relative to the susceptibility of air,
          and should be scaled by the value of delta.
          I.e., `true_chi = chi * delta + chi0`.
    vx : [sequence of] float
        Voxel size
    mode : {'jenkinson', 'marques'}, default='jenkinson'
        Method to compute the Greens kernel.

    Returns
    -------
    b : tensor
        MR field map (ppm).

    References
    ----------
    1.  "Perturbation Method for Magnetic Field Calculations of
         Nonconductive Objects"
        Mark Jenkinson, James L. Wilson, and Peter Jezzard
        MRM, 2004
        https://doi.org/10.1002/mrm.20194
    2.  "Application of a Fourier-Based Method for Rapid Calculation of
         Field Inhomogeneity Due to Spatial Variation of Magnetic Susceptibility"
        Jose P. Marques, and Richard Bowtell
        CMR B, 2005
        https://doi.org/10.1002/cmr.b.20034

    """
    chi = torch.as_tensor(chi)
    backend = dict(dtype=chi.dtype, device=chi.device)
    if chi.dtype is torch.bool:
        backend['dtype'] = torch.get_default_dtype()

    ndim = ndim or chi.ndim
    shape = chi.shape[-ndim:]
    vx = make_vector(vx, ndim)
    if isinstance(zaxis, int):
        zaxis = zaxis - chi.ndim if zaxis >= 0 else zaxis
        # ^ number from end so that we can apply to vx

    # Compute susceptibility difference
    if delta is False:
        chi1 = chi - chi0
    elif delta is not True:
        chi1 = chi * delta
    else:
        chi1 = chi

    # Compute convolution kernel (in Fourier domain)
    mode = mode[:1].lower()
    greens = {
        'j': greens_jenkinson,
        'm': greens_marques,
    }[mode]
    g = greens(shape, zaxis, voxel_size=vx, **backend)

    # and apply it to the susceptibility map
    f = greens_apply(chi1, g)

    # Apply rest of the equation
    if mode == 'j':
        # δB = B0 * (δχ / (3 + χ0) - (G" * δχ) / (1 + χ0)) [Jenkinson.13]
        # where G is the Green function of the (negative) Laplacian
        #    -> G(x) = 1 / (4 * π * |x|)
        out = chi1 * ((1. + chi0) / (3. + chi0))
        out = sub_(out, f)
        out = mul_(out, 1. / (1. + chi0))
        # NOTE
        #   The DC component of G" is -2/3, so the average delta field
        #   is B0 * average(δχ) * (1 / (3 + χ0) + 2 / (3 * (1 + χ0)))
    elif mode == 'm':
        # δB = B0 * (δχ / 3 - (Rz^2) * δχ) [Marques.6]
        #   where Rz^2 is the (negative) squared Riesz transform
        #      -> Rz^2 = (∂z^2/∇^2)
        out = chi1 / 3.
        out = sub_(out, f)
        # NOTE:
        #   The kernel is ill-defined at the center of k-space.
        #   Marques and Bowtell suggest to ensure that the average
        #   delta field is - B0 * χ0 / 3, which is the theoretical
        #   value for an infinite, homogeneous object with magnetic
        #   susceptibility χ0, when the Lorentz correction is included
        #   (i.e., B = (1 + χ0 / 3) B0).
        #
        # We therefore ensure that: average(δB) = - B0 * χ0 / 3
        mean = out.sum().div_(out.numel())  # weird fix for pytorch 2.0
        out = sub_(out, mean + chi0 / 3.)
        # out = out - (out.mean() + chi0 / 3.)

    out = mul_(out, 1e-6)  # account for ppm
    return out


def greens_jenkinson(shape, zaxis=-1, voxel_size=1, dtype=None, device=None):
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
    zaxis : int | sequence[float], default=-1
        Direction of the main magnetic field.
        If a vector or floats, it is unit vector that encodes the direction
        of the main magnetic field in the "scaled voxel" coordinate systems.
        If an int, the main magnetic field is assumed to be aligned
        with one of the dimensions of the voxel grid, and `zaxis` is the
        index of this dimension. I.e., `0` is equivalent to `[1, 0, 0]`.
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
        den = den.clone()
        den[(den < 1e-8) & (den >= 0)] += 1e-8
        den[(den > -1e-8) & (den < 0)] -= 1e-8
        return torch.atan2(num, den)

    def dot(a, b):
        return a.unsqueeze(-2).matmul(b.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    ndim = len(list(shape))
    dims = list(range(-ndim, 0))
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, ndim, dtype=dtype, device=device)

    if ndim not in (2, 3):
        raise ValueError('Invalid ndim', ndim)

    if isinstance(zaxis, int):

        zaxis = -ndim + zaxis if zaxis >= 0 else zaxis  # use negative indexing

        if zaxis not in range(-ndim, 0):
            raise ValueError('Invalid zaxis', zaxis)

        if ndim == 3:
            yaxis, xaxis = ([-2, -3] if zaxis == -1 else
                            [-1, -3] if zaxis == -2 else
                            [-1, -2])
        elif ndim == 2:
            yaxis = -2 if zaxis == -1 else -1
        else:
            raise NotImplementedError

    else:
        zaxis = make_vector(zaxis, dtype=dtype, device=device)
        zaxis = zaxis / zaxis.norm()  # ensure unit vector

        if len(zaxis) not in (2, 3):
            raise ValueError('Invalid zaxis', zaxis)

        if ndim == 3:
            yaxis, xaxis = zaxis.clone(), zaxis.clone()
            yaxis[:2] = yaxis[:2].flip(0)
            yaxis[1].neg_()
            yaxis[2].zero_()
            xaxis[-2:] = xaxis[-2:].flip(0)
            xaxis[2].neg_()
            xaxis[0].zero_()
        else:
            yaxis = zaxis.flip(0)
            yaxis[1].neg_()

    # make zero-centered meshgrid
    # center = [math.ceil((s-1) / 2) for s in shape]  # matches ifftshift
    center = [(s-1) / 2 for s in shape]
    g0 = identity_grid(shape, dtype=dtype, device=device)
    for g1, c in zip(g0.unbind(-1), center):
        g1 -= c

    def shift_and_scale_grid(grid, shift):
        return mul_(grid + make_vector(shift).to(grid), voxel_size.to(grid))

    g = 0
    for shift in itertools.product([-0.5, 0.5], repeat=ndim):
        # Integrate across a voxel
        g1 = shift_and_scale_grid(g0, shift)
        # Compute \int G" dx dy dz, where
        # * G is the Green function of the Laplacian, and
        # * G" is the second derivative of G wrt z.

        if isinstance(zaxis, int):
            z = g1[..., zaxis]
            y = g1[..., yaxis]
            if ndim == 3:
                x = g1[..., xaxis]
        else:
            z = dot(g1, zaxis)
            y = dot(g1, yaxis)
            if ndim == 3:
                x = dot(g1, xaxis)

        if ndim == 3:
            r = g1.square().sum(-1).clamp_min_(1e-8).sqrt_()
            g1 = atan(x * y, z * r)
        else:
            g1 = atan(y, z)
        if prod(shift) < 0:
            g -= g1
        else:
            g += g1

    # divide by 4*pi
    g = div_(g, (2 ** (ndim-1)) * math.pi)

    # move center voxel to first voxel
    g = fft.ifftshift(g, dims)

    # fourier transform
    g = fft.fftn(g, dim=dims).real
    return g


greens = greens_jenkinson  # backward compatibility


def greens_marques(shape, zaxis=-1, voxel_size=1, dtype=None, device=None):
    """Greens kernel according to Marques and Bowtell (2005)

    The center of the kernel is ill-defined and set to zero. The DC
    component of the convolved image will be set to s0/3.

    The returned tensor has already been Fourier transformed and can
    be cached if multiple field simulations with the same lattice size
    must be performed in a row.

    Parameters
    ----------
    shape : sequence of int
        Lattice shape
    zaxis : int | sequence[float], default=-1
        Direction of the main magnetic field.
        If a vector or floats, it is unit vector that encodes the direction
        of the main magnetic field in the "scaled voxel" coordinate systems.
        If an int, the main magnetic field is assumed to be aligned
        with one of the dimensions of the voxel grid, and `zaxis` is the
        index of this dimension. I.e., `0` is equivalent to `[1, 0, 0]`.
    voxel_size : [sequence of] int
        Voxel size
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    kernel : (*shape) tensor
        Fourier transform of the (second derivatives of the) Greens kernel.

    """
    def dot(a, b):
        return a.unsqueeze(-2).matmul(b.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    ndim = len(list(shape))
    dims = list(range(-ndim, 0))
    dtype = dtype or torch.get_default_dtype()
    voxel_size = make_vector(voxel_size, ndim, dtype=dtype, device=device)

    if ndim not in (2, 3):
        raise ValueError('Invalid ndim', ndim)

    if isinstance(zaxis, int):

        zaxis = -ndim + zaxis if zaxis >= 0 else zaxis  # use negative indexing

        if zaxis not in range(-ndim, 0):
            raise ValueError('Invalid zaxis', zaxis)

        if ndim == 3:
            yaxis, xaxis = ([-2, -3] if zaxis == -1 else
                            [-1, -3] if zaxis == -2 else
                            [-1, -2])
        elif ndim == 2:
            yaxis = -2 if zaxis == -1 else -1
        else:
            raise NotImplementedError

    else:
        zaxis = make_vector(zaxis, dtype=dtype, device=device)
        zaxis = zaxis / zaxis.norm()  # ensure unit vector

        if len(zaxis) not in (2, 3):
            raise ValueError('Invalid zaxis', zaxis)

        if ndim == 3:
            yaxis, xaxis = zaxis.clone(), zaxis.clone()
            yaxis[:2] = yaxis[:2].flip(0)
            yaxis[1].neg_()
            yaxis[2].zero_()
            xaxis[-2:] = xaxis[-2:].flip(0)
            xaxis[2].neg_()
            xaxis[0].zero_()
        else:
            yaxis = zaxis.flip(0)
            yaxis[1].neg_()

    # make zero-centered meshgrid
    center = [math.ceil((s-1) / 2) for s in shape]  # matches ifftshift
    g0 = identity_grid(shape, dtype=dtype, device=device)
    for g1, c in zip(g0.unbind(-1), center):
        g1 -= c

    # scale by voxel size
    g0 = mul_(g0, voxel_size)

    # compute kernel
    if isinstance(zaxis, int):
        g0 = square_(g0)
        z = g0[..., zaxis]
        y = g0[..., yaxis]
        r = add_(y, z)
        if ndim == 3:
            x = g0[..., xaxis]
            r = add_(r, x)
    else:
        z = square_(dot(g0, zaxis))
        y = square_(dot(g0, yaxis))
        r = add_(y, z)
        if ndim == 3:
            x = square_(dot(g0, xaxis))
            r = add_(r, x)

    r = r.clamp_min_(1e-8)
    g = div_(z, r)
    g[tuple(center)] = 0

    # move center voxel to first voxel
    g = fft.ifftshift(g, dims)

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
    ndim = greens.ndim
    dims = list(range(-ndim, 0))

    # fourier transform
    mom = fft.fftn(mom, dim=dims)

    # voxel wise multiplication
    mom = mul_(mom, greens)

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
    f = f.square().sum(-1).clamp_min_(1e-8).sqrt() <= radius
    return f


def diff(x, ndim):
    """Forward finite differences"""

    def diff1(out, inp, dim):
        out = out.transpose(dim, 0)
        inp = inp.transpose(dim, 0)
        if inp.requires_grad:
            out[:-1] = torch.sub(inp[1:], inp[:-1])
        else:
            torch.sub(inp[1:], inp[:-1], out=out[:-1])

    g = x.new_zeros([ndim, *x.shape])
    for dim in range(ndim):
        diff1(g[dim], x, dim - ndim)
    g = g.movedim(0, -1)
    return g


def shim(fmap, max_order=2, mask=None, isocenter=None, ndim=None,
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
    ndim : int, default=fmap.ndim
        Number of spatial dimensions
    lam_abs : float, default=1
        Penalty on absolute values
    lam_grad : float, default=10
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
    ndim = ndim or fmap.ndim
    shape = fmap.shape[-ndim:]
    batch = fmap.shape[:-ndim]
    backend = dict(dtype=fmap.dtype, device=fmap.device)

    if mask is not None:
        mask = ~mask  # make it a mask of background voxels
        mask = mask.expand(fmap.shape)

    # compute gradients
    if lam_grad != 0:
        gmap = diff(fmap, ndim)
        gmap *= lam_grad
        if lam_abs != 0:
            gmap = torch.cat([gmap, fmap.unsqueeze(-1)], -1)
            gmap[..., -1] *= lam_abs
    else:
        gmap = fmap.unsqueeze(-1) * lam_abs
    if mask is not None:
        gmap[mask, :] = 0
    gmap = gmap.reshape([*batch, -1])   # (*batch, k)

    # compute basis of spherical harmonics
    basis = []
    for i in range(0, max_order + 1):
        b0 = spherical_harmonics(shape, i, isocenter, **backend)
        b0 = torch.movedim(b0, -1, 0)
        if lam_grad != 0:
            b = diff(b0, ndim)
            b *= lam_grad
            if lam_abs != 0:
                b = torch.cat([b, b0.unsqueeze(-1)], -1)
                b[..., -1] *= lam_abs
        else:
            b = b0.unsqueeze(-1) * lam_abs
        del b0
        if mask is not None:
            b[mask.expand(b.shape[:-1]), :] = 0
        b = b.reshape([b.shape[0], *batch, -1])
        basis.append(b)
    basis = torch.cat(basis, 0)
    basis = torch.movedim(basis, 0, -1)  # (*batch, vox*ndim, k)

    # solve system
    prm = basis.pinverse().matmul(gmap.unsqueeze(-1)).squeeze(-1)
    # > (*batch, k)

    # rebuild basis (without taking gradients)
    basis = []
    for i in range(0, max_order + 1):
        b = spherical_harmonics(shape, i, isocenter, **backend)
        b = torch.movedim(b, -1, 0)
        b = b.reshape([b.shape[0], *batch, -1])
        basis.append(b)
    basis = torch.cat(basis, 0)
    basis = torch.movedim(basis, 0, -1)  # (*batch, vox*ndim, k)

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
        - Only orders 0, 1 and 2 implemented
        - I tried to implement some sort of "circular" harmonics in
         dimension 2 but I don't know what I am doing.
        - The basis is not orthogonal

    Parameters
    ----------
    shape : sequence of int
    order : {0, 1, 2}, default=2
    isocenter : [sequence of] int, default=shape/2
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    b : (*shape, 2*order + 1) tensor
        Basis

    """
    shape = list(shape)
    ndim = len(shape)
    if ndim not in (2, 3):
        raise ValueError('Dimension must be 2 or 3')
    if order not in (0, 1, 2):
        raise ValueError('Order must be 0, 1 or 2')

    if order == 0:
        return torch.ones(shape + [1], **backend)

    if isocenter is None:
        isocenter = [s / 2 for s in shape]
    isocenter = list(isocenter)
    isocenter += isocenter[-1:] * max(0, ndim-len(isocenter))

    ramps = identity_grid(shape, **backend)
    for i, ramp in enumerate(ramps.unbind(-1)):
        ramp -= isocenter[i]
        ramp /= shape[i] / 2

    if order == 1:
        return ramps
    # order == 2
    if ndim == 3:
        basis = [
            ramps[..., 0] * ramps[..., 1],
            ramps[..., 0] * ramps[..., 2],
            ramps[..., 1] * ramps[..., 2],
            ramps[..., 0].square() - ramps[..., 1].square(),
            ramps[..., 0].square() - ramps[..., 2].square()
        ]
        return torch.stack(basis, -1)
    else:  # basis == 2
        basis = [
            ramps[..., 0] * ramps[..., 1],
            ramps[..., 0].square() - ramps[..., 1].square()
        ]
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
    order : {0, 1, 2}, default=2
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
    ndim = len(shape)
    if ndim not in (2, 3):
        raise ValueError('Dimension must be 2 or 3')
    if order not in (0, 1, 2):
        raise ValueError('Order must be 0, 1 or 2')

    # zero-th order (global shift)
    yield torch.ones(shape, **backend)
    if order == 0:
        return

    # 1st order (linear ramps)
    if isocenter is None:
        isocenter = [s / 2 for s in shape]
    isocenter = list(isocenter)
    isocenter += isocenter[-1:] * max(0, ndim-len(isocenter))

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
    if ndim == 3:
        yield 2 * ramps[0] * ramps[1]
        yield 2 * ramps[0] * ramps[2]
        yield 2 * ramps[1] * ramps[2]
        yield ramps[0].square() - ramps[1].square()
        yield ramps[0].square() - ramps[2].square()
    else:
        assert ndim == 2
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
