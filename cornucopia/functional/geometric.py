__all__ = [
    "exp_velocity",
    "apply_flow",
    "apply_random_flow",
    "make_affine_matrix",
    "make_affine_flow",
    "apply_affine_matrix",
    "apply_affine",
]
from typing import Optional, Sequence
import torch
from ..baseutils import prepare_output, returns_update
from ..utils import warps, smart_math as math
from ..utils.py import ensure_list, make_vector
from ._utils import Tensor, Output, OneOrMore, Value, _backend_float
from .random import random_field_like
from .intensity import spline_upsample_like


def exp_velocity(
    input: Tensor,
    steps: int = 8,
    copy: bool = False,
    **kwargs
) -> Output:
    """
    Exponentiate a stationary velocity field (SVF) by squaring and scaling.

    Parameters
    ----------
    input : (C, *spatial, D) tensor
        Input velocity field
    steps : int
        Number of squaring and scaling steps
    copy : bool
        If steps == 0, force a copy.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input"}

    Returns
    -------
    output : (C, *spatial, D) tensor
        Exponentiated velocity field
    """
    if steps:
        output = warps.exp_velocity(input, steps)
    elif copy:
        output = input.clone()
    else:
        output = input
    return prepare_output(
        {"output": output, "input": input},
        kwargs.get("returns", "output")
    )


def apply_flow(input: Tensor, flow: Tensor, **kwargs) -> Output:
    """
    Apply a flow field to an image.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    flow : (C, *spatial, D) tensor
        Input flow field.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "flow"}

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.
    """
    has_identity = kwargs.get("has_identity", False)
    output = warps.apply_flow(
        input[:, None], flow, has_identity, padding_mode="border"
    )[:, 0]
    return prepare_output(
        {"output": output, "input": input, "flow": flow},
        kwargs.get("returns", "output")
    )


def apply_random_flow(
    input: Tensor,
    std: Optional[Value] = None,
    unit: str = "pct",
    shape: OneOrMore[int] = 5,
    steps: int = 0,
    order: int = 3,
    distrib: str = "uniform",
    shared: bool = True,
    zero_center: bool = False,
    **kwargs
) -> Output:
    """
    Apply a random flow field to an image.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    std : float | ([C],) tensor, default=0.06
        Standard deviation of the flow (or velocity) field.
    unit: {"vox", "pct"}
        Unit of the flow field (voxels or percent of field-of-view)
    shape : [list of] int
        Size of coarse tensor of spline coefficients.
    steps : int
        Number of integration steps.
    order : int
        Spline order.
    distrib : {"uniform", "gaussian", "generalized"}
        Probability distribution.
    shared : bool
        Apply the same flow field to all channels.
        If True, probability parameters must be scalars.
    zero_center : bool
        Subtract its mean displacement to the flow field so that
        it has an empirical mean of zero.

    Other Parameters
    ----------------
    peak, std, vmin, vmax, alpha, beta, mu, sigma : float | ([C],) tensor
        Other parameters of the probability distribution.
    returns : [list or dict of] {"output", "input", "coeff", "svf", "flow"}
        Values to return.

    Returns
    -------
    output : (C, *spatial) tensor

    """
    returns = kwargs.pop("returns", "output")

    ndim = input.ndim - 1
    C = len(input)
    CF = 1 if shared else C
    shape = [CF] + ensure_list(shape, ndim) + [ndim]

    if (
        (kwargs.get("mean", None) is None) and
        (kwargs.get("mu", None) is None) and
        (kwargs.get("peak", None) is None) and
        (kwargs.get("vmin", None) is None) and
        (kwargs.get("vmax", None) is None)
    ):
        kwargs["mean"] = 0

    if (
        (std is None) and
        (kwargs.get("sigma", None) is None) and
        (kwargs.get("std", None) is None) and
        (kwargs.get("vmin", None) is None) and
        (kwargs.get("vmax", None) is None) and
        (kwargs.get("alpha", None) is None)
    ):
        std = 0.06
    kwargs["std"] = std

    # sample spline coefficients
    coeff = random_field_like(distrib, shape, **kwargs)

    # rescale values
    if unit[0].lower() != "v":
        scale = make_vector(input.shape[1:]).to(coeff)
        coeff = math.mul_(coeff, scale)

    # upsample to image size
    svf = coeff.movedim(-1, 0).reshape((-1,) + coeff.shape[1:-1])
    svf = spline_upsample_like(svf, input, order, prefilter=False)
    svf = svf.reshape((ndim, CF) + input.shape[1:]).movedim(0, -1)

    # exponentiate
    flow = exp_velocity(svf, steps)

    # zero center
    if zero_center:
        mean_flow = flow.reshape([CF, -1, ndim]).mean(1)
        mean_flow = mean_flow.reshape([CF] + [1] * ndim + [ndim])
        flow = math.sub_(flow, mean_flow)

    # apply
    output = apply_flow(input, flow)

    return prepare_output({
        "output": output,
        "input": input,
        "flow": flow,
        "svf": svf,
        "coeff": coeff,
    }, returns)


def make_affine_matrix(
    translations: OneOrMore[Value],
    rotations: OneOrMore[Value],
    zooms: OneOrMore[Value],
    shears: OneOrMore[Value],
    **kwargs
) -> Output:
    """
    Build an affine matrix from its parameters.

    Parameters
    ----------
    translations : ([D],) (float | tensor)
    rotations : ([D*(D-1)/2],) (float | tensor)
    zooms : ([D],) (float | tensor)
    shears : ([D*(D-1)/2],) (float | tensor)

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", ...}
        Values to return.

    Returns
    -------
    matrix : (D+1, D+1) tensor

    """
    T_ = make_vector(translations)
    R_ = make_vector(rotations)
    Z_ = make_vector(zooms)
    S_ = make_vector(shears)

    backend = _backend_float(T_, R_, Z_, S_)

    # Guess dimensionality
    if len(T_) > 1:
        ndim = len(T_)
    elif len(Z_) > 1:
        ndim = len(Z_)
    elif len(R_) > 1:
        k = len(R_)
        ndim = int(round(((8 * k)**0.5 + 1)/2))
    elif len(S_) > 1:
        k = len(S_)
        ndim = int(round(((8 * k)**0.5 + 1)/2))
    else:
        ndim = kwargs["ndim"]
    ndim = kwargs.get("ndim", ndim)

    # Pad parameters
    # Default: zoom -> replicate, others -> zero

    Z_ = make_vector(Z_, ndim, **backend)
    T_ = make_vector(T_, ndim, **backend, default=0)
    S_ = make_vector(S_, ndim * (ndim - 1) // 2, **backend, default=0)
    R_ = make_vector(R_, ndim * (ndim - 1) // 2, **backend, default=0)
    R_ = R_ * (math.pi/180)

    # identity
    E = torch.eye(ndim+1, **backend)

    # zooms
    Z = E.clone()
    Z.diagonal(0, -1, -2)[:-1].copy_(1 + Z_)

    # translations
    T = E.clone()
    T[:ndim, -1] = T_

    if ndim == 2:

        # shear
        S = E.clone()
        S[0, 1] = S[1, 0] = S_[0]

        # rotation
        R = E.clone()
        R[0, 0] = R[1, 1] = R_[0].cos()
        R[0, 1] = R_[0].sin()
        R[1, 0] = -R[0, 1]

    elif ndim == 3:

        # shears
        Sz = E.clone()
        Sz[0, 1] = Sz[1, 0] = shears[0]
        Sy = E.clone()
        Sy[0, 2] = Sz[2, 0] = shears[1]
        Sx = E.clone()
        Sx[1, 2] = Sz[2, 1] = shears[2]
        S = Sx @ Sy @ Sz

        # rotations
        Rz = E.clone()
        Rz[0, 0] = Rz[1, 1] = rotations[0].cos()
        Rz[0, 1] = rotations[0].sin()
        Rz[1, 0] = -Rz[0, 1]
        Ry = E.clone()
        Ry[0, 0] = Ry[2, 2] = rotations[1].cos()
        Ry[0, 2] = rotations[1].sin()
        Ry[2, 0] = -Ry[0, 2]
        Rx = E.clone()
        Rx[1, 1] = Rx[2, 2] = rotations[2].cos()
        Rx[1, 2] = rotations[2].sin()
        Rx[2, 1] = -Rx[1, 2]
        R = Rx @ Ry @ Rz

    A = T @ R @ S @ Z
    return prepare_output({
        "output": A,
        "translations": translations,
        "rotations": rotations,
        "shears": shears,
        "zooms": zooms,
    }, kwargs.get("returns", "output"))


def make_affine_flow(
    matrix: Tensor,
    shape: Sequence[int],
    unit: str = "pct",
    **kwargs
) -> Output:
    """
    Convert an affine matrix to a flow field.

    Parameters
    ----------
    matrix : ([C], ndim+1, ndim+1) tensor
        Input affine matrix.
    shape : list[int]
        Spatial shape
    unit : {"vox", "pct"}
        Unit of the translation component.

    Returns
    -------
    flow : ([C], *spatial, ndim) tensor
        Flow field
    """
    ndim = input.shape[-1] - 1

    A = input.clone()

    # scale translation
    if unit[0].lower() != "v":
        A[..., :-1, -1] *= make_vector(shape).to(A)

    # apply transform at the center of the field of view
    offset = torch.as_tensor([(n-1)/2 for n in shape]).to(A)
    F = torch.eye(ndim+1).to(A)
    F[:-1, -1] = offset
    A = F.matmul(A).matmul(F.inverse())

    A = A.to(
        dtype=kwargs.get("dtype", None),
        device=kwargs.get("device", None)
    )

    # convert to flow field
    flow = warps.affine_flow(A, shape)

    return prepare_output(
        {"flow": flow, "output": flow, "matrix": matrix, "input": matrix},
        kwargs.get("returns", "output")
    )


def apply_affine_matrix(
    input: Tensor,
    matrix: Tensor,
    unit: str = "pct",
    **kwargs
) -> Output:
    """
    Apply an affine transformation encoded by a matrix.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    matrix : ([C], ndim+1, ndim+1) tensor
        Input affine matrix.
    unit : {"vox", "pct"}
        Unit of the translation component.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "flow", "matrix"}
        Values to return.

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor
    """
    dtype = input.dtype
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    backend = dict(dtype=dtype, device=input.device)

    # convert to flow field
    flow = make_affine_flow(matrix, input.shape[1:], unit, **backend)

    # apply flow field
    output = apply_flow(input, flow)

    return prepare_output({
        "output": output,
        "input": input,
        "matrix": matrix,
        "flow": flow,
    }, kwargs.get("returns", "output"))


def apply_affine(
    input: Tensor,
    translations: OneOrMore[Value],
    rotations: OneOrMore[Value],
    zooms: OneOrMore[Value],
    shears: OneOrMore[Value],
    unit: str = "pct",
    **kwargs
) -> Output:
    """
    Apply an affine transformation.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    translations : ([C, D]) (float | tensor)
        Translations.
    rotations : ([C, D*(D-1)/2]) (float | tensor)
        Rotations.
    zooms : ([C, D]) (float | tensor)
        Zooms.
    shears : ([C, D*(D-1)/2]) (float | tensor)
        Shears.
    unit : {"vox", "pct"}
        Unit of the translations.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "flow", "matrix", ...}
        Values to return.

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor
    """
    ndim = input.ndim - 1
    C = len(input)

    T = torch.as_tensor(translations).expand([C, ndim])
    R = torch.as_tensor(rotations).expand([C, (ndim*(ndim-1))//2])
    Z = torch.as_tensor(zooms).expand([C, ndim])
    S = torch.as_tensor(shears).expand([C, (ndim*(ndim-1))//2])

    # Build matrix
    matrix = torch.stack([
        make_affine_matrix(T1, R1, Z1, S1)
        for T1, R1, Z1, S1 in zip(T, R, Z, S)
    ])

    # Apply transform
    output = apply_affine_matrix(input, matrix, unit, **kwargs)

    returns = kwargs.get("returns", "output")
    output = returns_update(translations, "translations", output, returns)
    output = returns_update(rotations, "rotations", output, returns)
    output = returns_update(zooms, "zooms", output, returns)
    output = returns_update(shears, "shears", output, returns)
    return output


def apply_random_affine(
    input: Tensor,
    translations: 0.06,
    rotations: 9,
    zooms: 0.08,
    shears: 0.07,
    distrib: str = "uniform",
    distribz: str = "uniform",
    statistic: str = "std",
    unit: str = "pct",
    iso: bool = False,
    shared: bool = True,
    **kwargs,
) -> Output:
    """
    Apply a random affine transformation.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    translations : ([C],) (float | tensor)
        Scale of random translations.
    rotations : ([C],) (float | tensor)
        Scale of random rotations.
    zooms : ([C],) (float | tensor)
        Scale of random zooms.
    shears : ([C],) (float | tensor)
        Scale of random shears.
    distrib : [dict of] {"uniform", "gaussian"}
        Probability distribution over T/R/S (with mean 0).
    distribz : [dict of] {"uniform", "lognormal", "gamma"}
        Probability distribution over zooms (with mean 1).
    statistic : {"std", "fwhm"}
        Which statistics to use as "scale parameter".
    unit : {"vox", "pct"}
        Unit of the translations.
    shared : bool
        Apply the same transform to all channels.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "flow", "matrix", ...}
        Values to return.

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor
    """  # noqa: E501
    D = input.ndim - 1
    D2 = (D*(D-1))//2
    DZ = 1 if iso else D
    C = 1 if shared else len(input)

    T = random_field_like(distrib, input, [C, D], **{statistic: translations})
    R = random_field_like(distrib, input, [C, D2], **{statistic: rotations})
    Z = random_field_like(distribz, input, [C, DZ], **{statistic: zooms})
    S = random_field_like(distrib, input, [C, D2], **{statistic: shears})

    return apply_affine(input, T, R, Z, S, unit, **kwargs)
