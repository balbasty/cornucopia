__all__ = [
    "spline_upsample",
    "spline_upsample_like",
    "spline_sample",
    "spline_sample_coord",
]

# stdlib
from functools import partial
from typing import Sequence, Optional, Union

# external
import bounds as torch_bounds
import interpol
import torch
import torch.nn.functional as F

# internal
from ..baseutils import prepare_output, return_requires, returns_update
from ..utils.warps import apply_flow, sub_identity, add_identity
from ..utils.py import ensure_list, make_vector, meshgrid_ij
from ..utils.conv import smoothnd
from ._utils import Tensor, Output, OneOrMore


def spline_upsample(
    input: Tensor,
    factor: Optional[OneOrMore[float]] = None,
    *,
    shape: Optional[Sequence[int]] = None,
    order: OneOrMore[int] = 3,
    prefilter: bool = True,
    copy: bool = True,
    bound: OneOrMore[str] = "border",
    align: OneOrMore[str] = "center",
    recompute_factor: bool = True,
    backend: Optional[str] = None,
    **kwargs,
) -> Output:
    """
    Upsample a field of spline coefficients.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input spline coefficients (or values if `prefilter=True`).
    factor : [list of] float, optional
        Upsampling factor.

    Other Parameters
    ----------------
    shape : list[int], optional
        Target spatial shape. Required if `factor=None`.
    order : int
        Spline order.
    prefilter : bool
        If `False`, assume that the input contains spline coefficients,
        and returns the interpolated field.
        If `True`, assume that the input contains low-resolution values
        and convert them first to spline coefficients (= "prefilter"),
        before computing the interpolated field.
    copy : bool
        In cases where the output matches the input (the input and target
        shapes are identical, and no prefilter is required), the input
        tensor is returned when `copy=False`, and a copy is made when
        `copy=True`.
    bound : [list of] str
        Boundary condition used to interpolate/extrapolate out-of-bounds:

        * `{'zero', 'zeros'}` : All voxels outside of the FOV are zeros.
        * `{'border', 'nearest', 'replicate'} : Use nearest border value.
        * `{'mirror', 'dct1'}` : Reflect about the center of the border voxel.
          Equivalent to
          `grid_sample(..., padding_mode='reflection', align_corners=True)`.
        * `{'reflect', 'dct2'}` : Reflect about the edge of the border voxel.
          Equivalent to
          `grid_sample(..., padding_mode='reflection', align_corners=False)`.
        * `{'antimirror', 'dst1'}` : Negative reflection about the first
          out-of-bound voxel.
        * `{'antireflect', 'dst2'}` : Negative reflection about the edge of
          the border voxel.
        * `{'wrap', 'circular', 'dft'}` : Wrap the FOV.
        * `{'sliding'}` : Can only be used if the input tensor is a flow field.

        For more details, see the [`torch-bounds` documentation](
        https://torch-bounds.readthedocs.io/en/latest/api/types/).
    align : [list of] {"c[enter]", "e[dge]"}
        Whether the centers or the edges of the corner voxels should be
        aligned across resolutions.
    recompute_factor : bool
        Recompute the upsampling factor based on `align` and the effective
        input and output shapes. If `True`, backpropagation through
        `factor` is not possible.
    backend : {'torch', 'interpol'}, optional
        Backend to use. By default, the interpolation backend is used
        automatically based on the options selected. If `order` is
        in `{0, 1}`, and either `bound` is in `{'border', 'reflect'}`
        or `align='center'`, the `'torch'` backend is used (faster).
        Otherwise, the `'interpol'` backend is used (slower).
        If `backend='interpol'`, the interpol backend is always used.
        If `backend='torch'` and the chosen options are not supported
        by torch, an error is raised.
    returns : [list or dict of] {"output", "input", "coeff"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *shape) tensor
        Output tensor.
    """
    can_use_torch = True
    returns = kwargs.pop("returns", "output")
    bck = dict(dtype=input.dtype, device=input.device)
    if not input.dtype.is_floating_point:
        bck["dtype"] = torch.get_default_dtype()

    ndim = input.ndim - 1
    coeff = input

    # --- preprocess options and select backend ------------------------
    align_centers = align[:1].lower() != "c"

    bound = ensure_list(bound, ndim)
    bound = list(map(torch_bounds.to_fourier, bound))
    if len(set(bound)) > 1:
        can_use_torch = False
    if (
        not align_centers and
        any(x not in ("replicate", "reflect") for x in bound)
    ):
        can_use_torch = False

    order = ensure_list(order, ndim)
    if any(x > 1 for x in order):
        can_use_torch = False

    factor_is_one = False
    if factor is not None:
        factor = make_vector(factor, ndim, **bck)
        factor_is_one = (factor == 1).all()
        if factor.requires_grad:
            can_use_torch = False

    nothing_to_do = (
        (factor is None or factor_is_one) and
        (shape is None or (tuple(shape) == input.shape[1:])) and
        (prefilter or order <= 1)
    )
    need_prefilter = prefilter and (order > 1)

    if backend is None:
        backend = 'torch' if can_use_torch else 'interpol'
    if backend == 'torch' and not can_use_torch:
        raise ValueError(
            f'Cannot use torch interpolation backend with order={order}, '
            f'bound={bound}, align={align}.'
        )

    # --- Nothing to do ------------------------------------------------
    if nothing_to_do:
        output = input.clone() if copy else input
        if need_prefilter and ("coeff" in return_requires(returns)):
            coeff = interpol.spline_coeff_nd(input, order, dim=ndim)

    # --- Use torch.interpolate (faster) -------------------------------
    elif backend == "torch":
        mode = {3: "trilinear", 2: "bilinear", 1: "linear"}[len(shape)]
        if factor is not None:
            factor = factor.tolist()
        output = F.interpolate(
            input[None], shape,
            scale_factor=factor,
            mode=mode,
            align_corners=align_centers,
            recompute_scale_factor=recompute_factor,
        )[0]

    # --- Reimplement interpol :( --------------------------------------
    elif not recompute_factor:
        inshape = input.shape[1:]
        if prefilter:
            coeff = interpol.spline_coeff_nd(input, order, dim=ndim)
        if shape is None:
            if factor is None:
                raise ValueError("One of factor or shape must be provided.")
            shape = [int(i*f) for i, f in zip(inshape, factor)]
        if factor is None:
            if align_centers:
                factor = [(x - 1) / (y - 1) for x, y in zip(inshape, shape)]
            else:
                factor = [x / y for x, y in zip(inshape, shape)]
        lin = []
        for f, inshp, outshp in zip(factor, inshape, shape):
            shift = ((inshp - 1) - (outshp - 1) / f) * 0.5
            lin.append(torch.arange(0., outshp[0]) / f + shift)

        grid = torch.stack(meshgrid_ij(*lin), dim=-1)
        output = interpol.grid_pull(
            coeff, grid,
            bound=bound,
            interpolation=order,
            extrapolate=True,
            prefilter=False,
        )

    # --- Use interpol -------------------------------------------------
    else:
        if prefilter:
            coeff = interpol.spline_coeff_nd(input, order, dim=ndim)
        output = interpol.resize(
            coeff,
            factor=factor,
            shape=shape,
            interpolation=order,
            prefilter=False,
            anchor=align,
        )

    return prepare_output(
        {"input": input, "output": output, "coeff": coeff},
        returns
    )()


def spline_upsample_like(
    input: Tensor,
    like: Tensor,
    *,
    factor: Optional[OneOrMore[float]] = None,
    order: int = 3,
    prefilter: bool = True,
    copy: bool = True,
    bound: OneOrMore[str] = "border",
    align: OneOrMore[str] = "center",
    recompute_factor: bool = True,
    backend: Optional[str] = None,
    **kwargs
) -> Output:
    """
    Upsample a field of spline coefficients.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input spline coefficients (or values if `prefilter=True`)
    like : (C, *shape) tensor
        Target tensor.
    order : int
        Spline order
    prefilter : bool
        If `False`, assume that the input contains spline coefficients,
        and returns the interpolated field.
        If `True`, assume that the input contains low-resolution values
        and convert them first to spline coefficients (= "prefilter"),
        before computing the interpolated field.
    copy : bool
        In cases where the output matches the input (the input and target
        shapes are identical, and no prefilter is required), the input
        tensor is returned when `copy=False`, and a copy is made when
        `copy=True`.
    bound : [list of] str
        Boundary condition used to interpolate/extrapolate out-of-bounds:

        * `{'zero', 'zeros'}` : All voxels outside of the FOV are zeros.
        * `{'border', 'nearest', 'replicate'} : Use nearest border value.
        * `{'mirror', 'dct1'}` : Reflect about the center of the border voxel.
          Equivalent to
          `grid_sample(..., padding_mode='reflection', align_corners=True)`.
        * `{'reflect', 'dct2'}` : Reflect about the edge of the border voxel.
          Equivalent to
          `grid_sample(..., padding_mode='reflection', align_corners=False)`.
        * `{'antimirror', 'dst1'}` : Negative reflection about the first
          out-of-bound voxel.
        * `{'antireflect', 'dst2'}` : Negative reflection about the edge of
          the border voxel.
        * `{'wrap', 'circular', 'dft'}` : Wrap the FOV.
        * `{'sliding'}` : Can only be used if the input tensor is a flow field.

        For more details, see the [`torch-bounds` documentation](
        https://torch-bounds.readthedocs.io/en/latest/api/types/).
    align : [list of] {"c[enter]", "e[dge]"}
        Whether the centers or the edges of the corner voxels should be
        aligned across resolutions.
    recompute_factor : bool
        Recompute the upsampling factor based on `align` and the effective
        input and output shapes. If `True`, backpropagation through
        `factor` is not possible.
    backend : {'torch', 'interpol'}, optional
        Backend to use. By default, the interpolation backend is used
        automatically based on the options selected. If `order` is
        in `{0, 1}`, and either `bound` is in `{'border', 'reflect'}`
        or `align='center'`, the `'torch'` backend is used (faster).
        Otherwise, the `'interpol'` backend is used (slower).
        If `backend='interpol'`, the interpol backend is always used.
        If `backend='torch'` and the chosen options are not supported
        by torch, an error is raised.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "coeff", "like"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *shape) tensor
        Output tensor.

    """
    kwargs.setdefault("returns", "output")
    kwargs.setdefault("order", order)
    kwargs.setdefault("prefilter", prefilter)
    kwargs.setdefault("copy", copy)
    kwargs.setdefault("backend", backend)
    kwargs.setdefault("bound", bound)
    kwargs.setdefault("align", align)
    kwargs.setdefault("recompute_factor", recompute_factor)
    kwargs.setdefault("factor", factor)
    output = spline_upsample(input, shape=like.shape[1:], **kwargs)
    output = returns_update(like, "like", output, kwargs["returns"])
    return output()


def spline_downsample(
    input: Tensor,
    factor: OneOrMore[float] = 1,
    antialiasing: Union[bool, OneOrMore[float]] = True,
    bound: OneOrMore[str] = 'reflect',
    order: OneOrMore[int] = 1,
    shape: Optional[list[int]] = None,
    align: OneOrMore[str] = 'edge',
    **kwargs
) -> Output:
    """
    Downsample an image by some factor.
    """
    returns = kwargs.pop("returns", "output")

    ndim = input.ndim - 1
    factor = make_vector(factor, ndim)
    bound = ensure_list(bound, ndim)
    if not torch.is_tensor(antialiasing):
        antialiasing = ensure_list(antialiasing, ndim)
        antialiasing = [
            factor if x is True else
            0 if x is False else
            x for x in antialiasing
        ]

    if sum(antialiasing) > 0:
        smoothed = smoothnd(input, fwhm=antialiasing, bound=bound)
    else:
        smoothed = input

    output = spline_upsample(
        smoothed,
        factor=factor,
        shape=shape,
        order=order,
        prefilter=True,
        bound=bound,
        align=align,
        recompute_factor=not factor.requires_grad,
    )

    return prepare_output(
        {"input": input, "output": output, "smoothed": smoothed},
        returns
    )()


def spline_sample(
    input: Tensor,
    flow: Tensor,
    has_identity: bool = False,
    order: OneOrMore[int] = 1,
    bound: OneOrMore[str] = "border",
    extrapolate: bool = True,
    prefilter: bool = True,
    backend: Optional[str] = None,
    nearest_if_label: bool = False,
    **kwargs,
) -> Output:
    """
    Sample an image at locations encoded by a deformation field.

    Parameters
    ----------
    input : (C, *ishape) tensor
        Input tensor to sample.
    flow : (D, *oshape) tensor
        Displacement (or coordinate) field.
    has_identity : bool
        * If `True`, the `flow` field contains absolute voxel coordinates.
        * If `False`, the `flow` field contains relative voxel coordinates
          (_i.e._, it is a displacement field).
    order : [list of] {0..7}
        Spline order (per spatial dimension).
    bound : [list of] str
        Boundary condition used to interpolate/extrapolate out-of-bounds:

        * `{'zero', 'zeros'}` : All voxels outside of the FOV are zeros.
        * `{'border', 'nearest', 'replicate'} : Use nearest border value.
        * `{'mirror', 'dct1'}` : Reflect about the center of the border voxel.
          Equivalent to
          `grid_sample(..., padding_mode='reflection', align_corners=True)`.
        * `{'reflect', 'dct2'}` : Reflect about the edge of the border voxel.
          Equivalent to
          `grid_sample(..., padding_mode='reflection', align_corners=False)`.
        * `{'antimirror', 'dst1'}` : Negative reflection about the first
          out-of-bound voxel.
        * `{'antireflect', 'dst2'}` : Negative reflection about the edge of
          the border voxel.
        * `{'wrap', 'circular', 'dft'}` : Wrap the FOV.
        * `{'sliding'}` : Can only be used if the input tensor is a flow field.

        For more details, see the [`torch-bounds` documentation](
        https://torch-bounds.readthedocs.io/en/latest/api/types/).
    extrapolate : bool
        Whether to use boundary condition to extrapolate out-of-bound samples.
        If `False`, boundary conditions are only used to interpolate in-bound
        samples.
    prefilter : bool
        Whether to apply a spline prefilter to convert the input values
        into spline coefficients. This ensures that this function
        exactly interpolates the input tensor. This is equivalent to
        `scipy.ndimage.map_coordinates(..., prefilter=True)`.
        This has no effect is `order` is zero or one.
    backend : {'torch', 'interpol'}, optional
        Backend to use. By default, the interpolation backend is used
        automatically based on the options selected. If `order` is
        in `{0, 1}`, `bound` is in `{'zero', 'border', 'mirror', 'reflect'}`,
        and `extrapolate` is True, the `'torch'` backend is used (faster).
        Otherwise, the `'interpol'` backend is used (slower).
        If `backend='interpol'`, the interpol backend is always used.
        If `backend='torch'` and the chosen options are not supported
        by torch, an error is raised.
    nearest_if_label : bool
        By default, if a tensor has an integer data type, it is deformed
        using label-specific resampling (each unique label is extracted
        and resampled using linear interpolation, and an argmax output
        label map is computed on the fly).
        If `nearest_if_label=True`, the entire label map will be
        resampled at once using nearest-neighbour interpolation.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "coeff", "flow", "disp", "coord"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *oshape) tensor
        Output tensor.
    """  # noqa: E501
    returns = kwargs.pop("returns", "output")
    can_use_torch = True

    # --- preprocess options and select backend ------------------------

    bound = ensure_list(bound)
    bound = list(map(torch_bounds.to_fourier))
    if not len(set(bound)) != 1:
        can_use_torch = False
    if any(x in ('dst1', 'dst2', 'dft') for x in bound):
        can_use_torch = False

    order = ensure_list(order)
    if not len(set(order)) != 1:
        can_use_torch = False
    if any(x > 1 for x in order):
        can_use_torch = False

    if not extrapolate and any(x != 'zero' for x in bound):
        can_use_torch = False

    if backend is None:
        backend = 'torch' if can_use_torch else 'interpol'
    if backend == 'torch' and not can_use_torch:
        raise ValueError(
            f'Cannot use torch interpolation backend with order={order}, '
            f'bound={bound}, extrapolate={extrapolate}.'
        )

    disp = coord = None

    # --- torch backend ------------------------------------------------
    if backend == 'torch':
        if has_identity:
            coord = flow
            disp = sub_identity(flow.movedim(0, -1)).movedim(-1, 0)
        else:
            disp = flow
            if return_requires(returns, "coord"):
                coord = add_identity(flow.movedim(0, -1)).movedim(-1, 0)

        bound, align = torch_bounds.to_torch(bound[0])
        order = 'nearest' if order == 0 else 'bilinear'
        output = apply_flow(
            input, disp.movedim(0, -1),
            mode=order,
            padding_mode=bound,
            align_corners=align,
        )

    # --- interpol backend ---------------------------------------------
    else:
        if has_identity:
            coord = flow
            if return_requires(returns, "disp"):
                disp = sub_identity(flow.movedim(0, -1)).movedim(-1, 0)
        else:
            disp = flow
            coord = add_identity(flow.movedim(0, -1)).movedim(-1, 0)

        if bound == ["sliding"]:

            if len(input) != len(flow) or not input.dtype.is_floating_point:
                raise ValueError(
                    "Sliding boundary condition is only supported for "
                    "flow fields."
                )

            output = input.new_zeros((len(input),) + flow.shape[1:])
            bound0 = ["dct2"] * len(flow)
            for i, channel in enumerate(input):
                bound = list(bound0)
                bound[i] = "dst2"
                output[i] = interpol.grid_pull(
                    channel[None], coord.movedim(0, -1),
                    interpolation=order,
                    bound=bound,
                    extrapolate=extrapolate,
                    prefilter=prefilter,
                ).squeeze(0)

        if input.dtype.is_floating_point:

            output = interpol.grid_pull(
                input, coord.movedim(0, -1),
                interpolation=order,
                bound=bound,
                extrapolate=extrapolate,
                prefilter=prefilter,
            )

        elif nearest_if_label:

            dtype = input.dtype
            input = input.to(torch.get_default_dtype())
            output = interpol.grid_pull(
                input, coord.movedim(0, -1),
                interpolation='nearest',
                bound=bound,
                extrapolate=extrapolate,
                prefilter=prefilter,
            ).to(dtype)

        else:

            output = input.new_zeros((len(input),) + flow.shape[1:])
            prob = torch.zeros_like(output, dtype=flow.dtype)
            for label in torch.unique(input):
                prob1 = (input == label).to(torch.get_default_dtype())
                prob1 = interpol.grid_pull(
                    prob1, coord.movedim(0, -1),
                    interpolation=order,
                    bound=bound,
                    extrapolate=extrapolate,
                    prefilter=prefilter,
                )
                output.masked_fill_(prob1 > prob, label)
                prob.clamp_min_(prob1)

    return prepare_output(
        {"input": input, "output": output, "flow": flow, "coord": coord,
         "disp": disp},
        returns
    )()


spline_sample_coord = partial(spline_sample, has_identity=True)
