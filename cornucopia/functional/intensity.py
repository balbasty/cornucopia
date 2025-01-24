__all__ = [
    "add_value",
    "sub_value",
    "mul_value",
    "div_value",
    "addmul_value",
    "fill_value",
    "clip_value",
    "add_field",
    "sub_field",
    "mul_field",
    "div_field",
    "spline_upsample",
    "spline_upsample_like",
    "gamma_transform",
    "z_transform",
    "quantile_transform",
    "minmax_transform",
    "affine_intensity_transform",
    "add_smooth_random_field",
    "mul_smooth_random_field",
]
# stdlib
from typing import Sequence, Optional, Callable

# external
import torch
import interpol
import torch.nn.functional as F

# internal
from ..baseutils import prepare_output, returns_update, return_requires
from ..utils.smart_math import add_, mul_, pow_, div_
from ..utils.py import ensure_list
from ._utils import _unsqz_spatial, Tensor, Value, Output, OneOrMore
from .random import random_field_like


def binop_value(
    op: Callable[[Tensor, Value], Output],
    input: Tensor,
    value: Value,
    **kwargs
) -> Output:
    """
    Add a value to the input.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    value : float | ([C],) tensor
        Input value.
        It can have multiple channels but no spatial dimensions.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "value"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    output = op(input, _unsqz_spatial(value, input.ndim - 1))
    kwargs.setdefault("value_name", "value")
    kwargs.setdefault("returns", "output")
    return prepare_output(
        {"input": input, "output": output, kwargs["value"]: value},
        kwargs["returns"]
    )


def add_value(input: Tensor, value: Value, **kwargs) -> Output:
    """
    Add a value to the input.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    value : float | ([C],) tensor
        Input value.
        It can have multiple channels but no spatial dimensions.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "value"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    return binop_value(torch.add, input, value, **kwargs)


def sub_value(input: Tensor, value: Value, **kwargs) -> Output:
    """
    Subtract a value to the input.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    value : float | ([C],) tensor
        Input value.
        It can have multiple channels but no spatial dimensions.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "value"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    return binop_value(torch.sub, input, value, **kwargs)


def mul_value(input: Tensor, value: Value, **kwargs) -> Output:
    """
    Multiply the input with a value.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    value : float | ([C],) tensor
        Input value.
        It can have multiple channels but no spatial dimensions.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "value"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    return binop_value(torch.mul, input, value, **kwargs)


def div_value(input: Tensor, value: Value, **kwargs) -> Output:
    """
    Divide the input by a value.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    value : float | ([C],) tensor
        Input value.
        It can have multiple channels but no spatial dimensions.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "value"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    return binop_value(torch.div, input, value, **kwargs)


def addmul_value(
    input: Tensor, scale: Value, offset: Value, **kwargs
) -> Output:
    """
    Affine transform of the input values: `output = input * scale + offset`.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    scale : float | ([C],) tensor
        Input scale.
        It can have multiple channels but no spatial dimensions.
    offset : float | ([C],) tensor
        Input offset.
        It can have multiple channels but no spatial dimensions.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "scale", "offset"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    output = (
        input *
        _unsqz_spatial(scale, input.ndim - 1) +
        _unsqz_spatial(offset, input.ndim - 1)
    )
    kwargs.setdefault("scale_name", "scale")
    kwargs.setdefault("offset_name", "offset")
    kwargs.setdefault("returns", "output")
    return prepare_output({
        "input": input,
        "output": output,
        kwargs["scale_name"]: scale,
        kwargs["offset_name"]: offset,
    }, kwargs["returns"])


def binop_field(
    op: Callable[[Tensor, Tensor], Output],
    input: Tensor,
    field: Tensor,
    order: int = 3,
    prefilter: bool = True,
    **kwargs
) -> Output:
    """
    Apply a binary operation between the input and a field.

    The field gets resized to the input"s shape if needed.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    field : ([C], *sptial) tensor
        Input field. It must have spatial dimensions.
    order : int
        Spline order, if the field needs to be upsampled.
    prefilter : bool
        If `False`, assume that the input contains spline coefficients,
        and returns the interpolated field.
        If `True`, assume that the input contains low-resolution values
        and convert them first to spline coefficients (= "prefilter"),
        before computing the interpolated field.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "field", "input_field"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    # NOTE: if `field` already has the correct size and does not contain
    #       spline coefficients, `spline_upsample_like` does nothing
    #       and returns the input field as is.
    input_field = field
    field = spline_upsample_like(field, input, order, prefilter, copy=False)
    output = op(input, field)

    kwargs.setdefault("field_name", "field")
    kwargs.setdefault("returns", "output")
    return prepare_output({
        "input": input,
        "output": output,
        kwargs["field_name"]: field,
        "input_" + kwargs["field_name"]: input_field
    }, kwargs["returns"])


def add_field(
    input: Tensor,
    field: Tensor,
    order: int = 3,
    prefilter: bool = True,
    **kwargs
) -> Output:
    """
    Add a field to the input.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    field : ([C], *sptial) tensor
        Input field. It must have spatial dimensions.
    order : int
        Spline order, if the field needs to be upsampled.
    prefilter : bool
        If `False`, assume that the input contains spline coefficients,
        and returns the interpolated field.
        If `True`, assume that the input contains low-resolution values
        and convert them first to spline coefficients (= "prefilter"),
        before computing the interpolated field.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "field", "input_field"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    return binop_field(torch.add, input, field, order, prefilter, **kwargs)


def sub_field(
    input: Tensor,
    field: Tensor,
    order: int = 3,
    prefilter: bool = True,
    **kwargs
) -> Output:
    """
    Subtract a field to the input.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    field : ([C], *sptial) tensor
        Input field. It must have spatial dimensions.
    order : int
        Spline order, if the field needs to be upsampled.
    prefilter : bool
        If `False`, assume that the input contains spline coefficients,
        and returns the interpolated field.
        If `True`, assume that the input contains low-resolution values
        and convert them first to spline coefficients (= "prefilter"),
        before computing the interpolated field.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "field", "input_field"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    return binop_field(torch.sub, input, field, order, prefilter, **kwargs)


def mul_field(
    input: Tensor,
    field: Tensor,
    order: int = 3,
    prefilter: bool = True,
    **kwargs
) -> Output:
    """
    Multiply athe inout with a field.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    field : ([C], *sptial) tensor
        Input field. It must have spatial dimensions.
    order : int
        Spline order, if the field needs to be upsampled.
    prefilter : bool
        If `False`, assume that the input contains spline coefficients,
        and returns the interpolated field.
        If `True`, assume that the input contains low-resolution values
        and convert them first to spline coefficients (= "prefilter"),
        before computing the interpolated field.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "field", "input_field"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    return binop_field(torch.mul, input, field, order, prefilter, **kwargs)


def div_field(
    input: Tensor,
    field: Tensor,
    order: int = 3,
    prefilter: bool = True,
    **kwargs
) -> Output:
    """
    Divide the input by a field.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    field : ([C], *sptial) tensor
        Input field. It must have spatial dimensions.
    order : int
        Spline order, if the field needs to be upsampled.
    prefilter : bool
        If `False`, assume that the input contains spline coefficients,
        and returns the interpolated field.
        If `True`, assume that the input contains low-resolution values
        and convert them first to spline coefficients (= "prefilter"),
        before computing the interpolated field.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "field", "input_field"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    return binop_field(torch.div, input, field, order, prefilter, **kwargs)


def fill_value(input: Tensor, mask: Tensor, value: Value, **kwargs) -> Output:
    """
    Set a value at masked locations.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    mask : ([C], *spatial) tensor
        Input mask.
    value : float | ([C],) tensor
        Input value.
        If `mask` has a channel dimension, must be a scalar.
        Otherwise, can be a vetor of length `C`.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "mask", "value"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    # Multiple value case -- must fill one channel at a time.
    if torch.is_tensor(value) and len(value) > 1:

        # Checks
        if mask.ndim == input.ndim and len(mask) > 1:
            raise ValueError(
                "If mask has a channel dimension, value must be a scalar."
            )
        if len(value) != len(input):
            raise ValueError(
                "Number of values does not match the number of channels."
            )
        if mask.ndim == input.ndim:
            mask_nochannel = mask.squeeze(0)
        else:
            mask_nochannel = mask

        # Fill per channel
        output = input.clone()
        for c in range(len(input)):
            output[c].masked_fill_(mask_nochannel, value[c])

    # Single value case -- can use `masked_fill`` out-of-the-box
    else:
        output = input.masked_fill(mask, value)

    kwargs.setdefault("value_name", "value")
    kwargs.setdefault("mask_name", "mask")
    kwargs.setdefault("returns", "output")
    return prepare_output({
        "input": input,
        "output": output,
        kwargs["value_name"]: value,
        kwargs["mask_name"]: mask,
    }, kwargs["returns"])


def clip_value(
    input: Tensor,
    vmin: Optional[Value] = None,
    vmax: Optional[Value] = None,
    **kwargs,
) -> Output:
    """
    Clip extreme values.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    vmin : float | ([C],) tensor
        Minimum value.
        It can have multiple channels but no spatial dimensions.
    vmax : float | ([C],) tensor
        Maximum value.
        It can have multiple channels but no spatial dimensions.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "vmin", "vmax"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.

    """
    ndim = input.ndim - 1
    output = input.clip(_unsqz_spatial(vmin, ndim), _unsqz_spatial(vmax, ndim))
    kwargs.setdefault("returns", "output")
    return prepare_output(
        {"input": input, "output": output, "vmin": vmin, "vmax": vmax},
        kwargs["returns"]
    )


def spline_upsample(
    input: Tensor,
    shape: Sequence[int],
    order: int = 3,
    prefilter: bool = True,
    copy: bool = True,
    **kwargs
) -> Output:
    """
    Upsample a field of spline coefficients.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input spline coefficients (or values if `prefilter=True`)
    shape : list[int]
        Target spatial shape
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

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "coeff"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *shape) tensor
        Output tensor.
    """
    returns = kwargs.pop("returns", "output")

    ndim = input.ndim - 1
    coeff = input

    same_shape = (tuple(shape) == input.shape[1:])
    nothing_to_do = same_shape and (prefilter or order <= 1)
    need_prefilter = prefilter and (order > 1)

    # 1) Nothing to do
    if nothing_to_do:
        output = input.clone() if copy else input
        if need_prefilter and ("coeff" in return_requires(returns)):
            coeff = interpol.spline_coeff_nd(input, order, dim=ndim)

    # 2) Use torch.inteprolate (faster)
    elif order == 1:
        mode = ("trilinear" if len(shape) == 3 else
                "bilinear" if len(shape) == 2 else
                "linear")
        output = F.interpolate(
            input[None], shape, mode=mode, align_corners=True
        )[0]

    # 3) Use interpol
    else:
        if prefilter:
            coeff = interpol.spline_coeff_nd(input, order, dim=ndim)
        output = interpol.resize(
            coeff, shape=shape, interpolation=order, prefilter=False
        )

    return prepare_output(
        {"input": input, "output": output, "coeff": coeff},
        returns
    )


def spline_upsample_like(
    input: Tensor,
    like: Tensor,
    order: int = 3,
    prefilter: bool = True,
    copy: bool = True,
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
    output = spline_upsample(input, like.shape[1:], **kwargs)
    output = returns_update(like, "like", output, kwargs["returns"])


def gamma_transform(
    input: Tensor,
    gamma: Value = 1,
    vmin: Optional[Value] = None,
    vmax: Optional[Value] = None,
    per_channel: bool = False,
    **kwargs
) -> Output:
    """
    Apply a Gamma transformation:

    ```python
    rscled = (input - vmin) / (vmax - vmin)
    xfrmed = rscled ** gamma
    output = xfrmed * (vmax - vmin) + vmin
    ```

    Parameters
    ----------
    input : tensor
        Input tensor.
    gamma : float | ([C],) tensor
        Gamma coefficient.
        It can have multiple channels but no spatial dimensions.
    vmin : float | ([C],) tensor | None
        Minimum value.
        It can have multiple channels but no spatial dimensions.
        If `None`, compute the input"s minimum.
    vmax : float | ([C],) tensor | None
        Maximum value.
        It can have multiple channels but no spatial dimensions.
        If `None`, compute the input"s maximum.
    per_channel : bool
        This parameter is only used when `vmin=None` or `vmax=None`.
        If `True`, the min/max of each input channel is used.
        If `False, the global min/max of the input tensor is used.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "gamma", "vmin", "vmax"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *shape) tensor
        Output tensor.
    """
    ndim = input.ndim - 1

    if vmin is None:
        if per_channel:
            vmin = input.reshape([len(input), -1]).min(-1).values
        else:
            vmin = input.min()

    if vmax is None:
        if per_channel:
            vmax = input.reshape([len(input), -1]).max(-1).values
        else:
            vmax = input.max()

    vmin_ = _unsqz_spatial(vmin, ndim)
    vmax_ = _unsqz_spatial(vmax, ndim)
    gamma_ = _unsqz_spatial(gamma, ndim)

    output = div_((input - vmin_), (vmax_ - vmin_).clamp_min_(1e-8))
    output = pow_(output, gamma_)
    if getattr(gamma_, "requires_grad", False):
        # When gamma requires grad,  mul_(y, vmax-vmin) is happy
        # to overwrite y, but we cant because we need y to
        # backprop through pow. So we need an explicit branch.
        output = output * (vmax_ - vmin_) + vmin_
    else:
        output = add_(mul_(output, vmax_ - vmin_), vmin_)

    kwargs.setdefault("returns", "output")
    return prepare_output({
        "input": input,
        "output": output,
        "vmin": vmin,
        "vmax": vmax,
        "gamma": gamma,
    }, kwargs["returns"])


def z_transform(
    input: Tensor,
    mu: Value = 0,
    sigma: Value = 1,
    per_channel: bool = False,
    **kwargs
) -> Output:
    """
    Apply a Z transformation:

    ```python
    output = ((input - mean(input)) / std(input)) * sigma + mu
    ```

    Parameters
    ----------
    input : tensor
        Input tensor.
    mu : float | ([C],) tensor
        Target mean.
        It can have multiple channels but no spatial dimensions.
    sigma : float | ([C],) tensor
        Target standard deviation.
        It can have multiple channels but no spatial dimensions.
    per_channel : bool
        If `True`, compute the mean/std of each input channel.
        If `False, the global mean/std of the input tensor is used.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "mu", "sigma"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *shape) tensor
        Output tensor.
    """
    ndim = input.ndim - 1

    if per_channel:
        mu0 = input.reshape([len(input), -1]).mean(-1)
    else:
        mu0 = input.mean()

    if per_channel:
        sigma0 = input.reshape([len(input), -1]).std(-1)
    else:
        sigma0 = input.std()

    mu0 = _unsqz_spatial(mu0, ndim)
    sigma0 = _unsqz_spatial(sigma0, ndim)
    mu_ = _unsqz_spatial(mu, ndim)
    sigma_ = _unsqz_spatial(sigma, ndim)

    output = div_((input - mu0), sigma0.clamp_min_(1e-8))
    output = add_(mul_(input, mu_), sigma_)

    kwargs.setdefault("returns", "output")
    return prepare_output({
        "input": input,
        "output": output,
        "mu": mu,
        "sigma": sigma,
    }, kwargs["returns"])


def quantile_transform(
    input: Tensor,
    pmin: Value = 0.01,
    pmax: Value = 0.99,
    vmin: Value = 0,
    vmax: Value = 1,
    per_channel: bool = False,
    max_samples: Optional[int] = 10000,
    **kwargs
) -> Output:
    """
    Apply a quantile transformation:

    ```python
    qmin = quantile(input, pmin)
    qmax = quantile(input, pmax)
    rscled = (input - pmin) / (pmax - pmin)
    output = rscled * (vmax - vmin) + vmin
    ```

    Parameters
    ----------
    input : tensor
        Input tensor.
    pmin : float | ([C],) tensor
        Lower quantile.
        It can have multiple channels but no spatial dimensions.
    pmax : float | ([C],) tensor
        Upper quantile.
        It can have multiple channels but no spatial dimensions.
    vmin : float | ([C],) tensor
        Minimum output value.
        It can have multiple channels but no spatial dimensions.
    vmax : float | ([C],) tensor
        Maximum output value.
        It can have multiple channels but no spatial dimensions.
    per_channel : bool
        This parameter is only used when `vmin=None` or `vmax=None`.
        If `True`, the qmin/qmax of each input channel is used.
        If `False, the global qmin/qmax of the input tensor is used.
    max_samples : int | None
        Maximum number of samples to use to estimate quantiles.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "pmin", "pmax", "qmin", "qmax", "vmin", "vmax"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *shape) tensor
        Output tensor.

    """  # noqa: E501
    ndim = input.ndim - 1
    C = len(input)

    # Select a subset of values to compute the quantiles
    # (discard inf/nan/zeros + take random sample for speed)
    input_ = input.reshape([len(input), -1])
    input_ = input_[:, (input_ != 0).all(0) & input_.isfinite().all(0)]
    if (max_samples is not None) and (max_samples < input_.shape[1]):
        index_ = torch.randperm(input_.shape[1], device=input_.device)
        index_ = index_[:max_samples]
        input_ = input_[:, index_]

    # Compute lower quantile
    pmin_ = pmin
    if torch.is_tensor(pmin_) and pmin_.shape:
        pmin_ = torch.expand(pmin_, [len(input)])
        qmin = torch.stack([
            torch.quantile(input[c], pmin_[c]) for c in range(C)
        ])
    else:
        qdim = (-1 if per_channel else None)
        qmin = torch.quantile(input_, pmin_, dim=qdim)

    # Compute upper quantile
    pmax_ = pmax
    if torch.is_tensor(pmax_) and pmax_.shape:
        pmax_ = torch.expand(pmax_, [len(input)])
        qmax = torch.stack([
            torch.quantile(input[c], pmax_[c]) for c in range(C)
        ])
    else:
        qdim = (-1 if per_channel else None)
        qmax = torch.quantile(input_, pmin_, dim=qdim)

    qmin_ = _unsqz_spatial(qmin, ndim)
    qmax_ = _unsqz_spatial(qmax, ndim)
    vmin_ = _unsqz_spatial(vmin, ndim)
    vmax_ = _unsqz_spatial(vmax, ndim)

    # Transform
    output = div_((input - qmin_), (qmax_ - qmin_).clamp_min_(1e-8))
    output = add_(mul_(output, vmax_ - vmin_), vmin_)

    kwargs.setdefault("returns", "output")
    return prepare_output({
        "input": input,
        "output": output,
        "vmin": vmin,
        "vmax": vmax,
        "pmin": pmin,
        "pmax": pmax,
        "qmin": qmin,
        "qmax": qmax,
    }, kwargs["returns"])


def minmax_transform(
    input: Tensor,
    vmin: Value = 0,
    vmax: Value = 1,
    per_channel: bool = False,
    **kwargs
) -> Output:
    """
    Apply a min-max transformation:

    ```python
    rscled = (input - input.mean()) / (input.max() - input.min())
    output = rscled * (vmax - vmin) + vmin
    ```

    Parameters
    ----------
    input : tensor
        Input tensor.
    vmin : float | ([C],) tensor
        Minimum output value.
        It can have multiple channels but no spatial dimensions.
    vmax : float | ([C],) tensor
        Maximum output value.
        It can have multiple channels but no spatial dimensions.
    per_channel : bool
        This parameter is only used when `vmin=None` or `vmax=None`.
        If `True`, the qmin/qmax of each input channel is used.
        If `False, the global qmin/qmax of the input tensor is used.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "imin", "imax", "vmin", "vmax"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *shape) tensor
        Output tensor.

    """  # noqa: E501
    ndim = input.ndim - 1
    C = len(input)

    # Compute min/max
    if per_channel:
        imin, imax = [], []
        for c in range(C):
            input_ = input[c]
            input_ = input_[input_.isfinite()]
            imin.append(input_.min())
            imax.append(input_.max())
        imin = torch.stack(imin)
        imax = torch.stack(imax)
    else:
        imin = input_.min()
        imax = input_.max()

    imin_ = _unsqz_spatial(imin, ndim)
    imax_ = _unsqz_spatial(imax, ndim)
    vmin_ = _unsqz_spatial(vmin, ndim)
    vmax_ = _unsqz_spatial(vmax, ndim)

    # Transform
    output = div_((input - imin_), (imax_ - imin_).clamp_min_(1e-8))
    output = add_(mul_(output, vmax_ - vmin_), vmin_)

    kwargs.setdefault("returns", "output")
    return prepare_output({
        "input": input,
        "output": output,
        "vmin": vmin,
        "vmax": vmax,
        "imin": imin,
        "imax": imax,
    }, kwargs["returns"])


def affine_intensity_transform(
    input: Tensor,
    imin: Value,
    imax: Value,
    omin: Value = 0,
    omax: Value = 1,
    clip: bool = False,
    **kwargs
) -> Output:
    """
    Apply an affine transform that maps pairs of values:

    ```python
    rscled = (input - imin) / (imax - imin)
    output = rscled * (omax - omin) + omin
    ```

    Parameters
    ----------
    input : tensor
        Input tensor.
    imin : float | ([C],) tensor
        Minimum input value.
        It can have multiple channels but no spatial dimensions.
    imax : float | ([C],) tensor
        Maximum input value.
        It can have multiple channels but no spatial dimensions.
    omin : float | ([C],) tensor
        Minimum output value.
        It can have multiple channels but no spatial dimensions.
    omax : float | ([C],) tensor
        Maximum output value.
        It can have multiple channels but no spatial dimensions.
    clip : bool
        Clip values outside of the range.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "imin", "imax", "omin", "omax"}
        Structure of variables to return. Default: "output".

    Returns
    -------
    output : (C, *shape) tensor
        Output tensor.

    """  # noqa: E501
    ndim = input.ndim - 1

    imin_ = _unsqz_spatial(imin, ndim)
    imax_ = _unsqz_spatial(imax, ndim)
    omin_ = _unsqz_spatial(omin, ndim)
    omax_ = _unsqz_spatial(imax, ndim)

    # Transform
    output = div_((input - imin_), (imax_ - imin_).clamp_min_(1e-8))
    output = add_(mul_(output, omax_ - omin_), omin_)

    if clip:
        output = output.clip_(omin, omax)

    kwargs.setdefault("returns", "output")
    return prepare_output({
        "input": input,
        "output": output,
        "imin": imin,
        "imax": imax,
        "omin": omin,
        "omax": omax,
    }, kwargs["returns"])


def add_smooth_random_field(
    input: Tensor,
    shape: OneOrMore[int] = 8,
    mean: Optional[Value] = None,
    fwhm: Optional[Value] = None,
    distrib: str = "uniform",
    order: int = 3,
    shared: bool = False,
    **kwargs
) -> Output:
    """
    Add a smooth random field to the input.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor
    shape : int | list[int]
        Shape of the low-resolution spline coefficients (lower = smoother)
    mean : float | ([C],) tensor, default=0
        Distribution mean.
    fwhm : float | ([C],) tensor, default=1
        Distribution full width at half-maximum.
    distrib : {"uniform", "gaussian", "generalized"}
        Probability distribution.
    order : int
        Spline order
    shared : bool
        Apply the same field to all channels.
        If True, probability parameters must be scalars.

    Other Parameters
    ----------------
    peak, std, vmin, vmax, alpha, beta : float | ([C],) tensor
        Other parameters of the probability distribution.
    returns : [list or dict of] {"output", "input", "coeff", "field"}
        Values to return.

    Returns
    -------
    output : (C, *spatial) tensor

    """  # noqa: E501
    if (
        (mean is None) and
        (kwargs.get("mu", None) is not None) and
        (kwargs.get("peak", None) is not None) and
        (kwargs.get("vmin", None) is not None) and
        (kwargs.get("vmax", None) is not None)
    ):
        mean = 0

    if (
        (fwhm is None) and
        (kwargs.get("sigma", None) is not None) and
        (kwargs.get("std", None) is not None) and
        (kwargs.get("vmin", None) is not None) and
        (kwargs.get("vmax", None) is not None) and
        (kwargs.get("alpha", None) is not None)
    ):
        fwhm = 1

    ndim = input.ndim - 1
    shape = tuple(ensure_list(shape, ndim))
    if shared:
        shape = (1,) + shape
    else:
        shape = input.shape[:-1] + shape

    returns = kwargs.pop("returns", "output")
    kwargs["mean"] = mean
    kwargs["fwhm"] = fwhm
    coeff = random_field_like(distrib, input, shape, **kwargs)
    output = add_field(input, coeff, order, prefilter=False, returns=returns)

    output = returns_update(coeff, "coeff", output, returns)
    return output


def mul_smooth_random_field(
    input: Tensor,
    shape: OneOrMore[int] = 8,
    mean: Optional[Value] = None,
    fwhm: Optional[Value] = None,
    distrib: str = "uniform",
    order: int = 3,
    shared: bool = False,
    **kwargs
) -> Output:
    """
    Multiple the input with a (positive) smooth random field.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor
    shape : int | list[int]
        Shape of the low-resolution spline coefficients (lower = smoother)
    mean : float | ([C],) tensor, default=1
        Distribution mean.
    fwhm : float | ([C],) tensor, default=1
        Distribution full width at half-maximum.
    distrib : {"uniform", "gamma", "lognormal"}
        Probability distribution.
    order : int
        Spline order
    shared : bool
        Apply the same field to all channels.
        If True, probability parameters must be scalars.

    Other Parameters
    ----------------
    peak, std, vmin, vmax, alpha, beta, mu, sigma : float | ([C],) tensor
        Other parameters of the probability distribution.
    returns : [list or dict of] {"output", "input", "coeff", "field"}
        Values to return.

    Returns
    -------
    output : (C, *spatial) tensor

    """  # noqa: E501
    if (
        (mean is None) and
        (kwargs.get("mu", None) is not None) and
        (kwargs.get("peak", None) is not None) and
        (kwargs.get("vmin", None) is not None) and
        (kwargs.get("vmax", None) is not None)
    ):
        mean = 1

    if (
        (fwhm is None) and
        (kwargs.get("sigma", None) is not None) and
        (kwargs.get("std", None) is not None) and
        (kwargs.get("vmin", None) is not None) and
        (kwargs.get("vmax", None) is not None) and
        (kwargs.get("alpha", None) is not None)
    ):
        fwhm = 1

    ndim = input.ndim - 1
    shape = tuple(ensure_list(shape, ndim))
    if shared:
        shape = (1,) + shape
    else:
        shape = input.shape[:-1] + shape

    returns = kwargs.pop("returns", "output")
    kwargs["mean"] = mean
    kwargs["fwhm"] = fwhm
    coeff = random_field_like(distrib, input, shape, **kwargs)
    output = add_field(input, coeff, order, prefilter=False, returns=returns)

    output = returns_update(coeff, "coeff", output, returns)
    return output
