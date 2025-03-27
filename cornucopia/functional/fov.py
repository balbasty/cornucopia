__all__ = [
    "flip",
    "random_flip",
    "perm",
    "random_perm",
    "rot90",
    "rot180",
    "random_orient",
    "ensure_pow2",
    "pad",
    "crop",
    "patch",
    "random_patch",
]
import random
import itertools
from typing import Optional, Union, Tuple

import torch

from ..baseutils import prepare_output, returns_update
from ..utils.py import ensure_list, make_vector, prod
from ..utils.padding import pad as _pad
from ..utils import smart_math as math
from ._utils import (
    Tensor, OneOrMore, Output, _affine2layout, _axis_name2index
)


def flip(
    input: Tensor,
    axes: Optional[OneOrMore[Union[int, str]]] = None,
    orient: Union[str, Tensor] = "RAS",
    copy: bool = False,
    **kwargs
) -> Output:
    """
    Flip one or more spatial axes.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    axes : [list of] (int | {"LR", "AP", "IS"})
        Axes to flip, by index or by name.
        Indices correspond to spatial axes only (0 = first spatial dim, etc.)
        If None, flip all spatial axes.
    orient : str or tensor
        Tensor layout (`{"RAS", "LPS", ...}`) or orient matrix.
    copy : bool
        Copy the input even if no axes are flipped.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "axes"}

    Returns
    -------
    out : (C, *spatial) tensor
        Output tensor.
    """
    ndim = input.ndim - 1

    axes_ = axes
    if axes_ is None:
        axes_ = list(range(ndim))
    axes_ = ensure_list(axes_)

    if len(axes) == 0:

        output = input.clone() if copy else input

    else:

        # str to index
        if any(isinstance(ax, str) for ax in axes_):
            axes_ = _axis_name2index(axes_, orient)

        # neg to pos
        axes_ = [ndim + ax if ax < 0 else ax for ax in axes_]

        # flip
        output = input.flip([1 + ax for ax in axes_])

    return prepare_output(
        {"output": output, "input": input, "axes": axes},
        kwargs.pop("returns", "output")
    )()


def random_flip(
    input: Tensor,
    axes: Optional[OneOrMore[Union[int, str]]] = None,
    orient: Union[Tensor, str] = "RAS",
    copy: bool = False,
    **kwargs
) -> Output:
    """
    Flip one or more spatial axes.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    axes : [list of] (int | {"LR", "AP", "IS"})
        Axes that can be flipped, by index or by name.
        Indices correspond to spatial axes only (0 = first spatial dim, etc.)
        If None, all spatial axes can be flipped.
    orient : str or tensor
        Tensor layout (`{"RAS", "LPS", ...}`) or orient matrix.
    copy : bool
        Copy the input even if no axes are flipped.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "axes"}

    Returns
    -------
    out : (C, *spatial) tensor
        Output tensor.
    """
    ndim = input.ndim - 1

    if axes is None:
        axes = list(range(ndim))
    axes = list(ensure_list(axes))

    # sample axes to flip
    random.shuffle(axes)
    axes = axes[:random.randint(0, len(axes))]

    return flip(input, axes, orient, copy, **kwargs)


def perm(
    input: Tensor,
    perm: Optional[OneOrMore[Union[int, str]]] = None,
    orient: Union[str, Tensor] = "RAS",
    copy: bool = False,
    **kwargs
) -> Output:
    """
    Permute one or more spatial axes.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    perm : [list of] (int | {"LR", "AP", "IS"})
        Axes permutation, by index or by name.
        Indices correspond to spatial axes only (0 = first spatial dim, etc.)
        If None, inverse axis order.
    orient : str or tensor
        Tensor layout (`{"RAS", "LPS", ...}`) or orient matrix.
    copy : bool
        Copy the input (rather than returning a view).

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "perm"}

    Returns
    -------
    out : (C, *spatial) tensor
        Output tensor.
    """
    ndim = input.ndim - 1

    perm_ = perm
    if perm_ is None:
        perm_ = list(range(ndim))[::-1]
    perm_ = ensure_list(perm_)

    # str to index
    if any(isinstance(ax, str) for ax in perm_):
        perm_ = _axis_name2index(perm_, orient)

    # neg to pos
    perm_ = [ndim + ax if ax < 0 else ax for ax in perm_]

    # permute
    output = input.permute(0, *[1 + ax for ax in perm])

    if copy:
        output = output.clone()

    return prepare_output(
        {"output": output, "input": input, "perm": perm},
        kwargs.pop("returns", "output")
    )()


def random_perm(
    input: Tensor,
    axes: Optional[OneOrMore[Union[int, str]]] = None,
    orient: Union[str, Tensor] = "RAS",
    copy: bool = False,
    **kwargs
) -> Output:
    """
    Flip one or more spatial axes.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    axes : [list of] (int | {"LR", "AP", "IS"})
        Axes that can be permuted, by index or by name.
        Indices correspond to spatial axes only (0 = first spatial dim, etc.)
        If None, all spatial axes can be flipped.
    orient : str or tensor
        Tensor layout (`{"RAS", "LPS", ...}`) or orient matrix.
    copy : bool
        Copy the input (rather than returning a view).

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "perm"}

    Returns
    -------
    out : (C, *spatial) tensor
        Output tensor.
    """
    ndim = input.ndim - 1
    if axes is None:
        axes = list(range(ndim))
    axes = list(ensure_list(axes))

    # replace strings with integers
    if any(isinstance(ax, str) for ax in axes):
        axes = _axis_name2index(axes, orient)

    # sample axes to flip
    prm_axes = list(axes)
    random.shuffle(prm_axes)

    # build full permutation
    all_axes = list(range(ndim))
    for i, ax in zip(axes, prm_axes):
        all_axes[i] = ax

    return perm(input, all_axes, copy=copy, **kwargs)


def rot90(
    input: Tensor,
    plane: OneOrMore[Union[Tuple[int, int], str]] = (0, 1),
    negative: OneOrMore[bool] = False,
    double: OneOrMore[bool] = False,
    orient: Union[str, Tensor] = "RAS",
    copy: bool = False,
    **kwargs
) -> Output:
    """
    Rotate 90 degrees about an axis.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    plane : [list of]  ((int, int) | {"axial", "coronal", "sagittal"})
        Rotation plane.
    negative : [list of] bool
        Rotate by -90 deg instead of 90 deg.
    double : [list of] bool
        Rotate by 180 deg instead of 90 deg.
    orient : str or tensor
        Tensor layout (`{"RAS", "LPS", ...}`) or orient matrix.
    copy : bool
        Always copy the input (even if a view could be returned).

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "plane", "negative", "double"}

    Returns
    -------
    out : (C, *spatial) tensor
        Output tensor.
    """  # noqa: E501
    ndim = input.ndim - 1

    if plane is None or len(plane) == 0:

        output = input.clone() if copy else input

    else:

        plane_ = plane
        if isinstance(plane_, str) or isinstance(plane_[0], int):
            plane_ = [plane_]

        plane_ = list(ensure_list(plane_))
        negative_ = ensure_list(negative, len(plane_))
        double_ = ensure_list(double, len(plane_))

        # Convert named planes to indices
        if any(isinstance(p, str) for p in plane_):
            if not isinstance(orient, str):
                orient = _affine2layout(orient)
                orient = [
                    {"L": "R", "P": "A", "I": "S"}.get(ax, ax)
                    for ax in orient.upper()
                ]

            for i, p in enumerate(plane_):
                if not isinstance(p, str):
                    continue
                p = p[0].lower()
                if p == "c":
                    plane_[i] = (orient.index("R"), orient.index("S"))
                elif p == "a":
                    plane_[i] = (orient.index("R"), orient.index("P"))
                elif p == "s":
                    plane_[i] = (orient.index("P"), orient.index("S"))

        # Apply all rotations sequentially
        for p, n, d in zip(plane_, negative_, double_):
            # neg to pos + add 1 for channel dimension
            p = [1 + (ndim + ax if ax < 0 else ax) for ax in p]

            # add 1 for channel dimension
            if d:
                # 180 deg rotation == flip both axes
                output = input.flip(p)
            else:
                # 90 deg rotation == flip one axis, permute axes, then flip one
                output = input.transpose(*p)
                output = output.flip(p[1 if n else 0])

    return prepare_output(
        {"output": output, "input": input,
         "plane": plane, "negative": negative, "double": double},
        kwargs.pop("returns", "output")
    )()


def rot180(
    input: Tensor,
    plane: OneOrMore[Union[Tuple[int, int], str]] = (0, 1),
    orient: Union[str, Tensor] = "RAS",
    copy: bool = False,
    **kwargs
) -> Output:
    """
    Rotate 180 degrees about an axis.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    plane : [list of]  ((int, int) | {"axial", "coronal", "sagittal"})
        Rotation plane.
    orient : str or tensor
        Tensor layout (`{"RAS", "LPS", ...}`) or orient matrix.
    copy : bool
        Always copy the input (even if a view could be returned).

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "plane"}

    Returns
    -------
    out : (C, *spatial) tensor
        Output tensor.
    """  # noqa: E501
    return rot90(input, plane, double=True, orient=orient, copy=copy, **kwargs)


def random_orient(
    input: Tensor,
    posdet: bool = True,
    copy: bool = False,
    **kwargs
) -> Output:
    """
    Randomly reorient a tensor.

    Each pose has equal probability.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    posdet : bool
        Only accept transformations with a positive determinant.
    copy : bool
        Always copy the input (even if a view could be returned).

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "perm", "flip"}

    Returns
    -------
    out : (C, *spatial) tensor
        Output tensor.
    """
    ndim = input.ndim - 1

    def det(transformation):
        perm, flip = transformation
        det_perm = torch.eye(ndim)[perm].det()
        det_flip = prod(flip)
        return det_perm * det_flip

    # find all possible transformations
    perms = itertools.permutations(range(ndim))
    flips = itertools.product([True, False], repeat=ndim)
    xforms = itertools.product(perms, flips)
    if posdet:
        xforms = (xform for xform in xforms if det(xform) > 0)

    # sample transformation
    xforms = list(xforms)
    nforms = len(xforms)
    perm_, flip_ = xforms[random.randint(0, nforms-1)]
    flip_ = [i for i, f in enumerate(flip_) if f]

    # apply transformation
    output = flip(perm(input, perm_), flip_, copy=copy)

    return prepare_output(
        {"output": output, "input": input, "perm": perm_, "flip": flip_},
        kwargs.pop("returns", "output")
    )()


def ensure_pow2(
    input: Tensor,
    exponent: int = 1,
    bound: str = 'zero',
    **kwargs
) -> Output:
    """
    Pad the volume such that the tensor shape can be divided by `2**exponent`.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    exponent : [list of] int
        Exponent of the power of two.
    bound : [list of] str
        Boundary conditions used for padding.

    Returns
    -------
    output : (C, *spatial) tensor
        Output tensor.
    """
    shape = input.shape[1:]
    exponent = ensure_list(exponent, len(shape))
    bigshape = [max(2 ** e, s) for e, s in zip(exponent, shape)]
    return patch(input, bigshape, bound=bound, **kwargs)


def pad(
    input: Tensor,
    size: OneOrMore[Union[int, Tuple[int, int]]],
    unit: str = "vox",
    bound: str = "zero",
    side: Optional[str] = "both",
    **kwargs
) -> Output:
    """
    Pad (or crop) a tensor.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    size : [list of] (int | tuple[int, int])
        Padding (or cropping, if negative) per dimension.
        Tuples can be used to indicate different values on the left and right.
    unit : {"vox", "pct"}
        Unit of the padding size (voxels or percentage of the field of view).
    bound : [list of] str
        Boundary conditions used for padding.
    side : {"pre", "post", "both"}
        Apply padding on only one side, or on both.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "size"}

    Returns
    -------
    output : (C, *spatial)
        Output tensor
    """
    ndim = input.ndim - 1

    size_ = size
    if isinstance(size_, (tuple, int, float)):
        size_ = [size_] * ndim
    size_ = ensure_list(size_)
    size_ = max(0, ndim-len(size_)) * [0] + size_
    size_ = size_[:ndim]

    # fill left/right
    size_ = [
        (p, p) if isinstance(p, (int, float)) and side == "both" else
        (p, 0) if isinstance(p, (int, float)) and side == "pre" else
        (0, p) if isinstance(p, (int, float)) and side == "post" else
        p for p in size_
    ]
    # convert to voxels
    if unit[0].lower() == "v":
        size_ = [
            (int(round(q*s)) for q in p)
            for p, s in zip(size_, input.shape[1:])
        ]
    # convert to `pad` format
    size_ = [q for p in size_ for q in p]
    # add channel dimension
    size_ = [0, 0] + size_

    # apply padding
    output = _pad(input, size_, mode=bound)

    return prepare_output(
        {"output": output, "input": input, "size": size},
        kwargs.pop("returns", "output")
    )()


def crop(
    input: Tensor,
    size: OneOrMore[Union[int, Tuple[int, int]]],
    unit: str = "vox",
    bound: str = "zero",
    side: Optional[str] = "both",
    copy: bool = False,
    **kwargs
) -> Output:
    """
    Crop (or pad) a tensor.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    size : [list of] (int | tuple[int, int])
        Cropping (or padding, if negative) per dimension.
        Tuples can be used to indicate different values on the left and right.
    unit : {"vox", "pct"}
        Unit of the padding size (voxels or percentage of the field of view).
    bound : [list of] str
        Boundary conditions used for padding.
    side : {"pre", "post", "both"}
        Apply cropping on only one side, or on both.
    copy : bool
        Return a copy rather than a view.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "size"}

    Returns
    -------
    output : (C, *spatial)
        Output tensor
    """
    kwargs.setdefault("returns", "output")
    ndim = input.ndim - 1

    size_ = size
    if isinstance(size_, (tuple, int, float)):
        size_ = [size_] * ndim
    size_ = ensure_list(size_)
    size_ = max(0, ndim-len(size_)) * [0] + size_
    size_ = size_[:ndim]

    # If negative size, defer to pad (with opposite size)
    if any(
        (x < 0) if isinstance(x, int) else any(y < 0 for y in x)
        for x in size_
    ):
        size_ = [
            (-x) if isinstance(x, int) else tuple(-y for y in x)
            for x in size_
        ]
        output = pad(input, size, bound, side, **kwargs)
        return returns_update(size, "size", output, kwargs["returns"])

    # Otherwise, use slices

    # fill left/right
    size_ = [
        (p, p) if isinstance(p, (int, float)) and side == "both" else
        (p, 0) if isinstance(p, (int, float)) and side == "pre" else
        (0, p) if isinstance(p, (int, float)) and side == "post" else
        p for p in size_
    ]
    # convert to voxels
    if unit[0].lower() == "v":
        size_ = [
            (int(round(q*s)) for q in p)
            for p, s in zip(size_, input.shape[1:])
        ]
    # convert to slicer
    slicer = tuple(
        slice(s[0], (-s[1]) or None)
        for s in size_
    )
    output = input[(Ellipsis,) + slicer]

    if copy:
        output = output.clone()

    return prepare_output(
        {"output": output, "input": input, "size": size},
        kwargs.pop("returns", "output")
    )()


def patch(
    input: Tensor,
    shape: OneOrMore[int] = 64,
    center: OneOrMore[float] = 0,
    bound: str = "zero",
    copy: bool = False,
    **kwargs
) -> Output:
    """
    Extract a patch from the volume.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    shape : [list of] int
        Patch shape
    center : [list of] float
        Patch center, in relative coordinates -1..1
    bound : str
        Boundary condition in case padding is needed
    copy : bool
        Return a copy rather than a view.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "center"}

    Returns
    -------
    output : (C, *shape)
        Output tensor

    """
    # NOTE not differentiable wrt `center`, since we force the patch to
    # be aligned with the input lattice. If we want differentiability,
    # we need to use some sort of interpolation.

    ndim = input.ndim - 1
    ishape = input.shape[1:]
    shape_ = ensure_list(shape, ndim)
    center_ = make_vector(center, ndim).tolist()
    center_ = [(c + 1) / 2 * (s - 1) for c, s in zip(center_, ishape)]
    crop_size = []
    for ss, cc, sv in zip(shape_, center_, ishape):
        first = int(math.floor(cc - ss/2))
        last = first + ss
        left, right = first, sv - last
        crop_size.append((left, right))

    output = crop(input, crop_size, bound=bound, copy=copy)

    return prepare_output(
        {"output": output, "input": input, "center": center},
        kwargs.pop("returns", "output")
    )()


def random_patch(
    input: Tensor,
    shape: OneOrMore[int] = 64,
    bound: str = "zero",
    copy: bool = False,
    **kwargs
) -> Output:
    """
    Extract a random patch from the volume.

    Parameters
    ----------
    input : (C, *spatial) tensor
        Input tensor.
    shape : [list of] int
        Patch shape
    bound : str
        Boundary condition in case padding is needed
        (only needed if the input shape is smaller than the patch shape).
    copy : bool
        Return a copy rather than a view.

    Other Parameters
    ----------------
    returns : [list or dict of] {"output", "input", "center"}

    Returns
    -------
    output : (C, *shape)
        Output tensor

    """
    ishape = input.shape[1:]
    shape_ = ensure_list(shape, len(ishape))
    min_center = [max(p/s - 1, -1) for p, s in zip(shape_, ishape)]
    max_center = [min(1 - p/s, 1) for p, s in zip(shape_, ishape)]
    center = [
        random.random() * (mx - mn) + mn
        for mn, mx in zip(min_center, max_center)
    ]
    return patch(input, shape, center, bound, copy, **kwargs)
