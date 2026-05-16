"""This module contains transforms that operate on image intensities."""
__all__ = [
    'AddValueTransform',
    'MulValueTransform',
    'AddMulTransform',
    'ReturnValueTransform',
    'FillValueTransform',
    'ClipTransform',
    'BaseFieldTransform',
    'AddFieldTransform',
    'MulFieldTransform',
    'RandomAddFieldTransform',
    'RandomMulFieldTransform',
    'RandomSlicewiseMulFieldTransform',
    'RandomMulTransform',
    'RandomAddTransform',
    'RandomAddMulTransform',
    'GammaFinalTransform',
    'GammaTransform',
    'RandomGammaTransform',
    'ZTransform',
    'QuantileTransform',
    'MinMaxTransform',
]
# stdlib
import math
from math import inf
from numbers import Number

# dependencies
import interpol
import torch
import typing_extensions as tx
from torch import Tensor
from torch.nn.functional import interpolate

# internals
from .baseutils import Returned, prepare_output
from .base import Transform, FinalTransform, NonFinalTransform
from .special import RandomizedTransform, SequentialTransform
from .random import Sampler, Uniform, RandInt, Fixed, make_range
from .utils.py import ensure_list, positive_index
from .utils.smart_inplace import add_, mul_, div_, pow_
from .utils.compat import clamp
from . import typing as cct

# typing
_NumberOrTensor = tx.Union[Number, Tensor]
_UnaryOperator = tx.Callable[[Tensor], Tensor]
_BinaryOperator = tx.Callable[[Tensor, _NumberOrTensor], Tensor]


class OpConstTransform(FinalTransform):
    """Base class for arithmetic operations with a constant value"""

    _op: tx.Optional[_BinaryOperator] = None
    _inv: tx.Dict[_BinaryOperator, _UnaryOperator] = {
        torch.add: lambda x: -x,
        torch.mul: lambda x: 1/x,
    }

    def __init__(
        self,
        value: _NumberOrTensor,
        op: tx.Optional[_BinaryOperator] = None,
        value_name: str = 'value',
        **kwargs
    ):
        """
        Parameters
        ----------
        value : number or tensor
            right-hand side of the operation
        op : {torch.add, torch.mul}
            Arithmetic operation
        value_name : str
            Name used when returning the rhs value
        """
        super().__init__(**kwargs)
        self.value = value
        self.op = op or self._op
        self.value_name = value_name

    def __getattr__(self, name: str) -> _NumberOrTensor:
        if name == self.__dict__.get("value_name"):
            return self.__dict__.get("value")
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: _NumberOrTensor) -> None:
        if name == self.__dict__.get("value_name"):
            name = 'value'
        super().__setattr__(name, value)

    def xform(self, x: Tensor) -> Returned:
        value = self.value
        if torch.is_tensor(value):
            value = value.to(x)
        y = self.op(x, value)
        return prepare_output(
            {'input': x, 'output': y, self.value_name: value}, self.returns
        )

    def make_inverse(self) -> Transform:
        inv = self._inv[self.op]
        return type(self)(
            inv(self.value), **self.get_prm(), value_name=self.value_name
        )


class AddValueTransform(OpConstTransform):
    """Add a constant value"""
    _op: _BinaryOperator = torch.add


class MulValueTransform(OpConstTransform):
    """Multiply with a constant value"""
    _op: _BinaryOperator = torch.mul


class FillValueTransform(FinalTransform):
    """Fills the tensor with a value inside a mask"""

    def __init__(
        self,
        mask: Tensor,
        value: _NumberOrTensor,
        mask_name: str = 'mask',
        value_name: str = 'value',
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        mask : tensor
            Mask of voxels in which to set the value
        value : number or tensor
            right-hand side of the operation
        mask_name : str
            Name used when returning the mask
        value_name : str
            Name used when returning the rhs value
        """
        super().__init__(**kwargs)
        self.mask = mask
        self.value = value
        self.mask_name = mask_name
        self.value_name = value_name

    def xform(self, x: Tensor) -> Returned:
        mask, value = self.mask, self.value
        mask = mask.to(x.device)
        if torch.is_tensor(value):
            value = value.to(x)
        y = x.masked_fill(mask, value)
        return prepare_output(
            {'input': x, 'output': y,
             self.mask_name: mask,
             self.value_name: value},
            self.returns
        )


class ReturnValueTransform(FinalTransform):
    """Fills the tensor with a value inside a mask"""

    def __init__(
        self,
        value: _NumberOrTensor,
        value_name: str = 'output',
        dtype: tx.Optional[torch.dtype] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        value : number or tensor
            right-hand side of the operation
        value_name : str
            Name used when returning the rhs value
        """
        super().__init__(**kwargs)
        self.value = value
        self.value_name = value_name
        self.dtype = dtype

    def __getattr__(self, name: str) -> _NumberOrTensor:
        if name == self.__dict__.get("value_name"):
            return self.__dict__.get("value")
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: _NumberOrTensor) -> None:
        if name == self.__dict__.get("value_name"):
            name = 'value'
        super().__setattr__(name, value)

    def xform(self, x: Tensor) -> Returned:
        dtype = self.dtype or x.dtype
        return torch.as_tensor(self.value, dtype=dtype, device=x.device)


class AddMulTransform(FinalTransform):
    """Constant intensity affine transform: `y = x * slope + offset`"""

    def __init__(
        self,
        slope: _NumberOrTensor = 1,
        offset: _NumberOrTensor = 0,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        slope : number or tensor
            Affine slope
        offset : number or tensor
            Affine offset
        """
        super().__init__(**kwargs)
        self.slope = slope
        self.offset = offset

    def xform(self, x: Tensor) -> Returned:
        slope, offset = self.slope, self.offset
        if torch.is_tensor(slope):
            slope = slope.to(x)
        if torch.is_tensor(offset):
            offset = offset.to(x)
        y = slope * x + offset
        return prepare_output(
            {'input': x, 'output': y, 'slope': slope, 'offset': offset},
            self.returns
        )

    def make_inverse(self) -> 'AddMulTransform':
        return AddMulTransform(
            1/self.slope, -self.offset/self.slope, **self.get_prm()
        )


class ClipTransform(FinalTransform):
    """Clip extremum values"""

    def __init__(
        self,
        vmin: tx.Optional[_NumberOrTensor] = None,
        vmax: tx.Optional[_NumberOrTensor] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        vmin : number or tensor, optional
            Min value
        vmax : number or tensor, optional
            Max value
        """
        super().__init__(**kwargs)
        self.vmin = vmin
        self.vmax = vmax

    def xform(self, x: Tensor) -> Returned:
        vmin, vmax = self.vmin, self.vmax
        if torch.is_tensor(vmin):
            vmin = vmin.to(x)
        if torch.is_tensor(vmax):
            vmax = vmax.to(x)
        y = clamp(x, vmin, vmax)
        return prepare_output(
            {'input': x, 'output': y, 'vmin': vmin, 'vmax': vmax},
            self.returns
        )


class RandomMulTransform(RandomizedTransform):
    """
    Random multiplicative transform.
    """

    Final = Next = MulValueTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        value: tx.Union[Sampler, float, tx.Tuple[float, float]] = (0.5, 2),
        *,
        shared: cct.SharedType = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        value : Sampler | [pair of] float
            Bound for multiplicative value
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Apply same transform to all images/channels
        """
        super().__init__(
            MulValueTransform,
            Uniform.make(make_range(0, value)),
            shared=shared,
            **kwargs
        )


class RandomAddTransform(RandomizedTransform):
    """
    Random additive transform.
    """

    Final = Next = AddValueTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        value: tx.Union[Sampler, float, tx.Tuple[float, float]] = 1,
        *,
        shared: cct.SharedType = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        value : Sampler | [pair of] float
            Bound for additive value
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Apply same transform to all images/channels
        """
        super().__init__(
            AddValueTransform,
            Uniform.make(make_range(value)),
            shared=shared,
            **kwargs
        )


class RandomAddMulTransform(RandomizedTransform):
    """
    Random intensity affine transform.
    """

    Final = Next = AddMulTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        slope: tx.Union[Sampler, float, tx.Tuple[float, float]] = 1,
        offset: tx.Union[Sampler, float, tx.Tuple[float, float]] = 0.5,
        *,
        shared: cct.SharedType = False,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        slope : Sampler | [pair of] float
            Bound for slope
        offset : Sampler | [pair of] float
            Bound for offset
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Apply same transform to all images/channels
        """
        super().__init__(
            AddMulTransform,
            (Uniform.make(make_range(slope)),
             Uniform.make(make_range(offset))),
            shared=shared,
            **kwargs
        )


class SplineUpsampleTransform(FinalTransform):
    """Upsample a field using spline interpolation"""

    def __init__(
        self,
        order: int = 3,
        prefilter: bool = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        order : int
            Spline interpolation order
        prefilter : bool
            Spline prefiltering
            (True for interpolation, False for spline evaluation)
        """
        super().__init__(**kwargs)
        self.order = order
        self.prefilter = prefilter

    def xform(self, x: Tensor) -> Tensor:
        fullshape = x.shape[1:]
        if self.order == 1:
            mode = ('trilinear' if len(fullshape) == 3 else
                    'bilinear' if len(fullshape) == 2 else
                    'linear')
            y = interpolate(
                x.unsqueeze(0), fullshape, mode=mode,
                align_corners=True
            ).squeeze(-0)
        else:
            y = interpol.resize(
                x, shape=fullshape, interpolation=self.order,
                prefilter=self.prefilter
            )
        return y


class BaseFieldTransform(NonFinalTransform):
    """Base class for transforms that sample a smooth field"""

    Final = Next = AddValueTransform
    """The transform type returned by `make_final`."""

    value_name: str = 'field'

    def __init__(
        self,
        shape: tx.Union[int, tx.Sequence[int]] = 5,
        vmin: float = 0 ,
        vmax: float = 1,
        order: int = 3,
        slice: tx.Optional[int] = None,
        thickness: tx.Optional[int] = None,
        *,
        shared: cct.SharedType = False,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        shape : [list of] int
            Number of spline control points
        vmin : float
            Minimum value
        vmax : float
            Maximum value
        order : int
            Spline order
        slice : int
            Slice direction, if slicewise.
        thickness : int
            Slice thickness, if slicewise.
            Note that `shape` will be scaled along the slice direction
            so that the number of nodes is approximately preserved.
        returns : [list or dict of] {'input', 'output', 'field'}
            Which tensor(s) to return
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Apply the same field to all channels

        """
        super().__init__(shared=shared, **kwargs)
        self.shape = shape
        self.vmax = vmax
        self.vmin = vmin
        self.order = order
        self.slice = slice
        self.thickness = thickness

    def make_field(
        self,
        batch: int,
        smallshape: tx.Sequence[int],
        fullshape: tx.Optional[tx.Sequence[int]] = None,
        **backend
    ) -> None:
        """Generate the random coefficients.

        Parameters
        ----------
        batch : int
            Number of fields to generate
        smallshape : list of int
            Number of spline control points
        fullshape : list of int, optional
            If given, the coefficients will be upsampled to this shape.

        Other Parameters
        ----------------
        dtype : torch.dtype
            Data type of the generated field.
        device : torch.device | str
            Device on which to generate the field.

        Returns
        -------
        field : (batch, *smallshape) tensor | (batch, *fullshape) tensor
            If `fullshape` is given, returns the upsampled field of values.
            Otherise, returns the spline coefficients.

        """
        smallshape = ensure_list(smallshape, len(fullshape))
        smallshape = [min(small, full) for small, full
                      in zip(smallshape, fullshape)]
        if not backend['dtype'].is_floating_point:
            backend['dtype'] = torch.get_default_dtype()
        b = torch.rand([batch, *smallshape], **backend)
        if fullshape:
            b = self.upsample_field(b, fullshape)
        return b

    def upsample_field(self, coeff: Tensor, shape: tx.Sequence[int]) -> Tensor:
        """Compute the full-sized field from its spline coefficients.

        Parameters
        ----------
        coeff : (batch, *smallshape) tensor
            Spline coefficients
        shape : list of int
            Target shape for the upsampled field

        Returns
        -------
        field : (batch, *shape) tensor
            Upsampled field of values
        """
        if self.order == 1:
            mode = ('trilinear' if len(shape) == 3 else
                    'bilinear' if len(shape) == 2 else
                    'linear')
            b = interpolate(
                coeff.unsqueeze(0), shape, mode=mode,
                align_corners=True
            ).squeeze(-0)
        else:
            b = interpol.resize(
                coeff, shape=shape, interpolation=self.order,
                prefilter=False
            )
        return b

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self

        ndim = x.ndim - 1
        fullshape = list(x.shape[1:])
        batch = 1 if 'channels' in self.shared else len(x)
        backend = dict(dtype=x.dtype, device=x.device)

        # slicewise bias
        if self.slice is not None:
            slice = positive_index(self.slice, ndim)
            thickness = self.thickness or 1
            thickness = min(thickness, x.shape[1+slice])
            nb_slices = int(math.ceil(x.shape[1+slice] / thickness))

            smallshape = ensure_list(self.shape, ndim)
            smallshape[slice] = int(math.ceil(smallshape[slice] / nb_slices))
            smallshape = [min(small, full) for small, full
                          in zip(smallshape, fullshape)]

            if thickness == 1:
                # bias independent across slices -> batch it
                batch1 = batch * fullshape[slice]
                del smallshape[slice]
                del fullshape[slice]
                b = self.make_field(batch1, smallshape, fullshape, **backend)
                b = b.reshape([batch, -1, *b.shape[1:]])
                b = b.movedim(1, 1+slice)

            elif fullshape[slice] % thickness == 0:
                # shape divisible by thickness -> unfold and batch
                fullshape0 = list(fullshape)
                _, *fullshape = x.shape
                batch1 = batch * nb_slices
                fullshape[slice] = thickness
                b = self.make_field(batch1, smallshape, fullshape, **backend)
                b = b.reshape([batch, -1, *b.shape[1:]])
                b = b.movedim(1, 1+slice)
                b = b.reshape([batch, *fullshape0])

            else:
                # otherwise, the input is not exactly divisible by thickness
                b = x.new_empty([batch, *fullshape], **backend)

                # use same strategy as before for all but last slice
                fullshape0 = list(fullshape)
                _, *fullshape = x.shape
                batch1 = batch * (nb_slices - 1)
                fullshape[slice] = thickness
                fullshape0[slice] = (nb_slices - 1) * thickness
                b1 = self.make_field(batch1, smallshape, fullshape, **backend)
                b1 = b1.reshape([batch, -1, *b1.shape[1:]])
                b1 = b1.movedim(1, 1+slice)
                b1 = b1.reshape([batch, *fullshape0])

                # copy into the larger placeholder
                b1 = b1.movedim(1+slice, 0)
                b.movedim(1+slice, 0)[:len(b1)].copy_(b1)

                # process last slice
                fullshape[slice] = b.shape[1+slice] - len(b1)
                b1 = self.make_field(batch, smallshape, fullshape, **backend)
                b1 = b1.movedim(1+slice, 0)
                b.movedim(1+slice, 0)[-len(b1):].copy_(b1)

        else:
            # global bias
            b = self.make_field(batch, self.shape, fullshape, **backend)

        # rescale intensities
        batch = len(b)
        vmin, vmax = self.vmin, self.vmax
        if torch.is_tensor(vmin):
            while vmin.ndim < b.ndim:
                vmin = vmin.unsqueeze(-1)
            batch = max(batch, len(vmin))
        if torch.is_tensor(vmax):
            while vmax.ndim < b.ndim:
                vmax = vmax.unsqueeze(-1)
            batch = max(batch, len(vmax))
        if len(b) < batch:
            b = b.expand([batch, *b.shape[1:]]).clone()

        b = add_(mul_(b, self.vmax-self.vmin), self.vmin)

        return self.Next(
            b, value_name=self.value_name, **self.get_prm()
        ).make_final(x, max_depth-1)


class MulFieldTransform(BaseFieldTransform):
    """Smooth multiplicative (bias) field"""

    Final = Next = MulValueTransform
    """The transform type returned by `make_final`."""


class RandomMulFieldTransform(NonFinalTransform):
    """Random multiplicative bias field transform"""

    Next = MulFieldTransform
    """The transform type returned by `make_final(..., max_depth=1)`."""

    Final = MulValueTransform
    """The transform type returned by `make_final(..., max_depth=inf)`."""

    def __init__(
        self,
        shape: tx.Union[Sampler, int] = 8,
        vmax: tx.Union[Sampler, float] = 1,
        order: int = 3,
        symmetric: tx.Union[bool, float] = False,
        *,
        shared: cct.SharedType = False,
        shared_field: tx.Union[str, bool, None] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        shape : Sampler | int
            Sampler or Upper bound for number of control points
        vmax : Sampler | float
            Sampler or Upper bound for maximum value
        order : int
            Spline order
        symmetric : bool | float
            If a float, the bias field will take values in
            `(symmetric-vmax, symmetric+vmax)`.
            If False, it will take values in `(0, vmax)`.
            If True, it will take values in `(1-vmax, 1+vmax)`.

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'field'}
            Which tensor(s) to return
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Whether to share random parameters across tensors and/or channels
        shared_field : {'channels', 'tensors', 'channels+tensors', ''} | bool | None
            Whether to share random field across tensors and/or channels.
            By default: same as `shared`
        """  # noqa: E501
        super().__init__(shared=shared, **kwargs)
        self.vmax = Uniform.make(make_range(0, vmax))
        self.shape = RandInt.make(make_range(2, shape))
        self.order = Fixed.make(order)
        self.symmetric = symmetric
        self.shared_field = self._prepare_shared(shared_field)

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        vmax, shape, order = self.vmax, self.shape, self.order
        shared_field = self.shared_field
        if isinstance(vmax, Sampler):
            vmax = vmax()
        if isinstance(shape, Sampler):
            shape = shape(x.ndim-1)
        if isinstance(order, Sampler):
            order = order()
        if shared_field is None:
            shared_field = self.shared
        if self.symmetric is False:
            vmin = 0
        else:
            mid = self.symmetric
            vmin, vmax = mid - vmax, mid + vmax
        return MulFieldTransform(
            shape, vmin, vmax, order, shared=shared_field, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomSlicewiseMulFieldTransform(NonFinalTransform):
    """Random multiplicative bias field transform, per slice or slab"""

    Next = MulFieldTransform
    """The transform type returned by `make_final(..., max_depth=1)`."""

    Final = MulValueTransform
    """The transform type returned by `make_final(..., max_depth=inf)`."""

    def __init__(
        self,
        shape: tx.Union[Sampler, int] = 8,
        vmax: tx.Union[Sampler, float] = 1,
        order: int = 3,
        slice: tx.Optional[int] = None,
        thickness: tx.Union[Sampler, int] = 32,
        shape_through: tx.Optional[tx.Union[Sampler, int]] = None,
        *,
        shared: cct.SharedType = False,
        shared_field: tx.Union[str, bool, None] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        shape : Sampler | int
            Sampler or Upper bound for number of control points
        vmax : Sampler | float
            Sampler or Upper bound for maximum value
        order : int
            Spline order
        slice : int | None
            Slice axis. If None, sample one randomly
        thickness : Sampler | int
            Sampler or Upper bound for slice thickness
        shape_through : Sampler | int | None
            Sampler or Upper bound for number of control points
            along the slice direction. If None, same as `shape`.

        Other Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Whether to share random parameters across tensors and/or channels
        shared_field : {'channels', 'tensors', 'channels+tensors', ''} | bool | None
            Whether to share random field across tensors and/or channels.
            By default: same as `shared`
        """  # noqa: E501
        super().__init__(shared=shared, **kwargs)
        if shape_through is not None:
            shape_through = RandInt.make(make_range(1, shape_through))
        self.vmax = Uniform.make(make_range(0, vmax))
        self.shape = RandInt.make(make_range(2, shape))
        self.order = Fixed.make(order)
        self.slice = slice
        self.thickness = RandInt.make(make_range(1, thickness))
        self.shape_through = shape_through
        self.shared_field = self._prepare_shared(shared_field)

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        ndim = x.ndim - 1

        vmax = self.vmax
        shape = self.shape
        order = self.order
        slice = self.slice
        thickness = self.thickness
        shape_through = self.shape_through
        shared_field = self.shared_field

        if slice is None:
            slice = RandInt(x.ndim-2)

        if shape_through is not None:
            if isinstance(slice, Sampler):
                slice = slice()
            slice = positive_index(slice, ndim)
            if isinstance(shape, Sampler):
                shape = shape(ndim)
            shape = list(ensure_list(shape, ndim))
            if isinstance(thickness, Sampler):
                thickness = thickness()
            if isinstance(shape_through, Sampler):
                shape_through = shape_through()
            shape_through0 = x.shape[1+self.sample['slice']]
            shape_through *= int(math.ceil(shape_through0 / thickness))
            shape[slice] = shape_through

        if isinstance(vmax, Sampler):
            vmax = vmax()
        if isinstance(shape, Sampler):
            shape = shape(ndim)
        if isinstance(order, Sampler):
            order = order()
        if isinstance(slice, Sampler):
            slice = slice()
        if isinstance(thickness, Sampler):
            thickness = thickness()
        if isinstance(shape_through, Sampler):
            shape_through = shape_through()
        if shared_field is None:
            shared_field = self.shared

        return MulFieldTransform(
            shape, 0, vmax, order, slice, thickness,
            shared=shared_field, **self.get_prm()
        ).make_final(x, max_depth-1)


class AddFieldTransform(BaseFieldTransform):
    """Smooth additive (bias) field"""

    Final = Next = AddValueTransform
    """The transform type returned by `make_final`."""


class RandomAddFieldTransform(NonFinalTransform):
    """Random additive bias field transform"""

    def __init__(
        self,
        shape: tx.Union[Sampler, int] = 8,
        vmin: tx.Union[Sampler, float] = -1,
        vmax: tx.Union[Sampler, float] = 1,
        order: tx.Union[Sampler, int] = 3,
        *,
        shared: cct.SharedType = False,
        shared_field: tx.Union[str, bool, None] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        shape : Sampler | int
            Sampler or Upper bound for number of control points
        vmin : Sampler | float
            Sampler or Lower bound for minimum value
        vmax : Sampler | float
            Sampler or Upper bound for maximum value
        order : Sampler | int
            Spline order

        Other Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Whether to share random parameters across tensors and/or channels
        shared_field : {'channels', 'tensors', 'channels+tensors', ''} | bool | None
            Whether to share random field across tensors and/or channels.
            By default: same as `shared`
        """
        super().__init__(shared=shared, **kwargs)
        self.vmin = Uniform.make(make_range(vmin, 0))
        self.vmax = Uniform.make(make_range(0, vmax))
        self.shape = RandInt.make(make_range(2, shape))
        self.order = Fixed.make(order)
        self.shared_field = self._prepare_shared(shared_field)

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        vmin, vmax, shape, order = self.vmin, self.vmax, self.shape, self.order
        shared_field = self.shared_field
        if isinstance(vmin, Sampler):
            vmin = vmin()
        if isinstance(vmax, Sampler):
            vmax = vmax()
        if isinstance(shape, Sampler):
            shape = shape(x.ndim-1)
        if isinstance(order, Sampler):
            order = order()
        if shared_field is None:
            shared_field = self.shared
        return AddFieldTransform(
            shape, vmin, vmax, order, shared=shared_field, **self.get_prm()
        ).make_final(x, max_depth-1)


class GammaFinalTransform(FinalTransform):
    """Gamma correction with fixed parameters.

    The transform is defined as:

    ```python
    y = (x-vmin) / (vmax-vmin) ** gamma * (vmax-vmin) + vmin
    ```

    In this transform, `vmin` and `vmax` are pre-calculated and fixed,
    whereas in `GammaTransform`, they are computed from the image intensities.
    """

    _ScalarOrVector = tx.Union[float, tx.Sequence[float], Tensor]

    def __init__(
        self,
        gamma: _ScalarOrVector = 1,
        vmin: _ScalarOrVector = 0,
        vmax: _ScalarOrVector = 1,
        **kwargs
    ):
        """
        Parameters
        ----------
        gamma : number | (C,) list[number] | (C,) tensor
            Exponent of the Gamma transform
        vmin : number | (C,) list[number] | (C,) tensor
            Minimum value for the transform
        vmax : number | (C,) list[number] | (C,) tensor
            Maximum value for the transform
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.vmin = vmin
        self.vmax = vmax

    def __repr__(self) -> str:
        gamma, vmin, vmax = self.gamma, self.vmin, self.vmax
        if torch.is_tensor(gamma):
            gamma = gamma.detach().tolist()
        if torch.is_tensor(vmin):
            vmin = vmin.detach().tolist()
        if torch.is_tensor(vmax):
            vmax = vmax.detach().tolist()
        return f"{type(self).__name__}(gamma={gamma}, vmin={vmin}, vmax={vmax})"

    def xform(self, x: Tensor) -> Returned:
        vmin = torch.as_tensor(self.vmin, dtype=x.dtype, device=x.device)
        vmax = torch.as_tensor(self.vmax, dtype=x.dtype, device=x.device)
        gamma = torch.as_tensor(self.gamma, dtype=x.dtype, device=x.device)
        vmin = vmin.reshape([-1] + [1] * (x.ndim-1))
        vmax = vmax.reshape([-1] + [1] * (x.ndim-1))
        gamma = gamma.reshape([-1] + [1] * (x.ndim-1))

        # NOTE
        # * we add a little epsilon to the denominator to avoid
        #   division by zero.
        # * We also ensure that the rescaled input is in (0+eps, 1-eps)
        #   to ensure differentiability everywhere.
        # * The vmin/vmax may have been computed on a different image
        #   than x, so we cannot trust that x.min() < vmin.

        den = vmax - vmin
        num = x - vmin
        num.clamp_(1e-5 * den, (1.0 - 1e-5) * den)
        y = div_(num, add_(den, 1e-5))
        y = pow_(y, gamma)
        if gamma.requires_grad:
            # When gamma requires grad,  mul_(y, vmax-vmin) is happy
            # to overwrite y, but we cant because we need y to
            # backprop through pow. So we need an explicit branch.
            y = torch.add(torch.mul(y, vmax - vmin), vmin)
        else:
            y = add_(mul_(y, vmax - vmin), vmin)

        return prepare_output(
            dict(input=x, output=y, vmin=vmin, vmax=vmax, gamma=gamma),
            self.returns)


class GammaTransform(NonFinalTransform):
    """Gamma correction

    References
    ----------
    1. https://en.wikipedia.org/wiki/Gamma_correction
    """

    Final = Next = GammaFinalTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        gamma: float = 1,
        vmin: tx.Optional[float] = None,
        vmax: tx.Optional[float] = None,
        *,
        shared: cct.SharedType = False,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        gamma : float
            Exponent of the Gamma transform
        vmin : float | None
            Value to use as the minimum (default: x.min())
        vmax : float | None
            Value to use as the maximum (default: x.max())
        returns : [list or dict] {'input', 'output', 'vmin', 'vmax', 'gamma'}
            Which tensors to return

        Other Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Use the same vmin/vmax for all channels

        """
        super().__init__(shared=shared, **kwargs)
        self.gamma = kwargs.pop('value', gamma)
        self.vmin = vmin
        self.vmax = vmax

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        ndim = x.dim() - 1
        if self.vmin is None:
            vmin = x.reshape(len(x), -1).min(-1).values
            for _ in range(ndim):
                vmin = vmin.unsqueeze(-1)
            if 'channels' in self.shared:
                vmin = vmin.min()
        else:
            vmin = self.vmin
        if self.vmax is None:
            vmax = x.reshape(len(x), -1).max(-1).values
            for _ in range(ndim):
                vmax = vmax.unsqueeze(-1)
            if 'channels' in self.shared:
                vmax = vmax.max()
        else:
            vmax = self.vmax
        return self.Next(
            self.gamma, vmin, vmax, **self.get_prm()
        ).make_final(max_depth-1)


class RandomGammaTransform(NonFinalTransform):
    """
    Random Gamma transform.
    """

    Next = GammaTransform
    """The transform type returned by `make_final(..., max_depth=1)`."""

    Final = GammaFinalTransform
    """The transform type returned by `make_final(..., max_depth=inf)`."""

    def __init__(
        self,
        gamma: tx.Union[Sampler, float, tx.Tuple[float, float]] = (0.5, 2),
        *,
        shared: cct.SharedType = False,
        shared_minmax: tx.Optional[cct.SharedType] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        gamma : Sampler or [pair of] float
            Sampler or range for the exponent value

        Other Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Apply same gamma for all images/channels
        shared_minmax : {'channels', 'tensors', 'channels+tensors', '', None}
            Use the same vmin/vmax for all channels. Default: same as `shared`.

        """
        super().__init__(shared=shared, **kwargs)
        self.gamma = Uniform.make(kwargs.pop('value', gamma))
        self.shared_minmax = self._prepare_shared(shared_minmax)

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        gamma = self.gamma
        if isinstance(gamma, Sampler):
            gamma = gamma()
        shared_minmax = self.shared_minmax
        if shared_minmax is None:
            shared_minmax = self.shared
        return GammaTransform(
            gamma, shared=shared_minmax, **self.get_prm()
        ).make_final(x, max_depth-1)


class ZTransform(NonFinalTransform):
    """
    Z-transform the data -> zero mean, unit standard deviation
    """

    Final = Next = AddMulTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self, mu: float = 0, sigma: float = 1,
        *, shared: cct.SharedType = False, **kwargs
    ):
        """
        Parameters
        ----------
        mu : float
            Target mean. If None, keep the input mean.
        sigma : float
            Target standard deviation. If None, keep the input sd.

        Other Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Use the same mean/sigma for all images/channels
        """
        super().__init__(shared=shared, **kwargs)
        self.mu = mu
        self.sigma = sigma

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        if 'channels' in self.shared:
            opt = dict()
        else:
            opt = dict(dim=list(range(1, x.ndim)), keepdim=True)
        mu0, sigma0 = x.mean(**opt), x.std(**opt)
        mu1 = self.mu if self.mu is not None else mu0
        sigma1 = self.sigma if self.sigma is not None else sigma0
        scale = sigma1 / sigma0
        offset = mu1 - mu0 * scale
        return AddMulTransform(
            scale, offset, **self.get_prm()
        ).make_final(x, max_depth-1)


class QuantileTransform(NonFinalTransform):
    """Match lower and upper quantiles to (0, 1)"""

    Final = Next = AddMulTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        pmin: float = 0.01,
        pmax: float = 0.99,
        vmin: float = 0,
        vmax: float = 1,
        clip: bool = False,
        max_samples: int = 10000,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        pmin : (0..1)
            Lower quantile
        pmax : (0..1)
            Upper quantile
        vmin : float
            Lower target value
        vmax : float
            Upper target value
        clip : bool
            Clip values outside (vmin, vmax)
        max_samples : int
            Maximum number of pixels to use for quantile estimation (for speed)
        """
        super().__init__(**kwargs)
        self.pmin = pmin
        self.pmax = pmax
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip
        self.max_samples = max_samples

    def make_final(self, x: Tensor, max_depth: float = inf) -> Transform:
        if max_depth == 0:
            return self

        ndim = x.ndim - 1

        x_ = x.reshape([len(x), -1])
        x_ = x_[:, (x_ != 0).all(0) & x_.isfinite().all(0)]
        if self.max_samples and self.max_samples < x_.shape[1]:
            idx_ = torch.randperm(x_.shape[-1], device=x_.device)
            idx_ = idx_[:self.max_samples]
            x_ = x_[:, idx_]

        qdim = (-1 if 'channels' not in self.shared else None)
        pmin = torch.quantile(x_, self.pmin, dim=qdim)
        pmax = torch.quantile(x_, self.pmax, dim=qdim)
        pmin = pmin[(Ellipsis,) + (None,) * ndim]
        pmax = pmax[(Ellipsis,) + (None,) * ndim]

        num = self.vmax - self.vmin
        den = (pmax - pmin).clamp_min_(1e-16)
        slope = num / den
        offset = self.vmin - pmin * slope

        if self.clip:
            return SequentialTransform([
                AddMulTransform(slope, offset, **self.get_prm()),
                ClipTransform(self.vmin, self.vmax, **self.get_prm())
            ]).make_final(x, max_depth-1)
        else:
            return AddMulTransform(
                slope, offset, **self.get_prm()
            ).make_final(x, max_depth-1)


class MinMaxTransform(NonFinalTransform):
    """Match min and max values to (0, 1)"""

    Final = Next = AddMulTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self, vmin: float = 0, vmax: float = 1, clip: bool = False, **kwargs
    ) -> None:
        """

        Parameters
        ----------
        vmin : float
            Lower target value
        vmax : float
            Upper target value
        clip : bool
            Clip values outside (vmin, vmax)
        """
        super().__init__(**kwargs)
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip

    def make_final(self, x: Tensor, max_depth: float = inf) -> Transform:
        if max_depth == 0:
            return self

        ndim = x.ndim - 1

        x_ = x.reshape([len(x), -1])
        x_ = x_[:, x_.isfinite().all(0)]

        if 'channels' not in self.shared:
            pmin = torch.min(x_, dim=-1).values
            pmax = torch.max(x_, dim=-1).values
        else:
            pmin = torch.min(x_)
            pmax = torch.max(x_)
        pmin = pmin[(Ellipsis,) + (None,) * ndim]
        pmax = pmax[(Ellipsis,) + (None,) * ndim]

        slope = (self.vmax - self.vmin) / (pmax - pmin)
        offset = self.vmin - pmin * slope

        if self.clip:
            return SequentialTransform([
                AddMulTransform(slope, offset, **self.get_prm()),
                ClipTransform(self.vmin, self.vmax, **self.get_prm())
            ]).make_final(x, max_depth-1)
        else:
            return AddMulTransform(
                slope, offset, **self.get_prm()
            ).make_final(x, max_depth-1)
