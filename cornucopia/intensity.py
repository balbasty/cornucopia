__all__ = [
    'AddValueTransform',
    'MulValueTransform',
    'AddMulTransform',
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
    'GammaTransform',
    'RandomGammaTransform',
    'ZTransform',
    'QuantileTransform',
]

import torch
from torch.nn.functional import interpolate
import math
import interpol
from .baseutils import prepare_output
from .base import FinalTransform, NonFinalTransform
from .special import RandomizedTransform, SequentialTransform
from .random import Sampler, Uniform, RandInt, Fixed, make_range
from .utils.py import ensure_list, positive_index


class OpConstTransform(FinalTransform):
    """Base class for arithmetic operations with a constant value"""
    _op = None
    _inv = {
        torch.add: lambda x: -x,
        torch.mul: lambda x: 1/x,
    }

    def __init__(self, value, op=None, value_name='value', **kwargs):
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

    def apply(self, x):
        value = self.value
        if torch.is_tensor(value):
            value = value.to(x)
        y = self.op(x, value)
        return prepare_output(
            {'input': x, 'output': y, self.value_name: value}, self.returns
        )

    def make_inverse(self):
        inv = self._inv[self.op]
        return type(self)(
            inv(self.value), **self.get_prm(), value_name=self.value_name
        )


class AddValueTransform(OpConstTransform):
    """Add a constant value"""
    _op = torch.add


class MulValueTransform(OpConstTransform):
    """Multiply with a constant value"""
    _op = torch.mul


class FillValueTransform(FinalTransform):
    """Fills the tensor with a value inside a mask"""

    def __init__(self, mask, value, mask_name='mask', value_name='value',
                 **kwargs):
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

    def apply(self, x):
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


class AddMulTransform(FinalTransform):
    """Constant intensity affine transform"""

    def __init__(self, slope=1, offset=0, **kwargs):
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

    def apply(self, x):
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

    def make_inverse(self):
        return AddMulTransform(
            1/self.slope, -self.offset/self.slope, **self.get_prm()
        )


class ClipTransform(FinalTransform):
    """Clip extremum values"""

    def __init__(self, vmin=None, vmax=None, **kwargs):
        """
        Parameters
        ----------
        vmin : number or tensor
            Min value
        vmax : number or tensor
            Max valur
        """
        super().__init__(**kwargs)
        self.vmin = vmin
        self.vmax = vmax

    def apply(self, x):
        vmin, vmax = self.vmin, self.vmax
        if torch.is_tensor(vmin):
            vmin = vmin.to(x)
        if torch.is_tensor(vmax):
            vmax = vmax.to(x)
        y = x.clamp(vmin, vmax)
        return prepare_output(
            {'input': x, 'output': y, 'vmin': vmin, 'vmax': vmax},
            self.returns
        )


class RandomMulTransform(RandomizedTransform):
    """
    Random multiplicative transform.
    """

    def __init__(self, value=(0.5, 2), *, shared=True, **kwargs):
        """

        Parameters
        ----------
        value : Sampler or [pair of] float
            Bound for multiplicative value
        shared : {'channels', 'tensors', 'channels+tensors', ''}
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

    def __init__(self, value=1, *, shared=True, **kwargs):
        """

        Parameters
        ----------
        value : Sampler or [pair of] float
            Bound for additive value
        shared : {'channels', 'tensors', 'channels+tensors', ''}
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

    def __init__(self, slope=1, offset=0.5, *, shared=True, **kwargs):
        """

        Parameters
        ----------
        slope : Sampler or [pair of] float
            Bound for slope
        offset : Sampler or [pair of] float
            Bound for offset
        shared : {'channels', 'tensors', 'channels+tensors', None}
            Apply same transform to all images/channels
        """
        super().__init__(
            AddMulTransform,
            (Uniform.make(make_range(slope)), Uniform.make(make_range(offset))),
            shared=shared,
            **kwargs
        )


class SplineUpsampleTransform(FinalTransform):
    """Upsample a field using spline interpolation"""

    def __init__(self, order=3, prefilter=False, **kwargs):
        """
        Parameters
        ----------
        order : int
            Spline interpolation order
        prefilter : bool
            Splie prefiltering
            (True for interpolation, False for spline evaluation)
        """
        super().__init__(**kwargs)
        self.order = order
        self.prefilter = prefilter

    def apply(self, x):
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

    finalklass = AddValueTransform
    value_name = 'field'

    def __init__(self, shape=5, vmin=0, vmax=1, order=3,
                 slice=None, thickness=None,
                 *, shared=False, **kwargs):
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

    def make_field(self, batch, smallshape, fullshape=None, **backend):
        """Generate the random coefficients"""
        smallshape = ensure_list(smallshape, len(fullshape))
        smallshape = [min(small, full) for small, full
                      in zip(smallshape, fullshape)]
        if not backend['dtype'].is_floating_point:
            backend['dtype'] = torch.get_default_dtype()
        b = torch.rand([batch, *smallshape], **backend)
        if fullshape:
            b = self.upsample_field(b, fullshape)
        return b

    def upsample_field(self, coeff, fullshape):
        """Resize spline coefficients to full size"""
        if self.order == 1:
            mode = ('trilinear' if len(fullshape) == 3 else
                    'bilinear' if len(fullshape) == 2 else
                    'linear')
            b = interpolate(
                coeff.unsqueeze(0), fullshape, mode=mode,
                align_corners=True
            ).squeeze(-0)
        else:
            b = interpol.resize(
                coeff, shape=fullshape, interpolation=self.order,
                prefilter=False
            )
        return b

    def make_final(self, x, max_depth=float('inf')):
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

            if self.thickness == 1:
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

        b.mul_(self.vmax-self.vmin).add_(self.vmin)

        return self.finalklass(
            b, value_name=self.value_name, **self.get_prm()
        ).make_final(x, max_depth-1)


class MulFieldTransform(BaseFieldTransform):
    """Smooth multiplicative (bias) field"""
    finalklass = MulValueTransform


class RandomMulFieldTransform(NonFinalTransform):
    """Random multiplicative bias field transform"""

    def __init__(self, shape=8, vmax=1, order=3, symmetric=False, *,
                 shared=False, shared_field=None, **kwargs):
        """
        Parameters
        ----------
        shape : Sampler or int
            Sampler or Upper bound for number of control points
        vmax : Sampler or float
            Sampler or Upper bound for maximum value
        order : int
            Spline order
        symmetric : bool or float
            If a float, the bias field will take values in
            `(symmetric-vmax, symmetric+vmax)`.
            If False, it will take values in `(0, vmax)`.
            If True, it will take values in `(1-vmax, 1+vmax)`.

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'field'}
            Which tensor(s) to return
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Whether to share random parameters across tensors and/or channels
        shared_field : {'channels', 'tensors', 'channels+tensors', '', None}
            Whether to share random field across tensors and/or channels.
            By default: same as `shared`
        """
        super().__init__(shared=shared, **kwargs)
        self.vmax = Uniform.make(make_range(0, vmax))
        self.shape = RandInt.make(make_range(2, shape))
        self.order = Fixed.make(order)
        self.symmetric = symmetric
        self.shared_field = self._prepare_shared(shared_field)

    def make_final(self, x, max_depth=float('inf')):
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

    def __init__(self, shape=8, vmax=1, order=3, slice=None, thickness=32,
                 shape_through=None, *, shared=False, shared_field=None,
                 **kwargs):
        """
        Parameters
        ----------
        shape : Sampler or int
            Sampler or Upper bound for number of control points
        vmax : Sampler or float
            Sampler or Upper bound for maximum value
        order : int
            Spline order
        slice : int
            Slice axis. If None, sample one randomly
        thickness:
            Sampler or Upper bound for slice thickness
        shape_through : Sampler or int
            Sampler or Upper bound for number of control points
            along the slice direction. If None, same as `shape`.

        Other Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Whether to share random parameters across tensors and/or channels
        shared_field : {'channels', 'tensors', 'channels+tensors', '', None}
            Whether to share random field across tensors and/or channels.
            By default: same as `shared`
        """
        super().__init__(shared=shared, **kwargs)
        if shape_through is not None:
            shape_through = RandInt.make(make_range(1, shape_through))
        self.vmax = Uniform.make(make_range(0, vmax))
        self.shape = RandInt.make(make_range(2, shape))
        self.order = Fixed.make(order)
        self.slice = slice
        self.thickness = RandInt.make(make_range(0, thickness))
        self.shape_through = shape_through
        self.shared_field = self._prepare_shared(shared_field)

    def make_final(self, x, max_depth=float('inf')):
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
    finalklass = AddValueTransform


class RandomAddFieldTransform(NonFinalTransform):
    """Random additive bias field transform"""

    def __init__(self, shape=8, vmin=-1, vmax=1, order=3, *,
                 shared=False, shared_field=None, **kwargs):
        """
        Parameters
        ----------
        shape : Sampler or int
            Sampler or Upper bound for number of control points
        vmin : Sampler or float
            Sampler or Lower bound for minimum value
        vmax : Sampler or float
            Sampler or Upper bound for maximum value
        order : int
            Spline order

        Other Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Whether to share random parameters across tensors and/or channels
        shared_field : {'channels', 'tensors', 'channels+tensors', '', None}
            Whether to share random field across tensors and/or channels.
            By default: same as `shared`
        """
        super().__init__(shared=shared, **kwargs)
        self.vmin = Uniform.make(make_range(vmin, 0))
        self.vmax = Uniform.make(make_range(0, vmax))
        self.shape = RandInt.make(make_range(2, shape))
        self.order = Fixed.make(order)
        self.shared_field = self._prepare_shared(shared_field)

    def make_final(self, x, max_depth=float('inf')):
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


class GammaTransform(NonFinalTransform):
    """Gamma correction

    References
    ----------
    1. https://en.wikipedia.org/wiki/Gamma_correction
    """

    class FinalGammaTransform(FinalTransform):

        def __init__(self, gamma, vmin, vmax, **kwargs):
            super().__init__(**kwargs)
            self.gamma = gamma
            self.vmin = vmin
            self.vmax = vmax

        def apply(self, x):
            vmin = torch.as_tensor(self.vmin, dtype=x.dtype, device=x.device)
            vmax = torch.as_tensor(self.vmax, dtype=x.dtype, device=x.device)
            gamma = torch.as_tensor(self.gamma, dtype=x.dtype, device=x.device)
            vmin = vmin.reshape([-1] + [1] * (x.ndim-1))
            vmax = vmax.reshape([-1] + [1] * (x.ndim-1))
            gamma = gamma.reshape([-1] + [1] * (x.ndim-1))

            y = x.sub(vmin).div_(vmax - vmin)
            y = y.pow_(gamma)
            y = y.mul_(vmax - vmin).add_(vmin)

            return prepare_output(
                dict(input=x, output=y, vmin=vmin, vmax=vmax, gamma=gamma),
                self.returns)

    def __init__(self, gamma=1, vmin=None, vmax=None,
                 *, shared=False, **kwargs):
        """

        Parameters
        ----------
        gamma : float
            Exponent of the Gamma transform
        vmin : float
            Value to use as the minimum (default: x.min())
        vmax : float
            Value to use as the maximum (default: x.max())
        returns : [list or dict] {'input', 'output', 'vmin', 'vmax', 'gamma'}
            Which tensors to return

        Other Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Use the same vmin/vmax for all channels

        """
        super().__init__(shared=shared, **kwargs)
        self.gamma = kwargs.pop('value', gamma)
        self.vmin = vmin
        self.vmax = vmax

    def make_final(self, x, max_depth=float('inf')):
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
        return self.FinalGammaTransform(
            self.gamma, vmin, vmax, **self.get_prm()
        ).make_final(max_depth-1)


class RandomGammaTransform(NonFinalTransform):
    """
    Random Gamma transform.
    """

    def __init__(self, gamma=(0.5, 2), *, shared=False, shared_minmax=None,
                 **kwargs):
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

    def make_final(self, x, max_depth=float('inf')):
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

    def __init__(self, mu=0, sigma=1, *, shared=False, **kwargs):
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

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' in self.shared:
            opt = dict()
        else:
            opt = dict(dim=list(range(1, x.ndim)), keepdim=True)
        mu, sigma = x.mean(**opt), x.std(**opt)
        return AddMulTransform(
            1/sigma, -mu/sigma, **self.get_prm()
        ).make_final(x, max_depth-1)


class QuantileTransform(NonFinalTransform):
    """Match lower and upper quantiles to (0, 1)"""

    def __init__(self, pmin=0.01, pmax=0.99, vmin=0, vmax=1,
                 clip=False, **kwargs):
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
        """
        super().__init__(**kwargs)
        self.pmin = pmin
        self.pmax = pmax
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth)
        
        nmax = 10000
        x = x[x != 0]
        x = x[torch.rand_like(x) < (nmax / x.numel())]
        pmin = torch.quantile(x, self.pmin)
        pmax = torch.quantile(x, self.pmax)

        slope = (self.vmax - self.vmin) / (pmax - pmin)
        offset = self.vmin - pmin * slope

        if self.clip:
            return SequentialTransform(
                AddMulTransform(slope, offset, **self.get_prm()),
                ClipTransform(self.vmin, self.vmax, **self.get_prm())
            ).make_final(x, max_depth-1)
        else:
            return AddMulTransform(
                slope, offset, **self.get_prm()
            ).make_final(x, max_depth-1)
