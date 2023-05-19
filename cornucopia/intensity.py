__all__ = ['AddFieldTransform', 'MultFieldTransform', 'BaseFieldTransform',
           'RandomAddFieldTransform', 'RandomMultFieldTransform',
           'GlobalMultTransform', 'RandomGlobalMultTransform',
           'GlobalAdditiveTransform', 'RandomGlobalAdditiveTransform',
           'GammaTransform', 'RandomGammaTransform',
           'ZTransform', 'QuantileTransform']

import torch
from torch.nn.functional import interpolate
import interpol
from .base import Transform, RandomizedTransform, prepare_output
from .random import Sampler, Uniform, RandInt, Fixed, make_range
from .utils.py import ensure_list


class BaseFieldTransform(Transform):

    def __init__(self, shape=5, vmin=0, vmax=1, order=3,
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
        returns : [list or dict of] {'input', 'output', 'field'}
            Which tensor(s) to return
        shared : bool
            Apply the same field to all channels
        """
        super().__init__(shared=shared, **kwargs)
        self.shape = shape
        self.vmax = vmax
        self.vmin = vmin
        self.order = order

    def get_parameters(self, x):
        batch, *fullshape = x.shape
        smallshape = ensure_list(self.shape, len(fullshape))
        backend = dict(dtype=x.dtype, device=x.device)
        if not backend['dtype'].is_floating_point:
            backend['dtype'] = torch.get_default_dtype()
        b = torch.rand([batch, *smallshape], **backend)
        if self.order == 1:
            mode = ('trilinear' if len(fullshape) == 3 else
                    'bilinear' if len(fullshape) == 2 else
                    'linear')
            b = interpolate(b[None], fullshape, mode=mode, align_corners=True)[0]
        else:
            b = interpol.resize(b, shape=fullshape, interpolation=self.order,
                                prefilter=False)
        b.mul_(self.vmax-self.vmin).add_(self.vmin)
        return b


class MultFieldTransform(BaseFieldTransform):
    """Smooth multiplicative (bias) field"""

    def apply_transform(self, x, parameters):
        y = x * parameters
        return prepare_output(dict(input=x, output=y, field=parameters),
                              self.returns)


class RandomMultFieldTransform(RandomizedTransform):
    """Random multiplicative bias field transform"""

    def __init__(self, shape=8, vmax=1, order=3, shared=False):
        """
        Parameters
        ----------
        shape : Sampler or int
            Sampler or Upper bound for number of control points
        vmax : Sampler or float
            Sampler or Upper bound for maximum value
        order : int
            Spline order
        shared : bool
            Whether to share random parameters across channels
        """
        kwargs = dict(vmax=Uniform.make(make_range(0, vmax)),
                      shape=RandInt.make(make_range(2, shape)),
                      order=Fixed.make(order))
        super().__init__(MultFieldTransform, kwargs, shared=shared)


class AddFieldTransform(BaseFieldTransform):
    """Smooth additive (bias) field"""

    def apply_transform(self, x, parameters):
        y = x + parameters
        return prepare_output(dict(input=x, output=y, field=parameters),
                              self.returns)


class RandomAddFieldTransform(RandomizedTransform):
    """Random additive bias field transform"""

    def __init__(self, shape=8, vmin=-1, vmax=1, order=3,
                 *, shared=False, **kwargs):
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
        shared : bool
            Whether to share random parameters across channels
        """
        kwargs = dict(vmax=Uniform.make(make_range(0, vmax)),
                      vmin=Uniform.make(make_range(vmin, 0)),
                      shape=RandInt.make(make_range(2, shape)),
                      order=Fixed.make(order),
                      **kwargs)
        super().__init__(AddFieldTransform, kwargs, shared=shared)


class GlobalMultTransform(Transform):
    """Global multiplicative transform"""

    def __init__(self, value=1, *, shared=False, **kwargs):
        """

        Parameters
        ----------
        value : float
            Multiplicative value
        shared : bool
            Apply the same field to all channels
        """
        super().__init__(shared=shared, **kwargs)
        self.value = value

    def apply_transform(self, x, parameters):
        y = x * self.value
        value = x.new_full([1]*x.ndim, self.value)
        return prepare_output(dict(input=x, output=y, value=value),
                              self.returns)


class RandomGlobalMultTransform(RandomizedTransform):
    """
    Random multiplicative transform.
    """

    def __init__(self, value=(0.5, 2), *, shared=True, **kwargs):
        """

        Parameters
        ----------
        value : Sampler or [pair of] float
            Max multiplicative value
        shared : bool
            Apply same transform to all images/channels
        """
        kwargs['value'] = Uniform.make(make_range(0, value))
        super().__init__(GlobalMultTransform, kwargs, shared=shared)


class GlobalAdditiveTransform(Transform):
    """Global additive transform"""

    def __init__(self, value=0, shared=False):
        """

        Parameters
        ----------
        value : float
            Additive value
        shared : bool
            Apply the same field to all channels
        """
        super().__init__(shared=shared)
        self.value = value

    def apply_transform(self, x, parameters):
        y = x + self.value
        value = x.new_full([1]*x.ndim, self.value)
        return prepare_output(dict(input=x, output=y, value=value),
                              self.returns)


class RandomGlobalAdditiveTransform(RandomizedTransform):
    """
    Random additive transform.
    """

    def __init__(self, value=1, shared=True):
        """

        Parameters
        ----------
        value : Sampler or [pair of] float
            Max Additive value
        shared : bool
            Apply same transform to all images/channels
        """
        super().__init__(GlobalAdditiveTransform,
                         Uniform.make(make_range(value)),
                         shared=shared)


class GammaTransform(Transform):
    """Gamma correction

    References
    ----------
    .. https://en.wikipedia.org/wiki/Gamma_correction
    """

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
        returns : [list or dict of] {'input', 'output', 'vmin', 'vmax', 'gamma'}
            Which tensors to return
        shared : bool
            Use the same vmin/vmax for all channels
        """
        super().__init__(shared=shared, **kwargs)
        self.gamma = gamma
        self.vmin = vmin
        self.vmax = vmax

    def get_parameters(self, x):
        ndim = x.dim() - 1
        if self.vmin is None:
            vmin = x.reshape(len(x), -1).min(-1).values
            for _ in range(ndim):
                vmin = vmin.unsqueeze(-1)
        else:
            vmin = self.vmin
        if self.vmax is None:
            vmax = x.reshape(len(x), -1).max(-1).values
            for _ in range(ndim):
                vmax = vmax.unsqueeze(-1)
        else:
            vmax = self.vmax
        return vmin, vmax

    def apply_transform(self, x, parameters):
        vmin, vmax = parameters

        y = x.sub(vmin).div_(vmax - vmin)
        y = y.pow_(self.gamma)
        y = y.mul_(vmax - vmin).add_(vmin)

        vmin = torch.as_tensor(vmin, dtype=x.dtype, device=x.device)
        vmax = torch.as_tensor(vmax, dtype=x.dtype, device=x.device)
        gamma = torch.as_tensor(vmax, dtype=x.dtype, device=x.device)
        vmin = vmin.reshape([-1] + [1] * (x.ndim-1))
        vmax = vmin.reshape([-1] + [1] * (x.ndim-1))
        gamma = gamma.reshape([-1] + [1] * (x.ndim-1))

        return prepare_output(
            dict(input=x, output=y, vmin=vmin, vmax=vmax, gamma=gamma),
            self.returns)


class RandomGammaTransform(RandomizedTransform):
    """
    Random Gamma transform.
    """

    def __init__(self, value=(0.5, 2), shared=True):
        """
        Parameters
        ----------
        value : Sampler or [pair of] float
            Sampler or range for the exponent value
        shared : bool
            Apply same transform to all images/channels
        """
        super().__init__(GammaTransform,
                         Uniform.make(value),
                         shared=shared)


class ZTransform(Transform):
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
        shared : bool
            Apply same transform to all images/channels
        """
        super().__init__(shared=shared, **kwargs)
        self.mu = mu
        self.sigma = sigma

    def get_parameters(self, x):
        return x.mean(), x.std()

    def apply_transform(self, x, parameters):
        mu, sigma = parameters
        x = (x - mu) / sigma
        if self.sigma != 1:
            sigma = self.sigma or sigma
            x *= sigma
        if self.mu != 0:
            mu = self.mu or mu
            x += mu
        return x


class QuantileTransform(Transform):
    """Match lower and upper quantiles to (0, 1)"""

    def __init__(self, pmin=0.01, pmax=0.99, vmin=0, vmax=1,
                 clamp=False, *, shared=False, **kwargs):
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
        clamp : bool
            Clamp values outside (vmin, vmax)
        """
        super().__init__(shared=shared, **kwargs)
        self.pmin = pmin
        self.pmax = pmax
        self.vmin = vmin
        self.vmax = vmax
        self.clamp = clamp

    def get_parameters(self, x):
        nmax = 10000
        x = x[x != 0]
        x = x[torch.rand_like(x) < (nmax / x.numel())]
        pmin = torch.quantile(x, q=self.pmin)
        pmax = torch.quantile(x, q=self.pmax)
        return pmin, pmax

    def apply_transform(self, x, parameters):
        pmin, pmax = parameters
        x = x.sub(pmin).div_(pmax - pmin)
        if self.clamp:
            x = x.clamp_(0, 1)
        x = x.mul_(self.vmax - self.vmin).add_(self.vmin)
        return x
