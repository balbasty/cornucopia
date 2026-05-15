"""This module contains transforms that inject noise into an image."""
__all__ = [
    'GaussianNoiseFinalTransform',
    'GaussianNoiseTransform',
    'RandomGaussianNoiseTransform',
    'ChiNoiseFinalTransform',
    'ChiNoiseTransform',
    'RandomChiNoiseTransform',
    'GFactorFinalTransform',
    'GFactorTransform',
    'GammaNoiseFinalTransform',
    'GammaNoiseTransform',
    'RandomGammaNoiseTransform',
]
# stdlib
import math
from math import inf

# dependencies
import torch
import typing_extensions as tx
from torch import Tensor

# internals
from .baseutils import prepare_output
from .base import FinalTransform, NonFinalTransform, PerChannelTransform, Transform
from .special import RandomizedTransform
from .intensity import MulFieldTransform, AddValueTransform, MulValueTransform
from .random import Uniform, RandInt, Fixed, make_range
from .utils.smart_inplace import mul_, add_, sqrt_
from . import typing as cct
from . import ctx


def _parentof(child):

    def decorator(parent):
        child.Previous = parent
        return parent

    return decorator



class GaussianNoiseFinalTransform(AddValueTransform):
    """Precomputed Gaussian noise transform"""

    def __init__(self, value: Tensor, **kwargs) -> None:
        super().__init__(value, value_name='noise', **kwargs)


@_parentof(GaussianNoiseFinalTransform)
class GaussianNoiseTransform(NonFinalTransform):
    """Additive Gaussian noise"""

    Final = Next = GaussianNoiseFinalTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        sigma: float = 0.1,
        *,
        shared: cct.SharedType = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        sigma : float
            Standard deviation

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensors to return
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Add the exact same nosie to all channels/images
        """
        super().__init__(shared=shared, **kwargs)
        self.sigma = sigma

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        shape = list(x.shape)
        if 'channels' in self.shared:
            shape[0] = 1
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        noise = torch.randn(shape, dtype=dtype, device=x.device)
        noise = mul_(noise, self.sigma)
        return self.Next(
            noise, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomGaussianNoiseTransform(RandomizedTransform):
    """Additive Gaussian noise with random standard deviation"""

    Next = GaussianNoiseTransform
    """The transform type returned by `make_final(..., max_depth=1)`."""

    Final = GaussianNoiseFinalTransform
    """The transform type returned by `make_final(..., max_depth=inf)`."""

    def __init__(
        self,
        sigma: cct.SamplerOrBound[float] = 0.1,
        *,
        shared: cct.SharedType=False,
        shared_noise: tx.Optional[cct.SharedType] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        sigma : Sampler | float
            Distribution from which to sample the standard deviation.
            If a `float`, sample from `Uniform(0, value)`.
            To use a fixed value, pass `Fixed(value)`.

        Other Parameters
        ----------------
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensors to return
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Use the same sd for all channels/tensors
        shared_noise : {'channels', 'tensors', 'channels+tensors', '', None}
            Use the exact same noise for all channels/tensors
        """
        super().__init__(GaussianNoiseTransform,
                         dict(sigma=Uniform.make(make_range(0, sigma))),
                         shared=shared, **kwargs)
        self.shared_noise = shared_noise

    def get_prm(self) -> dict:
        prm = super().get_prm()
        prm['shared'] = (
            self.shared if self.shared_noise is None else
            self.shared_noise
        )
        return prm


class ChiNoiseFinalTransform(FinalTransform):

    def __init__(self, noise: Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.noise = noise

    def xform(self, x):
        noise = self.noise.to(x)
        y = sqrt_(add_(x.square(), noise.square()))
        return prepare_output(
            dict(input=x, output=y, noise=noise),
            self.returns
        )


@_parentof(ChiNoiseFinalTransform)
class ChiNoiseTransform(NonFinalTransform):
    """Additive Noncentral Chi noise

    (Rician is a special case with nb_channels = 2)
    """

    Final = Next = ChiNoiseFinalTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        sigma: float = 0.1,
        nb_channels: int = 2,
        *,
        shared: cct.SharedType = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        sigma : float
            Standard deviation
        nb_channels : int
            Number of independent channels

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensors to return
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Add the exact same values to all channels/images
        """
        super().__init__(shared=shared, **kwargs)
        self.sigma = sigma
        self.nb_channels = nb_channels

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self

        shape = list(x.shape)
        if 'channels' in self.shared:
            shape[0] = 1
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()

        df = self.nb_channels
        mu = math.sqrt(2) * math.gamma((df+1)/2) / math.gamma(df/2)
        noise = 0
        for _ in range(self.nb_channels):
            noise += torch.randn(shape, dtype=dtype, device=x.device).square_()
        noise = noise.sqrt_()

        # scale to reach target variance
        sigma = self.sigma / math.sqrt(df - mu*mu)
        noise = mul_(noise, sigma)

        return self.Next(
            noise, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomChiNoiseTransform(RandomizedTransform):
    """Additive Chi noise with random standard deviation and channels"""

    Next = ChiNoiseTransform
    """The transform type returned by `make_final(..., max_depth=1)`."""

    Final = ChiNoiseFinalTransform
    """The transform type returned by `make_final(..., max_depth=inf)`."""

    def __init__(
        self,
        sigma: cct.SamplerOrBound[float] = 0.1,
        nb_channels: cct.SamplerOrBound[int] = 8,
        *, shared: cct.SharedType = False,
        shared_noise: tx.Optional[cct.SharedType] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        sigma : Sampler | float
            Distribution from which to sample the standard deviation.
            If a `float`, sample from `Uniform(0, value)`.
            To use a fixed value, pass `Fixed(value)`.
        nb_channels : Sampler | int
            Distribution from which to sample the standard deviation.
            If a `int`, sample from `RandInt(1, value)`.
            To use a fixed value, pass `Fixed(value)`.

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensors to return
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Use the same sd for all channels/tensors
        shared_noise : {'channels', 'tensors', 'channels+tensors', '', None} | bool | None
            Use the exact same noise for all channels/tensors
        """
        super().__init__(
            ChiNoiseTransform,
            dict(sigma=Uniform.make(make_range(0, sigma)),
                 nb_channels=RandInt.make(make_range(2, nb_channels))),
            shared=shared,
            **kwargs
        )
        self.shared_noise = shared_noise

    def get_prm(self) -> dict:
        prm = super().get_prm()
        prm['shared'] = (
            self.shared if self.shared_noise is None else
            self.shared_noise
        )
        return prm


class GFactorFinalTransform(NonFinalTransform):
    """Multiplicative noise with precomputed noise and g-factor"""

    def __init__(
        self, noisetrf: Transform, gfactor: Transform, **kwargs
    ) -> None:
        """
        Parameters
        ----------
        noisetrf : Transform
            A transform that applies additive noise
        gfactor : Transform
            A transform that takes the noise as input and outputs a
            g-factor field
        """
        super().__init__(**kwargs)
        self.noisetrf = noisetrf
        self.gfactor = gfactor

    @property
    def is_final(self) -> bool:
        return self.noisetrf.is_final and self.gfactor.is_final

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0 or self.is_final:
            return self
        return type(self)(
            self.noisetrf.make_final(x, max_depth-1),
            self.gfactor.make_final(x, max_depth-1),
        ).make_final(x, max_depth-1)

    def xform(self, x: Tensor) -> Tensor:
        noisetrf = self.noisetrf.make_final(x)
        with ctx.returns(noisetrf, 'noise'):
            noise = noisetrf(x)
        with ctx.returns(self.gfactor, ['output', 'field']):
            scalednoise, gfactor = self.gfactor(noise)
        if isinstance(noisetrf, PerChannelTransform):
            # FIXME this is messy -- hope this works in most cases
            noisetrf = noisetrf.transforms[0]
        scalednoisetrf = type(noisetrf)(scalednoise)
        y = scalednoisetrf(x)
        return prepare_output(
            dict(input=x, output=y, noise=noise,
                    scalednoise=scalednoise, gfactor=gfactor),
            self.returns
        )


@_parentof(GFactorFinalTransform)
class GFactorTransform(NonFinalTransform):

    Final = Next = GFactorFinalTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        noise: Transform,
        shape: int = 5,
        vmin: float = 0.5,
        vmax: float = 1.5,
        order: int = 3,
        *,
        shared: cct.SharedType = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        noise : Transform
            A transform that applies additive noise
        shape : float
            Number of control points
        vmin : float
            Minimum g-factor
        vmax : float
            Maximum g-factor
        order : int
            Spline order

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'gfactor', 'noise', 'scalednoise'}
            Which tensors to return
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Use the same field for all channels/tensors
        """  # noqa: 501
        super().__init__(shared=shared, **kwargs)
        self.noise = noise
        self.gfactor = MulFieldTransform(
            shape, vmin=vmin, vmax=vmax, order=order, shared=shared
        )

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        return self.Next(
            self.noise.make_final(x, max_depth-1),
            self.gfactor.make_final(x, max_depth-1),
            **self.get_prm(),
        ).make_final(x, max_depth-1)


class GammaNoiseFinalTransform(MulValueTransform):
    """Multiplicative noise with precomputed noise"""

    def __init__(self, value, **kwargs):
        super().__init__(value, value_name='noise', **kwargs)


@_parentof(GammaNoiseFinalTransform)
class GammaNoiseTransform(NonFinalTransform):
    """Multiplicative Gamma noise"""

    Final = Next = GammaNoiseFinalTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        sigma: float = 0.1,
        mean: float = 1,
        *,
        shared: cct.SharedType = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        sigma : float
            Standard deviation
        mean : float
            Expected value

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensors to return
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Use the exact same noise for all channels/tensors
        """
        super().__init__(shared=shared, **kwargs)
        self.mean = mean
        self.sigma = sigma

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        shape = list(x.shape)
        if 'channels' in self.shared:
            shape[0] = 1
        var = self.sigma * self.sigma
        beta = self.mean / var
        alpha = self.mean * beta
        alpha = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
        beta = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
        noise = torch.distributions.Gamma(alpha, beta).rsample(shape)
        # ^ rsample() allows backprop, whereas sample() does not
        return self.Next(
            noise, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomGammaNoiseTransform(RandomizedTransform):
    """Multiplicative Gamma noise with random standard deviation and mean"""

    Next = GammaNoiseTransform
    """The transform type returned by `make_final(..., max_depth=1)`."""

    Final = GammaNoiseFinalTransform
    """The transform type returned by `make_final(..., max_depth=inf)`."""

    def __init__(
        self,
        sigma: cct.SamplerOrBound[float] = 0.1,
        mean: cct.SamplerOrBound[float] = Fixed(1.0),
        *,
        shared: cct.SharedType = False,
        shared_noise: tx.Optional[cct.SharedType] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        sigma : Sampler or float
            Distribution from which to sample the standard deviation.
            If a `float`, sample from `Uniform(0, value)`.
            To use a fixed value, pass `Fixed(value)`.
        mean : Sampler or float
            Distribution from which to sample the mean.
            If a `float`, sample from `Uniform(0, value)`.
            To use a fixed value, pass `Fixed(value)`.

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensors to return
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Use the same sd for all channels/tensors
        shared_noise : {'channels', 'tensors', 'channels+tensors', '', None}
            Use the exact same noise for all channels/tensors
        """
        super().__init__(GammaNoiseTransform,
                         dict(mean=Uniform.make(make_range(0, mean)),
                              sigma=Uniform.make(make_range(0, sigma))),
                         shared=shared, **kwargs)
        self.shared_noise = shared_noise

    def get_prm(self) -> dict:
        prm = super().get_prm()
        prm['shared'] = (
            self.shared if self.shared_noise is None else
            self.shared_noise
        )
        return prm
