__all__ = [
    'GaussianNoiseTransform',
    'RandomGaussianNoiseTransform',
    'ChiNoiseTransform',
    'RandomChiNoiseTransform',
    'GammaNoiseTransform',
    'RandomGammaNoiseTransform',
    'GFactorTransform',
]

import torch
import math
from .baseutils import prepare_output
from .base import FinalTransform, NonFinalTransform
from .special import RandomizedTransform
from .intensity import MulFieldTransform, AddValueTransform, MulValueTransform
from .random import Uniform, RandInt, Fixed, make_range
from . import ctx


class GaussianNoiseTransform(NonFinalTransform):
    """Additive Gaussian noise"""

    class Final(AddValueTransform):

        def __init__(self, value, **kwargs):
            super().__init__(value, value_name='noise', **kwargs)
            self.Parent = GaussianNoiseTransform

        @property
        def noise(self):
            return self.value

        @noise.setter
        def noise(self, value):
            self.value = value

    def __init__(self, sigma=0.1, *, shared=False, **kwargs):
        """
        Parameters
        ----------
        sigma : float
            Standard deviation

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensors to return
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Add the exact same nosie to all channels/images
        """
        super().__init__(shared=shared, **kwargs)
        self.sigma = sigma

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        shape = list(x.shape)
        if 'channels' in self.shared:
            shape[0] = 1
        dtype = x.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        noise = torch.randn(shape, dtype=dtype, device=x.device)
        noise = noise.mul_(self.sigma)
        return self.Final(
            noise, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomGaussianNoiseTransform(RandomizedTransform):
    """Additive Gaussian noise with random standard deviation"""

    def __init__(self, sigma=0.1,
                 *, shared=False, shared_noise=None, **kwargs):
        """
        Parameters
        ----------
        sigma : Sampler or float
            Distribution from which to sample the standard deviation.
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
        super().__init__(GaussianNoiseTransform,
                         dict(sigma=Uniform.make(make_range(0, sigma))),
                         shared=shared, **kwargs)
        self.shared_noise = shared_noise

    def get_prm(self):
        prm = super().get_prm()
        prm['shared'] = (
            self.shared if self.shared_noise is None else
            self.shared_noise
        )
        return prm


class ChiNoiseTransform(NonFinalTransform):
    """Additive Noncentral Chi noise

    (Rician is a special case with nb_channels = 2)
    """

    class Final(FinalTransform):

        def __init__(self, noise, **kwargs):
            super().__init__(**kwargs)
            self.noise = noise
            self.Parent = ChiNoiseTransform

        def apply(self, x):
            noise = self.noise.to(x)
            y = x.square().add_(noise.square()).sqrt_()
            return prepare_output(
                dict(input=x, output=y, noise=noise),
                self.returns
            )

    def __init__(self, sigma=0.1, nb_channels=2, *, shared=False, **kwargs):
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
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Add the exact same values to all channels/images
        """
        super().__init__(shared=shared, **kwargs)
        self.sigma = sigma
        self.nb_channels = nb_channels

    def make_final(self, x, max_depth=float('inf')):
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
            noise += torch.randn(
                shape, dtype=dtype, device=x.device
            ).square_()
        noise = noise.sqrt_()
        noise *= self.sigma / math.sqrt(df - mu*mu)

        return self.Final(
            noise, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomChiNoiseTransform(RandomizedTransform):
    """Additive Chi noise with random standard deviation and channels"""

    def __init__(self, sigma=0.1, nb_channels=8,
                 *, shared=False, shared_noise=None, **kwargs):
        """
        Parameters
        ----------
        sigma : Sampler or float
            Distribution from which to sample the standard deviation.
            If a `float`, sample from `Uniform(0, value)`.
            To use a fixed value, pass `Fixed(value)`.
        nb_channels : Sampler or int
            Distribution from which to sample the standard deviation.
            If a `int`, sample from `RandInt(1, value)`.
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
        super().__init__(
            ChiNoiseTransform,
            dict(sigma=Uniform.make(make_range(0, sigma)),
                 nb_channels=RandInt.make(make_range(2, nb_channels))),
            shared=shared,
            **kwargs
        )
        self.shared_noise = shared_noise

    def get_prm(self):
        prm = super().get_prm()
        prm['shared'] = (
            self.shared if self.shared_noise is None else
            self.shared_noise
        )
        return prm


class GFactorTransform(NonFinalTransform):

    class Final(NonFinalTransform):

        def __init__(self, noisetrf, gfactor, **kwargs):
            super().__init__(**kwargs)
            self.noisetrf = noisetrf
            self.gfactor = gfactor
            self.Parent = GFactorTransform

        @property
        def is_final(self):
            return self.noisetrf.is_final and self.gfactor.is_final

        def make_final(self, x, max_depth=float('inf')):
            if max_depth == 0 or self.is_final:
                return self
            return type(self)(
                self.noisetrf.make_final(x, max_depth-1),
                self.gfactor.make_final(x, max_depth-1),
            ).make_final(x, max_depth-1)

        def apply(self, x):
            noisetrf = self.noisetrf.make_final(x)
            with ctx.returns(noisetrf, 'noise'):
                noise = noisetrf(x)
            with ctx.returns(self.gfactor, ['output', 'field']):
                scalednoise, gfactor = self.gfactor(noise)
            self.noisetrf.noise = scalednoise
            y = self.noisetrf(x)
            self.noisetrf.noise = noise
            return prepare_output(
                dict(input=x, output=y, noise=noise,
                     scalednoise=scalednoise, gfactor=gfactor),
                self.returns
            )

    def __init__(self, noise, shape=5, vmin=0.5, vmax=1.5,
                 *, shared=False, **kwargs):
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

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output', 'gfactor', 'noise', 'scalednoise'}
            Which tensors to return
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Use the same field for all channels/tensors
        """  # noqa: 501
        super().__init__(shared=shared, **kwargs)
        self.noise = noise
        self.gfactor = MulFieldTransform(
            shape, vmin=vmin, vmax=vmax, shared=shared
        )

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        return self.Final(
            self.noise.make_final(x, max_depth-1),
            self.gfactor.make_final(x, max_depth-1),
            **self.get_prm(),
        ).make_final(x, max_depth-1)


class GammaNoiseTransform(NonFinalTransform):
    """Multiplicative Gamma noise"""

    class Final(MulValueTransform):
        def __init__(self, value, **kwargs):
            super().__init__(value, value_name='noise', **kwargs)
            self.Parent = GammaNoiseTransform

        @property
        def noise(self):
            return self.value

        @noise.setter
        def noise(self, value):
            self.value = value

    def __init__(self, sigma=0.1, mean=1, *, shared=False, **kwargs):
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

    def make_final(self, x, max_depth=float('inf')):
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
        noise = torch.distributions.Gamma(alpha, beta).sample(shape)
        return self.Final(
            noise, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomGammaNoiseTransform(RandomizedTransform):
    """Multiplicative Gamma noise with random standard deviation and mean"""

    def __init__(self, sigma=0.1, mean=Fixed(1),
                 *, shared=False, shared_noise=None, **kwargs):
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

    def get_prm(self):
        prm = super().get_prm()
        prm['shared'] = (
            self.shared if self.shared_noise is None else
            self.shared_noise
        )
        return prm
