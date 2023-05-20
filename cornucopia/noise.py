__all__ = ['GaussianNoiseTransform', 'RandomGaussianNoiseTransform',
           'ChiNoiseTransform', 'RandomChiNoiseTransform',
           'GammaNoiseTransform', 'RandomGammaNoiseTransform',
           'GFactorTransform']

import torch
from .base import Transform, RandomizedTransform, prepare_output
from .intensity import MultFieldTransform
from .random import Uniform, RandInt, Fixed, make_range


class GaussianNoiseTransform(Transform):
    """Additive Gaussian noise"""

    def __init__(self, sigma=0.1, *, shared=False, **kwargs):
        """

        Parameters
        ----------
        sigma : float
            Standard deviation
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensors to return
        shared : bool
            Add the exact same values to all channels/images
        """
        super().__init__(shared=shared, **kwargs)
        self.sigma = sigma

    def get_parameters(self, x):
        return torch.randn_like(x).mul_(self.sigma)

    def apply_transform(self, x, parameters):
        y = x + parameters
        return prepare_output(dict(input=x, output=y, noise=parameters),
                              self.returns)


class RandomGaussianNoiseTransform(RandomizedTransform):
    """Additive Gaussian noise with random standard deviation"""

    def __init__(self, sigma=0.1, *, shared=False, **kwargs):
        """
        Parameters
        ----------
        sigma : Sampler or float
            Sampler or upper bound for the standard deviation
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensors to return
        shared : bool
            Use the same sd for all channels/images
        """
        super().__init__(GaussianNoiseTransform,
                         dict(sigma=Uniform.make(make_range(0, sigma)),
                              **kwargs),
                         shared=shared)


class ChiNoiseTransform(Transform):
    """Additive Noncentral Chi noise

    (Rician is a special case with nb_channels = 2)
    """

    def __init__(self, sigma=0.1, nb_channels=2, *, shared=False, **kwargs):
        """
        Parameters
        ----------
        sigma : float
            Standard deviation
        nb_channels : int
            Number of independent channels
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensors to return
        shared : bool
            Add the exact same values to all channels/images
        """
        super().__init__(shared=shared, **kwargs)
        self.sigma = sigma
        self.nb_channels = nb_channels

    def get_parameters(self, x):
        noise = 0
        for _ in range(self.nb_channels):
            noise += torch.randn_like(x).mul_(self.sigma).square_()
        noise /= self.nb_channels
        return noise.sqrt_()

    def apply_transform(self, x, parameters):
        y = x.square().add_(parameters.square()).sqrt_()
        return prepare_output(dict(input=x, output=y, noise=parameters),
                              self.returns)


class RandomChiNoiseTransform(RandomizedTransform):
    """Additive Chi noise with random standard deviation and channels"""

    def __init__(self, sigma=0.1, nb_channels=8, *, shared=False, **kwargs):
        """
        Parameters
        ----------
        sigma : Sampler or float
            Sampler or upper bound for the standard deviation
        nb_channels : Sampler or int
            Sampler or upper bound for the number of channels
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensors to return
        shared : bool
            Use the same sd for all channels/images
        """
        super().__init__(ChiNoiseTransform,
                         dict(sigma=Uniform.make(make_range(0, sigma)),
                              nb_channels=RandInt.make(make_range(nb_channels, 2)),
                              **kwargs),
                         shared=shared)


class GFactorTransform(Transform):

    def __init__(self, noise, shape=5, vmin=1, vmax=4, *, returns=None, **kwargs):
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
        returns : [list or dict of] {'input', 'output', 'gfactor', 'noise', 'scalednoise'}
            Which tensors to return
        """
        super().__init__(returns=returns, **kwargs)
        self.noise = noise
        self.gfactor = MultFieldTransform(shape, vmin=vmin, vmax=vmax)

    def get_parameters(self, x):
        noisetrf, noise = self.noise, self.noise.get_parameters(x)
        if isinstance(noise, Transform):
            noisetrf, noise = noise, noise.get_parameters(x)
        gfactor = self.gfactor.get_parameters(x)
        return noisetrf, noise, gfactor

    def apply_transform(self, x, parameters):
        noisetrf, noise, gfactor = parameters
        scalednoise = noise * gfactor
        y = noisetrf.apply_transform(x, noise * gfactor)
        return prepare_output(
            dict(input=x, output=y, gfactor=gfactor,
                 noise=noise, scalednoise=scalednoise),
            self.returns)


class GammaNoiseTransform(Transform):
    """Multiplicative Gamma noise"""

    def __init__(self, mean=1, sigma=0.1, *, shared=False, **kwargs):
        """

        Parameters
        ----------
        mean : float
            Expected value
        sigma : float
            Standard deviation
        returns : [list or dict of] {'input', 'output', 'noise'}
            Which tensors to return
        shared : bool
            Add the exact same values to all channels/images
        """
        super().__init__(shared=shared, **kwargs)
        self.mean = mean
        self.sigma = sigma

    def get_parameters(self, x):
        var = self.sigma * self.sigma
        beta = self.mean / var
        alpha = self.mean * beta
        alpha = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
        beta = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
        return torch.distributions.Gamma(alpha, beta).sample(x.shape)

    def apply_transform(self, x, parameters):
        y = x * parameters
        return prepare_output(dict(input=x, outptu=y, noise=parameters),
                              self.returns)


class RandomGammaNoiseTransform(RandomizedTransform):
    """Multiplicative Gamma noise with random standard deviation and mean"""

    def __init__(self, mean=Fixed(1), sigma=0.1, *, shared=False, **kwargs):
        """
        Parameters
        ----------
        mean : Sampler or float
            Sampler or upper bound for the mean
        sigma : Sampler or float
            Sampler or upper bound for the standard deviation
        shared : bool
            Use the same mean/sd for all channels/images
        """
        super().__init__(GammaNoiseTransform,
                         dict(mean=Uniform.make(make_range(0, mean)),
                              sigma=Uniform.make(make_range(0, sigma)),
                              **kwargs),
                         shared=shared)
