__all__ = ['GaussianNoiseTransform', 'RandomGaussianNoiseTransform',
           'ChiNoiseTransform', 'RandomChiNoiseTransform',
           'GammaNoiseTransform', 'RandomGammaNoiseTransform',
           'GFactorTransform']

import torch
from .base import Transform, RandomizedTransform
from .intensity import MultFieldTransform
from .random import Uniform, RandInt, upper_range


class GaussianNoiseTransform(Transform):
    """Additive Gaussian noise"""

    def __init__(self, sigma=0.1, shared=False):
        """

        Parameters
        ----------
        sigma : float
            Standard deviation
        shared : bool
            Add the exact same values to all channels/images
        """
        super().__init__(shared=shared)
        self.sigma = sigma

    def get_parameters(self, x):
        return torch.randn_like(x).mul_(self.sigma)

    def apply_transform(self, x, parameters):
        return x + parameters


class RandomGaussianNoiseTransform(RandomizedTransform):
    """Additive Gaussian noise with random standard deviation"""

    def __init__(self, sigma=0.1, shared=False):
        """
        Parameters
        ----------
        sigma : Sampler or float
            Sampler or upper bound for the standard deviation
        shared : bool
            Use the same sd for all channels/images
        """
        super().__init__(GaussianNoiseTransform,
                         dict(sigma=Uniform.make(upper_range(sigma))),
                         shared=shared)


class ChiNoiseTransform(Transform):
    """Additive Noncentral Chi noise

    (Rician is a special case with nb_channels = 2)
    """

    def __init__(self, sigma=0.1, nb_channels=2, shared=False):
        """
        Parameters
        ----------
        sigma : float
            Standard deviation
        nb_channels : int
            Number of independent channels
        shared : bool
            Add the exact same values to all channels/images
        """
        super().__init__(shared=shared)
        self.sigma = sigma
        self.nb_channels = nb_channels

    def get_parameters(self, x):
        noise = 0
        for _ in range(self.nb_channels):
            noise += torch.randn_like(x).mul_(self.sigma).square_()
        noise /= self.nb_channels
        return noise

    def apply_transform(self, x, parameters):
        return x.square().add_(parameters).sqrt_()


class RandomChiNoiseTransform(RandomizedTransform):
    """Additive Chi noise with random standard deviation and channels"""

    def __init__(self, sigma=0.1, nb_channels=8, shared=False):
        """
        Parameters
        ----------
        sigma : Sampler or float
            Sampler or upper bound for the standard deviation
        nb_channels : Sampler or int
            Sampler or upper bound for the number of channels
        shared : bool
            Use the same sd for all channels/images
        """
        super().__init__(ChiNoiseTransform,
                         dict(sigma=Uniform.make(upper_range(sigma)),
                              nb_channels=RandInt.make(upper_range(nb_channels, 2))),
                         shared=shared)


class GFactorTransform(Transform):

    def __init__(self, noise, shape=5, vmin=1, vmax=4):
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
        """
        super().__init__()
        self.noise = noise
        self.gfactor = MultFieldTransform(shape, vmin=vmin, vmax=vmax)

    def get_parameters(self, x):
        noisetrf, noise = self.noise, self.noise.get_parameters(x)
        if isinstance(noise, Transform):
            noisetrf, noise = noise, noise.get_parameters(x)
        gfactor = self.gfactor.get_parameters(x)
        return noisetrf, noise * gfactor

    def apply_transform(self, x, parameters):
        noisetrf, noiseprm = parameters
        return noisetrf.apply_transform(x, noiseprm)


class GammaNoiseTransform(Transform):
    """Multiplicative Gamma noise"""

    def __init__(self, mean=1, sigma=0.1, shared=False):
        """

        Parameters
        ----------
        mean : float
            Expected value
        sigma : float
            Standard deviation
        shared : bool
            Add the exact same values to all channels/images
        """
        super().__init__(shared=shared)
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
        return x * parameters


class RandomGammaNoiseTransform(RandomizedTransform):
    """Multiplicative Gamma noise with random standard deviation and mean"""

    def __init__(self, mean=2, sigma=0.1, shared=False):
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
                         dict(mean=Uniform.make(upper_range(mean)),
                              sigma=Uniform.make(upper_range(sigma))),
                         shared=shared)