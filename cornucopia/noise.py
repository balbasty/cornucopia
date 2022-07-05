import torch
from .base import Transform
from .intensity import MultFieldTransform


__all__ = ['GaussianNoiseTransform', 'ChiNoiseTransform']


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
