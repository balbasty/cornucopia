__all__ = ['Sampler', 'Fixed', 'Uniform', 'RandInt', 'Normal', 'LogNormal']

import random
import copy
import torch
from .utils.py import ensure_list


class Sampler:
    """
    Base class for random samplers, with a bunch of helpers.
    This class is for developers of new Sampler classes only.
    """

    def __init__(self, **theta):
        self.theta = theta

    @classmethod
    def make(cls, other):
        if isinstance(other, Sampler):
            return other
        elif isinstance(other, dict):
            return cls(**other)
        elif isinstance(other, tuple):
            return cls(*other)
        else:
            return cls(other)

    def _ensure_same_length(self, theta, nsamples=None):
        theta = dict(theta)
        if nsamples:
            for k, v in theta.items():
                theta[k] = ensure_list(theta[k], nsamples)
        elif any(isinstance(v, (list, tuple)) for v in theta.values()):
            nsamples = 0
            for k, v in theta.items():
                theta[k] = ensure_list(v)
                nsamples = max(nsamples, len(theta[k]))
            for k, v in theta.items():
                theta[k] = ensure_list(theta[k], nsamples)
        return theta

    @classmethod
    def map(cls, fn, *values, n=None):
        if n:
            values = tuple(ensure_list(v, n) for v in values)
        if isinstance(values[0], list):
            return [fn(*args) for args in zip(*values)]
        else:
            return fn(*values)

    def __getattr__(self, item):
        theta = self.__getattribute__('theta')
        if item in theta:
            theta = self._ensure_same_length(theta)
            return theta[item]
        raise AttributeError(item)

    def __setattr__(self, item, value):
        if item == 'theta':
            return super().__setattr__(item, value)
        if item in self.theta:
            self.theta[item] = value
        else:
            return super().__setattr__(item, value)

    def __call__(self, n=None, **backend):
        """
        Parameters
        ----------
        n : int or list[int], optional
            Number of values to sample
            - if None, return a scalar
            - if an int, return a list
            - if a list, return a tensor

        Returns
        -------
        sample : number or list[number] or tensor

        """
        raise NotImplementedError


class Fixed(Sampler):
    """Fixed value"""
    def __init__(self, value):
        """
        Parameters
        ----------
        value : number or sequence[number]
        """
        super().__init__(value=value)

    def __call__(self, n=None, **backend):
        if isinstance(n, (list, tuple)):
            return torch.full(n, self.value, **backend)
        return self.map(copy.deepcopy, self.value, n=n)


class Uniform(Sampler):
    """Continuous uniform sampler"""

    def __init__(self, min, max):
        """
        Parameters
        ----------
        min : float or sequence[float]
            Lower bound (inclusive)
        max : float or sequence[float]
            Upper bound (inclusive or exclusive, depending on rounding)
        """
        super().__init__(min=min, max=max)

    def __call__(self, n=None, **backend):
        if isinstance(n, (list, tuple)):
            return torch.rand(n, **backend).mul_(self.max - self.min).add_(self.min)
        return self.map(random.uniform, self.min, self.max, n=n)


class RandInt(Sampler):
    """Discrete uniform sampler"""

    def __init__(self, min, max):
        """
        Parameters
        ----------
        min : float or sequence[float]
            Lower bound (inclusive)
        max : float or sequence[float]
            Upper bound (inclusive)
        """
        super().__init__(min=min, max=max)

    def __call__(self, n=None, **backend):
        if isinstance(n, (list, tuple)):
            return torch.randint(1 + self.max - self.min, n, **backend).add_(self.min)
        return self.map(random.randint, self.min, self.max, n=n)


class Normal(Sampler):
    """Gaussian sampler"""

    def __init__(self, mu, sigma):
        """
        Parameters
        ----------
        mu : float or sequence[float]
            Mean
        sigma : float or sequence[float]
            Standard deviation
        """
        super().__init__(mu=mu, sigma=sigma)

    def __call__(self, n=None, **backend):
        if isinstance(n, (list, tuple)):
            return torch.randn(n, **backend).mul_(self.sigma).add_(self.mu)
        return self.map(random.normalvariate, self.mu, self.sigma, n=n)


class LogNormal(Sampler):
    """LogNormal sampler"""

    def __init__(self, mu, sigma):
        """
        Parameters
        ----------
        mu : float or sequence[float]
            Mean of the log
        sigma : float or sequence[float]
            Standard deviation of the log
        """
        super().__init__(mu=mu, sigma=sigma)

    def __call__(self, n=None, **backend):
        if isinstance(n, (list, tuple)):
            return torch.randn(n, **backend).mul_(self.sigma).add_(self.mu).exp_()
        return self.map(random.lognormvariate, self.mu, self.sigma, n=n)
