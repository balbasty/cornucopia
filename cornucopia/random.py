import random
import copy
from .utils.py import ensure_list


__all__ = ['Sampler', 'Fixed', 'Uniform', 'RandInt', 'Normal', 'LogNormal']


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

    def __getattribute__(self, item):
        theta = super().__getattribute__('theta')
        if item in theta:
            theta = self._ensure_same_length(theta)
            return theta[item]
        return super().__getattribute__(item)

    def __setattr__(self, item, value):
        if item == 'theta':
            return super().__setattr__(item, value)
        if item in self.theta:
            self.theta[item] = value
        else:
            return super().__setattr__(item, value)


class Fixed(Sampler):
    """Fixed value"""
    def __init__(self, value):
        super().__init__(value=value)

    def __call__(self, n=None):
        return self.map(copy.deepcopy, self.value, n=n)


class Uniform(Sampler):
    """Continuous uniform sampler"""

    def __init__(self, a, b):
        super().__init__(a=a, b=b)

    def __call__(self, n=None):
        return self.map(random.uniform, self.a, self.b, n=n)


class RandInt(Sampler):
    """Discrete uniform sampler"""

    def __init__(self, a, b):
        super().__init__(a=a, b=b)

    def __call__(self, n=None):
        return self.map(random.randint, self.a, self.b, n=n)


class Normal(Sampler):
    """Gaussian sampler"""

    def __init__(self, mu, sigma):
        super().__init__(mu=mu, sigma=sigma)

    def __call__(self, n=None):
        return self.map(random.normalvariate, self.mu, self.sigma, n=n)


class LogNormal(Sampler):
    """LogNormal sampler"""

    def __init__(self, mu, sigma):
        super().__init__(mu=mu, sigma=sigma)

    def __call__(self, n=None):
        return self.map(random.lognormvariate, self.mu, self.sigma, n=n)
