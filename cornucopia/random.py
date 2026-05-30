__all__ = [
    'Sampler',
    'Fixed',
    'Uniform',
    'RandInt',
    'Normal',
    'LogNormal',
    'RandKFrom',
    "UniformSphere",
    "min",
    "max",
    "sum",
    "exp",
    "log",
]
# stdlib
import copy
import functools
import math
import random
from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Callable, Protocol, Sequence, overload

# external
import torch

# internal
from .utils.py import ensure_list
from .utils.smart_inplace import add_, div_, mul_, exp_, square_

# aliases
_builtin_min, _builtin_max, _builitin_sum = min, max, sum


class Sampler(ABC):
    """
    Base class for random samplers, with a bunch of helpers.

    !!! note
        Samplers can be combined with determinstic values or between
        themselves using arithmetic operators:

        ```python
        P + X -> ShiftedSampler(P, X)
        P - X -> ShiftedSampler(P, -X)
        P * X -> MultipliedSampler(P, X)
        P / X -> DividedSampler(P, X)
        X / P -> DividedSampler(X, P)
        P ** X -> PoweredSampler(P, X)
        X ** P -> ExponentSampler(X, P)

        P + Q -> SumOfSamplers(P, Q)
        P - Q -> DifferenceOfSamplers(P, Q)
        P * Q -> ProductOfSamplers(P, Q)
        P / Q -> RatioOfSamplers(P, Q)
        P ** Q -> PowerOfSamplers(P, Q)

        min(P, Q, ...) -> MinimumOfSamplers(P, Q, ...)
        max(P, Q, ...) -> MaximumOfSamplers(P, Q, ...)
        sum(P, Q, ...) -> SumOfSamplers(P, Q, ...)
        exp(P)         -> ExponentiatedSampler(P)
        log(P)         -> LogarithmOfSampler(P)
        ```

    Attributes
    ----------
    theta : dict
        Parameters of the sampler
    """

    class Parameters(dict):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._ensure_same_length(inplace=True)

        def __getattr__(self, name):
            if name in self.keys():
                return self[name]
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name}"
            )

        def __setattr__(self, name, value):
            if name in self.keys():
                self[name] = value
            else:
                super().__setattr__(name, value)

        def _ensure_same_length(self, nsamples=None, inplace=False):
            theta = self
            if not inplace:
                theta = type(self)(**theta)

            if nsamples:
                for k, v in theta.items():
                    theta[k] = ensure_list(v, nsamples)

            elif any(isinstance(v, (list, tuple)) for v in theta.values()):
                nsamples = 0
                for k, v in theta.items():
                    theta[k] = ensure_list(v)
                    nsamples = _builtin_max(nsamples, len(theta[k]))
                for k, v in theta.items():
                    theta[k] = ensure_list(theta[k], nsamples)

            return theta

    def __init__(self, **theta):
        self.theta = self.Parameters(**theta)

    @overload
    @classmethod
    def make(cls, other: 'Sampler') -> 'Sampler':
        """Pass-through factory: return the same sampler"""

    @overload
    @classmethod
    def make(cls, other: dict) -> 'Sampler':
        """Keyword factory: return `cls(**other)`"""

    @overload
    @classmethod
    def make(cls, other: tuple) -> 'Sampler':
        """Positional factory: return `cls(*other)`"""

    @overload
    @classmethod
    def make(cls, other: Any) -> 'Sampler':
        """Fallback factory: return `cls(other)`"""

    @classmethod
    def make(cls, other):
        """
        Build a sampler from another sample, or from parameters.
        """
        if isinstance(other, Sampler):
            return other
        elif isinstance(other, dict):
            return cls(**other)
        elif isinstance(other, tuple):
            return cls(*other)
        else:
            return cls(other)

    @classmethod
    def _map(cls, fn, *values, n=None):
        if n:
            values = tuple(ensure_list(v, n) for v in values)
        if isinstance(values[0], list):
            return [fn(*args) for args in zip(*values)]
        else:
            return fn(*values)

    def __getattr__(self, item):
        if item in self.theta:
            return self.theta[item]
        raise AttributeError(
            f"{type(self).__name__} has no attribute {item}"
        )

    def __setattr__(self, item, value):
        # Set known attributes by name
        if item == 'theta':
            return super().__setattr__(item, value)
        if item in self.theta:
            self.theta[item] = value
        else:
            return super().__setattr__(item, value)

    @overload
    def __call__(self) -> Number: ...

    @overload
    def __call__(self, n: int) -> Sequence[Number]: ...

    @overload
    def __call__(
        self,
        n: Sequence[int],
        *,
        dtype=None,
        device=None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def __call__(self, n=None, **backend):
        """
        Parameters
        ----------
        n : int or list[int]
            Number of values to sample

            - if `None`, return a scalar
            - if an `int`, return a list
            - if a `list`, return a tensor

        Other parameters
        ----------------
        dtype : torch.dtype, optional
            Data type of the returned tensor. By default, inferred from
            the sampler parameters. Only used if `n` is a list.
        device : torch.device, optional
            Device of the returned tensor. By default, inferred from
            the sampler parameters. Only used if `n` is a list.

        Returns
        -------
        sample : number or list[number] or tensor

        """
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, Sampler):
            return ProductOfSamplers(self, other)
        else:
            return MultipliedSampler(self, other)

    def __rmul__(self, other):
        if isinstance(other, Sampler):
            return ProductOfSamplers(other, self)
        else:
            return MultipliedSampler(self, other)

    def __truediv__(self, other):
        if isinstance(other, Sampler):
            return RatioOfSamplers(self, other)
        else:
            return DividedSampler(self, other)

    def __rtruediv__(self, other):
        if isinstance(other, Sampler):
            return RatioOfSamplers(other, self)
        else:
            return DividingSampler(self, other)

    def __add__(self, other):
        if isinstance(other, Sampler):
            return SumOfSamplers(self, other)
        else:
            return ShiftedSampler(self, other)

    def __radd__(self, other):
        if isinstance(other, Sampler):
            return SumOfSamplers(other, self)
        else:
            return ShiftedSampler(self, other)

    def __sub__(self, other):
        if isinstance(other, Sampler):
            return DifferenceOfSamplers(self, other)
        else:
            return ShiftedSampler(self, -other)

    def __rsub__(self, other):
        if isinstance(other, Sampler):
            return DifferenceOfSamplers(other, self)
        else:
            return ReverseShiftedSampler(self, other)

    def __neg__(self):
        return MultipliedSampler(self, -1)

    def __pow__(self, other):
        return PoweredSampler(self, other)

    def __rpow__(self, other):
        return ExponentSampler(self, other)

    def exp(self):
        return ExponentiatedSampler(self)

    def log(self):
        return LogarithmOfSampler(self)


def _check_same(new, old):
    if old is None:
        return new
    if new is None:
        return old
    if new is old:
        return old
    if torch.is_tensor(new) or torch.is_tensor(old):
        old = torch.as_tensor(old)
        new = torch.as_tensor(new)
        check = (old == new).all()
    else:
        check = (old == new)
    if not check:
        raise ValueError(f"Conflicting values for {old} and {new}")
    return old if old is not None else new


def _ensure_same_length(*args, nsamples=None):
    if nsamples:
        return tuple(ensure_list(arg, nsamples) for arg in args)
    elif any(isinstance(arg, (list, tuple)) for arg in args):
        nsamples = 0
        for arg in args:
            if arg is None:
                continue
            arg = ensure_list(arg)
            nsamples = _builtin_max(nsamples, len(arg))
        return tuple(
            ensure_list(arg, nsamples)
            if arg is not None else None
            for arg in args
        )
    else:
        return args


class Fixed(Sampler):
    """Fixed value.

    Attributes
    ----------
    value : number or sequence[number]
    """

    class Parameters(Sampler.Parameters):

        @classmethod
        def make(cls, **kwargs):

            value = None
            value = _check_same(kwargs.pop("value", None), value)
            value = _check_same(kwargs.pop("mean", None), value)
            value = _check_same(kwargs.pop("median", None), value)
            value = _check_same(kwargs.pop("min", None), value)
            value = _check_same(kwargs.pop("max", None), value)

            _check_same(kwargs.pop("std", 0), 0)
            _check_same(kwargs.pop("var", 0), 0)
            _check_same(kwargs.pop("fwhm", 0), 0)

            if value is None:
                raise ValueError("No value provided for Fixed sampler")
            if kwargs:
                raise TypeError(f"got unexpected keyword arguments: {kwargs}")
            return cls(value=value)

        @property
        def mean(self):
            return self.value

        median = mode = min = mean

        @property
        def std(self):
            return Sampler.map(lambda _: 0.0, self.value)

        var = fwhm = std

    @overload
    def __init__(self, *, mean, std=0): ...

    @overload
    def __init__(self, *, mean, var=0): ...

    @overload
    def __init__(self, *, mean, fwhm=0): ...

    @overload
    def __init__(self, *, min, max=None): ...

    def __init__(self, value=None, **kwargs):
        kwargs["value"] = value
        return super().__init__(**self.Parameters.make(**kwargs))

    def _use_torch(self, n):
        return (
            torch.is_tensor(self.value) or
            isinstance(n, (list, tuple))
        )

    def __call__(self, n=None, **backend):
        if self._use_torch(n):
            n = tuple(ensure_list(n or []))
            if not torch.is_tensor(self.value):
                return torch.full(n, self.value, **backend)
            else:
                value = self.value[(None,) * len(n)]
                n = n + (1,) * self.value.ndim
                return torch.tile(value, n)
        return self._map(copy.deepcopy, self.value, n=n)


class Uniform(Sampler):
    """Continuous uniform sampler.

    Attributes
    ----------
    min : float or sequence[float], default=0
        Lower bound (inclusive)
    max : float or sequence[float], default=1
        Upper bound (inclusive or exclusive, depending on rounding)
    """

    class Parameters(Sampler.Parameters):

        @classmethod
        def make(cls, **kwargs):
            min = kwargs.pop("min", None)
            max = kwargs.pop("max", None)
            min, max = _ensure_same_length(min, max)

            mean = fwhm = None
            if min is not None and max is not None:
                _minmax2mean = lambda x: _builitin_sum(x) / len(x)
                _minmax2fwhm = lambda x: x[1] - x[0]
                if isinstance(min, list):
                    mean = _check_same(list(map(_minmax2mean, zip(min, max))), None)
                    fwhm = _check_same(list(map(_minmax2fwhm, zip(min, max))), None)
                else:
                    mean = _check_same(_minmax2mean((min, max)), None)
                    fwhm = _check_same(_minmax2fwhm((min, max)), None)

            mean = _check_same(kwargs.pop("mean", None), mean)
            mean = _check_same(kwargs.pop("median", None), mean)
            mean = _check_same(kwargs.pop("mode", None), mean)

            fwhm = _check_same(kwargs.pop("fwhm", None), fwhm)
            if "std" in kwargs:
                _std2fwhm = lambda x: math.sqrt(8 * math.log(2)) * x
                std = kwargs.pop("std")
                if isinstance(std, (list, tuple)):
                    fwhm = _check_same(list(map(_std2fwhm, std)), fwhm)
                else:
                    fwhm = _check_same(_std2fwhm(std), fwhm)
            if "var" in kwargs:
                _var2fwhm = lambda x: math.sqrt(8 * math.log(2) * x)
                var = kwargs.pop("var")
                if isinstance(var, (list, tuple)):
                    fwhm = _check_same(list(map(_var2fwhm, var)), fwhm)
                else:
                    fwhm = _check_same(_var2fwhm(var), fwhm)

            if min is None and max is None:
                if fwhm is None:
                    fwhm = 1
                if mean is None:
                    min, max = 0, fwhm
                else:
                    min = mean - fwhm / 2
                    max = mean + fwhm / 2
            elif min is None:
                if fwhm is not None:
                    min = max - fwhm
                elif mean is not None:
                    min = mean - (max - mean)
            elif max is None:
                if fwhm is not None:
                    max = min + fwhm
                else:
                    max = mean + (mean - min)

            if min is None:
                min = 0
            if max is None:
                max = 1

            if kwargs:
                raise TypeError(f"got unexpected keyword arguments: {kwargs}")
            return cls(min=min, max=max)

        @property
        def mean(self):
            _mean = lambda x: _builitin_sum(x) / len(x)
            if isinstance(self.min, list):
                return list(map(_mean, zip(self.min, self.max)))
            return _mean((self.min, self.max))

        @property
        def median(self):
            return self.mean

        @property
        def mode(self):
            return self.mean

        @property
        def std(self):
            _std = lambda x: (x[1] - x[0]) / math.sqrt(12)
            if isinstance(self.min, list):
                return list(map(_std, zip(self.min, self.max)))
            return _std((self.min, self.max))

        @property
        def var(self):
            _var = lambda x: (x[1] - x[0]) ** 2 / 12
            if isinstance(self.min, list):
                return list(map(_var, zip(self.min, self.max)))
            return _var((self.min, self.max))

        @property
        def fwhm(self):
            _fwhm = lambda x: x[1] - x[0]
            if isinstance(self.min, list):
                return list(map(_fwhm, zip(self.min, self.max)))
            return _fwhm((self.min, self.max))

    @overload
    def __init__(self): ...

    @overload
    def __init__(self, max): ...

    @overload
    def __init__(self, min, max): ...

    @overload
    def __init__(self, *, mean, fwhm=1): ...

    @overload
    def __init__(self, *, mean, std): ...

    @overload
    def __init__(self, *, mean, var): ...

    def __init__(self, *args, **kwargs):
        if len(args) == 2:
            kwargs["min"], kwargs["max"] = args
        elif len(args) == 1:
            kwargs["max"] = args[0]
        elif len(args) != 0:
            raise TypeError(f"Got unexpected positional arguments: {args}")
        return super().__init__(**self.Parameters.make(**kwargs))

    def _use_torch(self, n):
        return (
            torch.is_tensor(self.min) or
            torch.is_tensor(self.max) or
            isinstance(n, (list, tuple))
        )

    def __call__(self, n=None, **backend):
        if self._use_torch(n):
            n = tuple(ensure_list(n or []))
            x = torch.rand(n, **backend)
            x = add_(mul_(x, self.max - self.min), self.min)
            return x
        return self._map(random.uniform, self.min, self.max, n=n)


class RandInt(Sampler):
    """Discrete uniform sampler

    Attributes
    ----------
    min : int or sequence[int], default=0
        Lower bound (inclusive)
    max : int or sequence[int]
        Upper bound (inclusive)
    """

    class Parameters(Sampler.Parameters):

        @property
        def mean(self):
            _mean = lambda x: _builitin_sum(x) / len(x)
            if isinstance(self.min, list):
                return list(map(_mean, zip(self.min, self.max)))
            return _mean((self.min, self.max))

        @property
        def median(self):
            return self.mean

        @property
        def mode(self):
            return self.mean

        @property
        def std(self):
            _std = lambda x: ((x[1] - x[0] + 1) ** 2 - 1) ** 2 / math.sqrt(12)
            if isinstance(self.min, list):
                return list(map(_std, zip(self.min, self.max)))
            return _std((self.min, self.max))

        @property
        def var(self):
            _var = lambda x: ((x[1] - x[0] + 1) ** 2 - 1) / 12
            if isinstance(self.min, list):
                return list(map(_var, zip(self.min, self.max)))
            return _var((self.min, self.max))

        @property
        def fwhm(self):
            _fwhm = lambda x: x[1] - x[0] + 1
            if isinstance(self.min, list):
                return list(map(_fwhm, zip(self.min, self.max)))
            return _fwhm((self.min, self.max))

    @overload
    def __init__(self): ...

    @overload
    def __init__(self, max): ...

    @overload
    def __init__(self, min, max): ...

    def __init__(self, *args, **kwargs):
        min = max = None
        # Parse min/max arguments
        if len(args) == 2:
            min, max = args
        elif len(args) == 1:
            max = args[0]
        if 'min' in kwargs:
            min = kwargs.pop('min')
        if 'max' in kwargs:
            max = kwargs.pop('max')
        if kwargs:
            raise TypeError(f"got unexpected keyword arguments: {kwargs}")
        # Set defaults
        if min is None:
            min = 0
        if max is None:
            max = 1
        super().__init__(min=min, max=max)

    def _use_torch(self, n):
        return (
            torch.is_tensor(self.min) or
            torch.is_tensor(self.max) or
            isinstance(n, (list, tuple))
        )

    def __call__(self, n=None, **backend):
        if self._use_torch(n):
            n = tuple(ensure_list(n or []))
            backend.setdefault(
                'device',
                getattr(self.max - self.min, 'device', None)
            )
            return torch.randint(
                low=self.min, high=1 + self.max, size=n, **backend
            )
        return self._map(random.randint, self.min, self.max, n=n)


class RandKFrom(Sampler):
    """Discrete uniform sampler

    Attributes
    ----------
    range : sequence
        Values from which to sample
    k : int, default=None
        Number of values to sample.
        Sample random number if None.
    replacement : bool, default=False
        Whether to sample with replacement
    """

    def __init__(self, range, k=None, replacement=False):
        super().__init__()
        range = list(range)
        if not replacement and k and k > len(range):
            raise ValueError(
                'Cannot sample more element than available. '
                'To sample with replacement, use `replacement=True`'
            )
        self.range = range
        self.k = k
        self.replacement = replacement

    def __call__(self, n=None, **backend):
        k = self.k or RandInt(1, len(self.range))()
        if isinstance(n, (list, tuple)) or n:
            raise ValueError('RandKFrom cannot sample multiple elements')
        if not self.replacement:
            range = list(self.range)
            random.shuffle(range)
            return range[:k]
        else:
            index = RandInt(0, len(self.range)-1)(k)
            return [self.range[i] for i in index]


class Normal(Sampler):
    """Gaussian sampler.

    Attributes
    ----------
    mu : float or sequence[float], default=0
        Mean
    sigma : float or sequence[float], default=1
        Standard deviation
    """

    class Parameters(Sampler.Parameters):

        @classmethod
        def make(cls, **kwargs):
            mu, sigma = kwargs.pop("mu", None), kwargs.pop("sigma", None)
            mu, sigma = _ensure_same_length(mu, sigma)

            mu = _check_same(kwargs.pop("mean", None), mu)
            mu = _check_same(kwargs.pop("median", None), mu)
            mu = _check_same(kwargs.pop("mode", None), mu)

            sigma = _check_same(kwargs.pop("std", None), sigma)
            if "var" in kwargs:
                _var2std = lambda x: x ** 0.5
                var = kwargs.pop("var")
                if isinstance(var, (list, tuple)):
                    sigma = _check_same(list(map(_var2std, var)), sigma)
                else:
                    sigma = _check_same(_var2std(var), sigma)
            if "fwhm" in kwargs:
                _fwhm2std = lambda x: x / math.sqrt(8 * math.log(2))
                fwhm = kwargs.pop("fwhm")
                if isinstance(fwhm, (list, tuple)):
                    sigma = _check_same(list(map(_fwhm2std, fwhm)), sigma)
                else:
                    sigma = _check_same(_fwhm2std(fwhm), sigma)

            if mu is None:
                mu = 0
            if sigma is None:
                sigma = 1

            if kwargs:
                raise TypeError(f"got unexpected keyword arguments: {kwargs}")
            return cls(mu=mu, sigma=sigma)

        @property
        def mean(self):
            return self.mu

        @property
        def median(self):
            return self.mu

        @property
        def mode(self):
            return self.mu

        @property
        def std(self):
            return self.sigma

        @property
        def var(self):
            if isinstance(self.sigma, list):
                return [s ** 2 for s in self.sigma]
            return self.sigma ** 2

        @property
        def fwhm(self):
            if isinstance(self.sigma, list):
                return [s * math.sqrt(8 * math.log(2)) for s in self.sigma]
            return self.sigma * math.sqrt(8 * math.log(2))

    @overload
    def __init__(self, mu=0, sigma=1): ...

    @overload
    def __init__(self, *, mean, std=1): ...

    @overload
    def __init__(self, *, mean, var): ...

    @overload
    def __init__(self, *, mean, fwhm): ...

    def __init__(self, mu=None, sigma=None, **kwargs):
        super().__init__(**self.Parameters.make(mu=mu, sigma=sigma, **kwargs))

    def _use_torch(self, n):
        return (
            torch.is_tensor(self.mu) or
            torch.is_tensor(self.sigma) or
            isinstance(n, (list, tuple))
        )

    def __call__(self, n=None, **backend):
        if self._use_torch(n):
            n = tuple(ensure_list(n or []))
            x = torch.randn(n, **backend)
            x = add_(mul_(x, self.sigma), self.mu)
            return x
        return self._map(random.normalvariate, self.mu, self.sigma, n=n)


class LogNormal(Sampler):
    """LogNormal sampler

    Attributes
    ----------
    mu : float or sequence[float], default=0
        Mean of the log
    sigma : float or sequence[float], default=1
        Standard deviation of the log
    """

    class Parameters(Sampler.Parameters):

        @classmethod
        def make(cls, **kwargs):
            mu, sigma = kwargs.pop("mu", None), kwargs.pop("sigma", None)
            mu, sigma = _ensure_same_length(mu, sigma)

            mu = _check_same(kwargs.pop("logmean", None), mu)
            mu = _check_same(kwargs.pop("logmedian", None), mu)
            mu = _check_same(kwargs.pop("logmode", None), mu)

            sigma = _check_same(kwargs.pop("logstd", None), sigma)
            if "logvar" in kwargs:
                _logvar2logstd = lambda x: x ** 0.5
                logvar = kwargs.pop("logvar")
                if isinstance(logvar, (list, tuple)):
                    sigma = _check_same(list(map(_logvar2logstd, logvar)), sigma)
                else:
                    sigma = _check_same(_logvar2logstd(logvar), sigma)
            if "logfwhm" in kwargs:
                _logfwhm2logstd = lambda x: x / math.sqrt(8 * math.log(2))
                logfwhm = kwargs.pop("logfwhm")
                if isinstance(logfwhm, (list, tuple)):
                    sigma = _check_same(list(map(_logfwhm2logstd, logfwhm)), sigma)
                else:
                    sigma = _check_same(_logfwhm2logstd(logfwhm), sigma)

            if mu is None:
                mu = 0
            if sigma is None:
                sigma = 1

            if kwargs:
                raise TypeError(f"got unexpected keyword arguments: {kwargs}")
            return cls(mu=mu, sigma=sigma)

        @property
        def min(self):
            return Sampler._map(lambda _: 0.0, self.mu)

        @property
        def max(self):
            return Sampler._map(lambda _: float('inf'), self.mu)

        @property
        def logmean(self):
            return self.mu

        logmedian = logmode = logmean

        @property
        def logstd(self):
            return self.sigma

        @property
        def logvar(self):
            if isinstance(self.sigma, list):
                return [s ** 2 for s in self.sigma]
            return self.sigma ** 2

        @property
        def logfwhm(self):
            if isinstance(self.sigma, list):
                return [s * math.sqrt(8 * math.log(2)) for s in self.sigma]
            return self.sigma * math.sqrt(8 * math.log(2))

        @property
        def mean(self):
            if torch.is_tensor(self.mu) or torch.is_tensor(self.sigma):
                return (self.mu + 0.5 * self.sigma ** 2).exp()
            _mean = lambda x: math.exp(x[0] + 0.5 * x[1] ** 2)
            if isinstance(self.mu, list):
                return list(map(_mean, zip(self.mu, self.sigma)))
            return _mean((self.mu, self.sigma))

        @property
        def var(self):
            if torch.is_tensor(self.mu) or torch.is_tensor(self.sigma):
                return self.mean() ** 2 * (self.sigma ** 2).exp() - 1
            _var = lambda x: math.exp(2 * x[0] + x[1] ** 2) * (math.exp(x[1] ** 2) - 1)
            if isinstance(self.mu, list):
                return list(map(_var, zip(self.mu, self.sigma)))
            return _var((self.mu, self.sigma))

        @property
        def mode(self):
            if torch.is_tensor(self.mu) or torch.is_tensor(self.sigma):
                return (self.mu - self.sigma ** 2).exp()
            _mode = lambda x: math.exp(x[0] - x[1] ** 2)
            if isinstance(self.mu, list):
                return list(map(_mode, zip(self.mu, self.sigma)))
            return _mode((self.mu, self.sigma))

        @property
        def fwhm(self):
            if torch.is_tensor(self.mu) or torch.is_tensor(self.sigma):
                return (self.mu + self.sigma * math.sqrt(8 * math.log(2))).exp() - (self.mu - self.sigma * math.sqrt(8 * math.log(2))).exp()
            _fwhm = lambda x: math.exp(x[0] + x[1] * math.sqrt(8 * math.log(2))) - math.exp(x[0] - x[1] * math.sqrt(8 * math.log(2)))
            if isinstance(self.mu, list):
                return list(map(_fwhm, zip(self.mu, self.sigma)))
            return _fwhm((self.mu, self.sigma))

        @property
        def median(self):
            if torch.is_tensor(self.mu):
                return self.mu.exp()
            if isinstance(self.mu, list):
                return list(map(math.exp, self.mu))
            return math.exp(self.mu)

    @overload
    def __init__(self, mu=0, sigma=1): ...

    @overload
    def __init__(self, *, logmean, logstd=1): ...

    @overload
    def __init__(self, *, logmean, logvar): ...

    @overload
    def __init__(self, *, logmean, logfwhm): ...

    @overload
    def __init__(self, *, mean, std=1): ...

    @overload
    def __init__(self, *, mean, var): ...

    @overload
    def __init__(self, *, mean, fwhm): ...

    def __init__(self, mu=None, sigma=None, **kwargs):
        super().__init__(**self.Parameters.make(mu=mu, sigma=sigma, **kwargs))

    def _use_torch(self, n):
        return (
            torch.is_tensor(self.mu) or
            torch.is_tensor(self.sigma) or
            isinstance(n, (list, tuple))
        )

    def __call__(self, n=None, **backend):
        if self._use_torch(n):
            n = tuple(ensure_list(n or []))
            x = torch.randn(n, **backend)
            x = exp_(add_(mul_(x, self.sigma), self.mu))
            return x
        return self._map(random.lognormvariate, self.mu, self.sigma, n=n)


class UniformSphere(Sampler):
    """
    Uniform distribution on the (d-1)-sphere
    """

    def __init__(self, ndim=3, **backend):
        """
        Parameters
        ----------
        ndim : int
            Number of dimensions of the embedding space.
            If `ndim=3`, generate sample that lie on the 2-sphere.
        """
        super().__init__()
        backend.setdefault('dtype', torch.get_default_dtype())
        backend.setdefault('device', 'cpu')
        self.ndim = ndim
        self.backend = backend

    def _use_torch(self, n):
        return isinstance(n, (list, tuple))

    def __call__(self, n=None, **backend):
        backend.setdefault('dtype', self.backend.dtype)
        backend.setdefault('device', self.backend.device)
        batch = n
        if batch is None:
            batch = ()
        if isinstance(batch, int):
            batch = (batch,)
        batch = tuple(batch)
        shape = batch + (self.ndim,)
        x = torch.randn(shape, **backend)
        mask = x.norm(2, dim=1) < 1e-3
        while mask.any():
            x = torch.where(mask, torch.randn(shape, **backend), x)
            mask = x.norm(2, dim=1) < 1e-3
        x /= x.norm(2, dim=1, keepdim=True)
        if n is None or isinstance(n, int):
            return x.tolist()
        return x


class MultivatiateNormal(Sampler):

    class Parameters(Sampler.Parameters):

        @property
        def mean(self):
            return self.mu

        @property
        def cov(self):
            return self.sigma

    def __init__(self, mu=None, sigma=None, **kwargs):
        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def _use_torch(self, n):
        return (
            torch.is_tensor(self.mu) or
            torch.is_tensor(self.sigma) or
            isinstance(n, (list, tuple))
        )

    def __call__(self, n=None, **backend):
        # prepare parameters
        mu = torch.as_tensor(self.mu, **backend)
        sigma = torch.as_tensor(self.sigma, **backend)
        k = _builtin_max(sigma.shape[-1], mu.shape[-1] if mu.shape else 1)
        mu = mu.expand(mu.shape[:-1] + (k,))
        sigma = sigma.expand(sigma.shape[:-2] + (k, k))

        # sample
        n = tuple(ensure_list(n or []))
        x = torch.distributions.MultivariateNormal(mu, sigma).sample(n)

        # return
        return x.tolist() if not self._use_torch(n) else x


class SquaredExponentialGP(Sampler):

    def __init__(self, mu=0, sigma=1, l=1, **kwargs):
        super().__init__(mu=mu, sigma=sigma, l=l, **kwargs)

    def __call__(self, n=None, **backend):
        backend.setdefault('dtype', torch.get_default_dtype())

        n = ensure_list(n or [])
        k = n[-1] if n else 1
        n = n[:-1]

        # compute covariance matrix
        l = torch.as_tensor(self.l, **backend)
        x = torch.arange(k, **backend)
        cov = square_(x[:, None] - x[None, :])
        cov = div_(cov, 2*(l*l))
        cov = exp_(cov)

        # scale by sigma
        sig = torch.as_tensor(self.sigma, **backend)[..., None, None]
        cov *= sig

        return MultivatiateNormal(mu=self.mu, sigma=cov)(n, **backend)


class AbsoluteExponentialGP(Sampler):

    def __init__(self, mu=0, sigma=1, l=1, **kwargs):
        super().__init__(mu=mu, sigma=sigma, l=l, **kwargs)

    def __call__(self, n=None, **backend):
        backend.setdefault('dtype', torch.get_default_dtype())

        n = ensure_list(n or [])
        k = n[-1] if n else 1
        n = n[:-1]

        # compute covariance matrix
        l = torch.as_tensor(self.l, **backend)
        x = torch.arange(k, **backend)
        cov = (x[:, None] - x[None, :]).abs()
        cov = mul_(div_(cov, l), -1)
        cov = exp_(cov)

        # scale by sigma
        sig = torch.as_tensor(self.sigma, **backend)[..., None, None]
        cov *= sig

        return MultivatiateNormal(mu=self.mu, sigma=cov)(n, **backend)


if False:  # WIP
    class MatternGP(Sampler):

        def __init__(self, mu=0, sigma=1, l=1, nu=0.5, **kwargs):
            super().__init__(
                **self.Parameters.make(mu=mu, sigma=sigma, l=l, nu=nu, **kwargs)
            )

        def __call__(self, n=None, **backend):
            n = ensure_list(n or [])
            k = n[-1] if n else 1
            n = n[:-1]

            # compute covariance matrix
            l = torch.as_tensor(self.l, **backend)
            nu = torch.as_tensor(self.nu, **backend)
            x = torch.arange(k, **backend)
            d = (x[:, None] - x[None, :]).abs()
            d = mul_(d, (2*nu)**0.5 / l)
            d = d.pow(nu) * torch.special.modified_bessel_k0
            cov = exp_(d)

            return MultivatiateNormal(mu=self.mu, sigma=cov)(n, **backend)


class TransformedSampler(Sampler):
    """A random variable that gets transformed by a deterministic function.

    1. Apply the sampler to get samples
    2. Apply the transform to each sample

    Attributes
    ----------
    sampler : Sampler
        A random sampler
    transform : Callable[[float], float]
        A value-wise transformation
    """

    Transform = Callable[[float], float]

    def __init__(self, sampler: Sampler, transform: Transform):
        super().__init__()
        self.sampler = sampler
        self.transform = transform

    def __call__(self, n=None, **backend):
        out = self.sampler(n, **backend)
        if torch.is_tensor(out) or isinstance(out, Number):
            return self.transform(out)
        else:
            return list(map(self.transform, out))


class CombinedSamplers(Sampler):
    """Random variables that get combined by a deterministic function.

    1. Apply each sampler to get samples
    2. Apply the transform to the samples

    Attributes
    ----------
    samplers : list[Sampler]
        A random sampler
    transform : Callable[[float, ...], float]
        A value-wise transformation

    """

    class Transform(Protocol):
        def __call__(self, *args: float) -> float: ...

    def __init__(self, samplers: Sequence[Sampler], transform: Transform):
        super().__init__()
        self.samplers = samplers
        self.transform = transform

    def __call__(self, n=None, **backend):
        out = [sampler(n, **backend) for sampler in self.samplers]
        if isinstance(out[0], list):
            return list(map(lambda x: self.transform(*x), zip(*out)))
        else:
            return self.transform(*out)


class MultipliedSampler(TransformedSampler):
    """Multiply each sample by a fixed value."""

    class _Op:
        def __init__(self, value):
            self.value = value

        def __call__(self, x):
            return x * self.value

    def __init__(self, sampler, other):
        super().__init__(sampler, self._Op(other))


class DividedSampler(TransformedSampler):
    """Divide each sample by a fixed value."""

    class _Op:
        def __init__(self, value):
            self.value = value

        def __call__(self, x):
            return x / self.value

    def __init__(self, sampler, other):
        super().__init__(sampler, self._Op(other))


class DividingSampler(TransformedSampler):
    """Divide a fixed value by each sample."""

    class _Op:
        def __init__(self, value):
            self.value = value

        def __call__(self, x):
            return self.value / x

    def __init__(self, sampler, other):
        super().__init__(sampler, self._Op(other))


class ShiftedSampler(TransformedSampler):
    """Add a fixed value to each sample."""

    class _Op:
        def __init__(self, value):
            self.value = value

        def __call__(self, x):
            return x + self.value

    def __init__(self, sampler, other):
        super().__init__(sampler, self._Op(other))


class ReverseShiftedSampler(TransformedSampler):
    """Subtract each sample from a fixed value."""

    class _Op:
        def __init__(self, value):
            self.value = value

        def __call__(self, x):
            return self.value - x

    def __init__(self, sampler, other):
        super().__init__(sampler, self._Op(other))


class PoweredSampler(TransformedSampler):
    """Raise each sample to a fixed power."""

    class _Op:
        def __init__(self, value):
            self.value = value

        def __call__(self, x):
            return x ** self.value

    def __init__(self, sampler, other):
        super().__init__(sampler, self._Op(other))


class ExponentSampler(TransformedSampler):
    """Raise a fixed value to the power of each sample."""

    class _Op:
        def __init__(self, value):
            self.value = value

        def __call__(self, x):
            return self.value ** x

    def __init__(self, sampler, other):
        super().__init__(sampler, self._Op(other))


class ProductOfSamplers(CombinedSamplers):
    """Multiply samples from two samplers."""

    class _Op:
        def __call__(self, *x):
            return functools.reduce(lambda a, b: a * b, x)

    def __init__(self, *samplers):
        super().__init__(samplers, self._Op())


class RatioOfSamplers(CombinedSamplers):
    """Divide samples from two samplers."""

    class _Op:
        def __call__(self, x, y):
            return x / y

    def __init__(self, left, right):
        super().__init__([left, right], self._Op())


class SumOfSamplers(CombinedSamplers):
    """Add samples from two samplers."""

    class _Op:
        def __call__(self, *x):
            return _builitin_sum(x)

    def __init__(self, *samplers):
        super().__init__(samplers, self._Op())


class DifferenceOfSamplers(CombinedSamplers):
    """Subtract samples from two samplers."""

    class _Op:
        def __call__(self, x, y):
            return x - y

    def __init__(self, left, right):
        super().__init__([left, right], self._Op())


class MinimumOfSamplers(CombinedSamplers):
    """Return the minimum of samples from two samplers."""

    class _Op:
        def __call__(self, *x):
            return _builtin_min(*x)

    def __init__(self, *samplers):
        super().__init__(samplers, self._Op())


class MaximumOfSamplers(CombinedSamplers):
    """Return the maximum of samples from two samplers."""

    class _Op:
        def __call__(self, *x):
            return _builtin_max(*x)

    def __init__(self, *samplers):
        super().__init__(samplers, self._Op())


class AverageOfSamplers(CombinedSamplers):
    """Return the average of samples from two samplers."""

    class _Op:
        def __call__(self, *x):
            return _builitin_sum(x) / len(x)

    def __init__(self, *samplers):
        super().__init__(samplers, self._Op())


class ExponentiatedSampler(TransformedSampler):
    """Return the exponentiated value of samples from a sampler."""

    class _Op:
        def __call__(self, x):
            return x.exp() if torch.is_tensor(x) else math.exp(x)

    def __init__(self, sampler):
        super().__init__(sampler, self._Op())


class LogarithmOfSampler(TransformedSampler):
    """Return the logarithm of samples from a sampler."""

    class _Op:
        def __call__(self, x):
            return x.log() if torch.is_tensor(x) else math.log(x)

    def __init__(self, sampler):
        super().__init__(sampler, self._Op())


class PowerOfSamplers(CombinedSamplers):
    """Raise samples from one sampler to the power of samples from another sampler."""

    class _Op:
        def __call__(self, x, y):
            return x ** y

    def __init__(self, base, exponent):
        super().__init__([base, exponent], self._Op())


# Aliases

min = MinimumOfSamplers
"""Alias for [`MinimumOfSamplers`][cornucopia.random.MinimumOfSamplers]"""

max = MaximumOfSamplers
"""Alias for [`MaximumOfSamplers`][cornucopia.random.MaximumOfSamplers]"""

sum = SumOfSamplers
"""Alias for [`SumOfSamplers`][cornucopia.random.SumOfSamplers]"""

exp = ExponentiatedSampler
"""Alias for [`ExponentiatedSampler`][cornucopia.random.ExponentiatedSampler]"""

log = LogarithmOfSampler
"""Alias for [`LogarithmOfSampler`][cornucopia.random.LogarithmOfSampler]"""

pow = PowerOfSamplers
"""Alias for [`PowerOfSamplers`][cornucopia.random.PowerOfSamplers]"""


@overload
def make_range(max, *, min=0, offset=0) -> Uniform: ...


@overload
def make_range(min, *, max, offset=0) -> Uniform: ...


@overload
def make_range(min, max, *, offset=0) -> Uniform: ...


def make_range(*args, **kwargs) -> Uniform:
    """
    If any of the inputs is a Sampler, return the Sampler.

    Else, build a {lower|upper} range.

    !!! examples
        === "Range"

            ```python
            make_range(x, y)           -> (x, y)
            make_range(x, y, offset=1) -> (1+x, 1+y)
            ```

        === "Symmetric range"

            ```python
            make_range(x)           -> (-x, x)
            make_range(x, offset=1) -> (1-x, 1+x)
            ```

        === "Upper bound"

            ```python
            make_range(0, x)     -> (0, x)
            make_range(x, min=0) -> (0, x)
            ```

        === "Lower bound"

            ```python
            make_range(x, 1)     -> (x, 1)
            make_range(x, max=1) -> (x, 1)
            ```
    """
    for x in args:
        if isinstance(x, Sampler):
            return x
        elif isinstance(x, (list, tuple)):
            return x
    for x in kwargs.values():
        if isinstance(x, Sampler):
            return x
        elif isinstance(x, (list, tuple)):
            return x
    assert len(args) <= 2
    vmid = kwargs.get('offset', 0)
    if len(args) == 2:
        return (vmid + args[0], vmid + args[1])
    vmin = kwargs.get('min', None)
    vmax = kwargs.get('max', None)
    if vmin is not None and vmax is not None:
        assert len(args) == 0
        return vmin + vmin, vmid + vmax
    elif vmax is not None:
        assert len(args) <= 1
        if args:
            return vmid + args[0], vmid + vmax
        else:
            return vmid - vmax, vmid + vmax
    elif vmin is not None:
        assert len(args) <= 1
        if args:
            return vmid + vmin, vmid + args[0]
        else:
            return vmid + vmin, vmid - vmin
    else:
        assert len(args) > 0
        return vmid - args[0], vmid + args[0]
