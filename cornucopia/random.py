__all__ = [
    'Sampler',
    'Fixed',
    'Uniform',
    'RandInt',
    'Normal',
    'LogNormal',
    'RandKFrom',
]

import random
import copy
import torch
from abc import ABC, abstractmethod
from numbers import Number
from .utils.py import ensure_list
from .utils.smart_inplace import add_, mul_, exp_


class Sampler(ABC):
    """
    Base class for random samplers, with a bunch of helpers.

    !!! note
        Samplers can be combined with determinstic values or between
        themselves using arithmetic operators:

        - `P + X -> ShiftedSampler(P, X)`
        - `P - X -> ShiftedSampler(P, -X)`
        - `P * X -> MultipliedSampler(P, X)`
        - `P / X -> DividedSampler(P, X)`
        - `X / P -> DividedSampler(X, P)`
        - `P + Q -> SumOfSamplers(P, Q)`
        - `P - Q -> DifferenceOfSamplers(P, Q)`
        - `P * Q -> ProductOfSamplers(P, Q)`
        - `P / Q -> RatioOfSamplers(P, Q)`

    Attributes
    ----------
    theta : dict
        Parameters of the sampler
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


class Fixed(Sampler):
    """Fixed value.

    ```python
    Fixed(value)
    ```

    Attributes
    ----------
    value : number or sequence[number]
    """

    def __init__(self, value):
        super().__init__(value=value)

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
        return self.map(copy.deepcopy, self.value, n=n)


class Uniform(Sampler):
    """Continuous uniform sampler.

    ```python
    Uniform()
    Uniform(max)
    Uniform(min, max)
    ```

    Attributes
    ----------
    min : float or sequence[float], default=0
        Lower bound (inclusive)
    max : float or sequence[float], default=1
        Upper bound (inclusive or exclusive, depending on rounding)
    """

    def __init__(self, *args, **kwargs):
        min, max = 0, 1
        if len(args) == 2:
            min, max = args
        elif len(args) == 1:
            max = args[0]
        if 'min' in kwargs:
            min = kwargs['min']
        if 'max' in kwargs:
            max = kwargs['max']
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
            x = torch.rand(n, **backend)
            x = add_(mul_(x, self.max - self.min), self.min)
            return x
        return self.map(random.uniform, self.min, self.max, n=n)


class RandInt(Sampler):
    """Discrete uniform sampler

    ```python
    RandInt(max)
    RandInt(min, max)
    ```

    Attributes
    ----------
    min : int or sequence[int], default=0
        Lower bound (inclusive)
    max : int or sequence[int]
        Upper bound (inclusive)
    """

    def __init__(self, *args, **kwargs):
        min, max = 0, None
        if len(args) == 2:
            min, max = args
        elif len(args) == 1:
            max = args[0]
        if 'min' in kwargs:
            min = kwargs['min']
        if 'max' in kwargs:
            max = kwargs['max']
        if max is None:
            raise ValueError('Expected at least one argument')
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
            backend.setdefault('device', getattr(self.max - self.min, 'device', None))
            return torch.randint(low=self.min, high=1 + self.max, size=n, **backend)
        return self.map(random.randint, self.min, self.max, n=n)


class RandKFrom(Sampler):
    """Discrete uniform sampler

    ```python
    RandKFrom(range, k=None, replacement=False)
    ```

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

    ```python
    Normal()
    Normal(mu)
    Normal(mu, sigma)
    ```

    Attributes
    ----------
    mu : float or sequence[float], default=0
        Mean
    sigma : float or sequence[float], default=1
        Standard deviation
    """

    def __init__(self, mu=0, sigma=1):
        super().__init__(mu=mu, sigma=sigma)

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
        return self.map(random.normalvariate, self.mu, self.sigma, n=n)


class LogNormal(Sampler):
    """LogNormal sampler

    ```python
    LogNormal()
    LogNormal(mu)
    LogNormal(mu, sigma)
    ```

    Attributes
    ----------
    mu : float or sequence[float], default=0
        Mean of the log
    sigma : float or sequence[float], default=1
        Standard deviation of the log
    """

    def __init__(self, mu=0, sigma=1):
        super().__init__(mu=mu, sigma=sigma)

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
        return self.map(random.lognormvariate, self.mu, self.sigma, n=n)


class TransformedSampler(Sampler):
    """A random variable that gets transformed by a deterministic function.

    1. Apply the sampler to get samples
    2. Apply the transform to each sample

    ```python
    TransformedSampler(
        sampler: Sample,
        transform: Callable[[float], float]
    )
    ```

    Attributes
    ----------
    sampler : Sampler
        A random sampler
    transform : callable(float) -> float
        A value-wise transformation
    """

    def __init__(self, sampler, transform):
        super().__init__(sampler=sampler, transform=transform)

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

    ```python
    CombinedSamplers(
        samplers: list[Sampler],
        transform: Callable[[float, ...], float]
    )
    ```

    Attributes
    ----------
    samplers : list[Sampler]
        A random sampler
    transform : callable(*float) -> float
        A value-wise transformation

    """

    def __init__(self, samplers, transform):
        super().__init__(samplers=samplers, transform=transform)

    def __call__(self, n=None, **backend):
        out = [sampler(n, **backend) for sampler in self.samplers]
        if isinstance(out[0], list):
            return list(map(lambda x: self.transform(*x), out))
        else:
            self.transform(*out)


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


class ProductOfSamplers(CombinedSamplers):
    """Multiply samples from two samplers."""

    class _Op:
        def __call__(self, x, y):
            return x * y

    def __init__(self, left, right):
        super().__init__([left, right], self._Op())


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
        def __call__(self, x, y):
            return x + y

    def __init__(self, left, right):
        super().__init__([left, right], self._Op())


class DifferenceOfSamplers(CombinedSamplers):
    """Subtract samples from two samplers."""

    class _Op:
        def __call__(self, x, y):
            return x - y

    def __init__(self, left, right):
        super().__init__([left, right], self._Op())


def make_range(*args, **kwargs):
    """
    ```python
    make_range([min], max, *, offset=0)
    ```

    If any of the inputs is a Sampler, return the Sampler.

    Else, build a (lower, upper) range.

    !!! examples
        ```python
        # full range
        make_range(x, y) -> (x, y)
        make_range(x, y, offset=1) -> (1+x, 1+y)
        # symmetric range
        make_range(x) -> (-x, x)
        make_range(x, offset=1) -> (1-x, 1+x)
        # upper bound
        make_range(0, x) -> (0, x)
        make_range(x, min=0) -> (0, x)
        # lower bound
        make_range(x, 1) -> (x, 1)
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
