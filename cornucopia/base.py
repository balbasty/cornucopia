__all__ = ['Transform', 'SequentialTransform', 'RandomizedTransform',
           'MaybeTransform', 'MappedTransform', 'randomize', 'map']

import torch
from torch import nn
import random
from .utils.py import ensure_list, cumsum


def _get_first_element(x):
    """Return the fist element (tensor or string) in the nested structure"""
    def _recursive(x):
        if hasattr(x, 'items'):
            for k, v in x.items():
                v, ok = _recursive(v)
                if ok: return v, True
            return None, False
        if isinstance(x, (list, tuple)):
            for v in x:
                v, ok = _recursive(v)
                if ok: return v, True
            return None, False
        return x, True
    return _recursive(x)[0]


def _recursive_cat(x, **kwargs):
    """Concatenate tensors across the channel axis in a nested structure"""
    def _rec(*x):
        if all(torch.is_tensor(x1) for x1 in x):
            return torch.cat(x, **kwargs)
        if isinstance(x[0], (list, tuple)):
            return type(x[0])(_rec(*x1) for x1 in zip(*x))
        if hasattr(x[0], 'items'):
            return {k: _rec(*x2)
                    for k in x[0].keys()
                    for x2 in zip(x1[k] for x1 in x)}
        raise TypeError(f'What should I do with a {type(x[0])}?')
    return _rec(*x)


class Arguments:
    """Base class for returned arguments"""
    pass


class Args(tuple, Arguments):
    """Tuple-like"""
    pass


class Kwargs(dict, Arguments):
    """Dict-like, except that unzipping works on values instead of keys"""

    def __iter__(self):
        for v in self.values():
            yield v


class ArgsAndKwargs(Arguments):
    def __init__(self, args, kwargs):
        self.args = Args(args)
        self.kwargs = Kwargs(kwargs)

    def __iter__(self):
        for a in self.args:
            yield a
        for v in self.kwargs:
            yield v


class Transform(nn.Module):
    """
    Base class for transforms.

    Transforms are parameter-free modules that usually act on tensors
    without a batch dimension (e.g., [C, *spatial])

    In general, transforms take an argument `shared` that can take values
    `True`, `False`, `'tensors'` or `'channels'`. If a transform is
    shared across tensors and/or channels, the parameters of the
    transform are computed (or sampled) once and applied to all
    tensors and/or channels.

    Note that in nested transforms (i.e., a `MaybeTransform` or
    `RandomizedTransform`), the value of `shared` may be different in
    the parent anf child transform. For example, we may want to randomly
    decide to apply (or not) a bias field at the parent level but, when
    applied, let the bias field be different in each channel. Such a
    transform would be defined as::

        t = MaybeTransform(MultFieldTransform(shared=False), shared=True)

    Furthermore, the addition of two transforms implictly defines
    (or extends) a `SequentialTransform`::

        t1 = MultFieldTransform()
        t2 = GaussianNoiseTransform()
        seq = t1 + t2

    """

    def __init__(self, shared=False):
        """

        Parameters
        ----------
        shared : bool or 'channels' or 'tensors'
            If True: shared across tensors and channels
            If 'channels': shared across channels
            If 'tensors': shared across tensors
            If False: independent across tensors and channels
        """
        super().__init__()
        self.shared = shared

    def forward(self, *a, **k):
        """Apply the transform recursively.

        Parameters
        ----------
        x : [nested list or dict of] (C, *shape) tensor
            Input tensors

        Returns
        -------
        [nested list or dict of] (C, *shape) tensor
            Output tensors

        """
        # DEV: this function should in general not be overloaded

        if k:
            return self.forward(ArgsAndKwargs(a, k) if a else Kwargs(k))
        elif len(a) > 1:
            return self.forward(Args(a))
        elif not a:
            return None

        x = a[0]
        intype = type(x)
        outtype = lambda x: (ArgsAndKwargs(*x) if intype is ArgsAndKwargs else
                             intype(x) if issubclass(intype, (Args, Kwargs))
                             else x)
        if isinstance(x, ArgsAndKwargs):
            x = [tuple(x.args), dict(x.kwargs)]
        elif isinstance(x, Args):
            x = tuple(x)
        elif isinstance(x, Kwargs):
            x = dict(x)

        if self.shared is True or (self.shared and self.shared[0] == 't'):
            x0 = _get_first_element(x)
            if torch.is_tensor(x0) and self.shared is True:
                x0 = x0[0, None]
            theta = self.get_parameters(x0)
            x = self.forward_with_parameters(x, parameters=theta)
            return outtype(x)

        # not shared across images -> unfold
        if isinstance(x, (list, tuple)):
            return type(x)(self(elem) for elem in x)
        if hasattr(x, 'items'):
            return {key: self(value) for key, value in x.items()}

        # now we're working with a single tensor (or str)
        if self.shared is False:
            return outtype(self.transform_tensor_perchannel(x))
        else:
            return outtype(self.transform_tensor(x))

    def forward_with_parameters(self, x, parameters=None):
        """Apply the transform with pre-computed parameters

        Parameters
        ----------
        x : [nested list or dict of] (C, *shape) tensor
            Input tensors
        parameters : any
            Pre-computed parameters of the transform

        Returns
        -------
        [nested list or dict of] (C, *shape) tensor
            Output tensors

        """
        # DEV: this function should in general not be overloaded
        if hasattr(x, 'items'):
            return {key: self.forward_with_parameters(value, parameters=parameters)
                    for key, value in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self.forward_with_parameters(elem, parameters=parameters)
                           for elem in x)
        return self.apply_transform(x, parameters=parameters)

    def transform_tensor(self, x):
        """Apply the transform to a single tensor

        Parameters
        ----------
        x : (C, *shape) tensor
            A single input tensor

        Returns
        -------
        x : (C, *shape) tensor
            A single output tensor

        """
        # DEV: this function can be overloaded if `shared` is not supported
        theta = self.get_parameters(x)
        return self.apply_transform(x, parameters=theta)

    def transform_tensor_perchannel(self, x):
        """Apply the transform to each channel of a single tensor

        Parameters
        ----------
        x : (C, *shape) tensor
            A single input tensor

        Returns
        -------
        x : (C, *shape) tensor
            A single output tensor

        """
        # DEV: This function should usually not be overloaded
        channels = x.unbind(0)
        channels = [self.transform_tensor(c[None]) for c in channels]
        return _recursive_cat(channels, dim=0)

    def get_parameters(self, x):
        """Compute the parameters of a transform from an input tensor

        Parameters
        ----------
        x : (C, *shape) tensor
            A single input tensor

        Returns
        -------
        parameters : any
            Computed parameters

        """
        # DEV: This function should be overloaded if `shared` is supported
        return None

    def apply_transform(self, x, parameters):
        """Apply the transform to a single tensor using precomputed parameters

        Parameters
        ----------
        x : (C, *shape) tensor
            A single input tensor
        parameters : any
            Precomputed parameters

        Returns
        -------
        x : (C, *shape) tensor
            A single output tensor

        """
        raise NotImplementedError("This function should be implemented "
                                  "in Transforms that handle `shared`.")

    def transform_tensor_and_get_parameters(self, x):
        """Apply the transform to a single tensor and return its parameters

        Parameters
        ----------
        x : (C, *shape) tensor
            A single input tensor

        Returns
        -------
        x : (C, *shape) tensor
            A single output tensor
        parameters : any
            Computed parameters

        """

        # DEV: This function should probably not be overloaded
        theta = self.get_parameters(x)
        return self.apply_transform(x, parameters=theta), theta

    def __add__(self, other):
        if isinstance(other, SequentialTransform):
            return other.__radd__(self)
        else:
            return SequentialTransform([self, other])

    def __radd__(self, other):
        if isinstance(other, SequentialTransform):
            return other.__add__(self)
        else:
            return SequentialTransform([other, self])


class SequentialTransform(Transform):
    """A sequence of transforms

    Sequences can be built explicitly, or simply by adding transforms
    together::

        t1 = MultFieldTransform()
        t2 = GaussianNoiseTransform()
        seq = SequentialTransform([t1, t2])     # explicit
        seq = t1 + t2                           # implicit

    Sequences can also be extended by addition::

        seq += SmoothTransform()


    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, *args, **kwargs):
        if kwargs and args:
            x = ArgsAndKwargs(args, kwargs)
        elif kwargs:
            x = Kwargs(kwargs)
        elif len(args) > 1:
            x = Args(args)
        elif args:
            x = args[0]
        else:
            return None
        for transform in self.transforms:
            x = transform(x)
        return x

    def __len__(self):
        return len(self.transforms)

    def __iter__(self):
        for t in self.transforms:
            yield t

    def __getitem__(self, item):
        if isinstance(item, slice):
            return SequentialTransform(self.transforms[item])
        else:
            return self.transforms[item]

    def __add__(self, other):
        if isinstance(other, SequentialTransform):
            return SequentialTransform([*self.transforms, *other.transforms])
        else:
            return SequentialTransform([*self.transforms, other])

    def __radd__(self, other):
        if isinstance(other, SequentialTransform):
            return SequentialTransform([*other.transforms, *self.transforms])
        else:
            return SequentialTransform([other, *self.transforms])

    def __iadd__(self, other):
        if isinstance(other, SequentialTransform):
            self.transforms += other.transforms
        else:
            self.transforms.append(other)

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.transforms)})'


class MaybeTransform(Transform):
    """Randomly apply a transform"""

    def __init__(self, transform, prob=0.5, shared=False):
        """

        Parameters
        ----------
        transform : Transform
            A transform to randomly apply
        prob : float
            Probability to apply the transform
        shared : bool
            Roll the dice once for all input tensors
        """
        super().__init__(shared=shared)
        self.subtransform = transform
        self.prob = prob

    def apply_transform(self, x, parameters):
        if parameters > 1 - self.prob:
            return self.subtransform(x)
        else:
            return x

    def get_parameters(self, x):
        return random.random()


class SwitchTransform(Transform):
    """Randomly one of multiple transforms"""

    def __init__(self, transforms, prob=0, shared=False):
        """

        Parameters
        ----------
        transforms : list[Transform]
            A list of transforms to sample from
        prob : list[float]
            Probability of applying each transform
        shared : bool
            Roll the dice once for all input tensors
        """
        super().__init__(shared=shared)
        self.transforms = transforms
        if not prob:
            prob = [1/len(transforms)] * len(transforms)
        prob = ensure_list(prob, len(transforms), default=0)
        prob = [x / sum(prob) for x in prob]
        self.prob = prob

    def apply_transform(self, x, parameters):
        prob = cumsum(self.prob)
        for k, t in enumerate(self.transforms):
            if parameters > 1 - prob[k]:
                return t(x)
            else:
                return x

    def get_parameters(self, x):
        return random.random()


class RandomizedTransform(Transform):
    """
    Transform generated by randomizing some parameters of another transform.
    """

    def __init__(self, transform, sample, ksample=None, shared=False):
        """

        Parameters
        ----------
        transform : callable(...) -> Transform
            A Transform subclass or a function that constructs a Transform.
        sample : [list or dict of] callable
            A collection of functions that generate parameter values provided
            to `transform`.
        """
        super().__init__(shared=shared)
        self.sample = sample
        self.ksample = ksample
        self.subtransform = transform

    def get_parameters(self, x):
        args = []
        kwargs = {}
        if isinstance(self.sample, (list, tuple)):
            args += [f() if callable(f) else f for f in self.sample]
        elif hasattr(self.sample, 'items'):
            kwargs.update({k: f() if callable(f) else f
                           for k, f in self.sample.items()})
        else:
            args += [self.sample() if callable(self.sample) else self.sample]
        if self.ksample:
            kwargs.update({k: f() if callable(f) else f
                           for k, f in self.ksample.items()})
        return self.subtransform(*args, **kwargs)

    def apply_transform(self, x, parameters):
        return parameters(x)

    def __repr__(self):
        if type(self) is RandomizedTransform:
            try:
                if issubclass(self.subtransform, Transform):
                    return f'Randomized{self.subtransform.__name__}()'
            except TypeError:
                pass
        return super().__repr__()


def randomize(klass, shared=False):
    """Decorator to convert a Transform into a RandomizedTransform

    Parameters
    ----------
    klass : subclass(Transform)
        Transform type to randomize
    shared : bool, default=False
        Shared random parameters across channels

    Returns
    -------
    init : callable
        Function that initializes a RandomizedTransform
    """
    def init(*args, **kwargs):
        return RandomizedTransform(klass, args, kwargs, shared=shared)
    return init


class MappedTransform(Transform):
    """
    Transforms that are applied to specific positional or arguments

    Examples
    --------
    ::
        img = torch.randn([1, 32, 32])
        seg = torch.randn([3, 32, 32]).softmax(0)

        # positional variant
        trf = MappedTransform(GaussianNoise(), None)
        img, seg = trf(img, seg)

        # keyword variant
        trf = MappedTransform(image=GaussianNoise())
        img, seg = trf(image=img, label=seg)


        # alternative version
        dat = {'img': torch.randn([1, 32, 32]),
               'seg': torch.randn([3, 32, 32]).softmax(0)}
        dat = MappedTransform(img=GaussianNoise(), nested=True)(dat)


    """

    def __init__(self, *mapargs, nested=False, **mapkwargs):
        """

        Parameters
        ----------
        mapargs : tuple[Transform]
            Transform to apply to positional arguments
        mapkwargs : dict[key -> Transform]
            Transform to apply to keyword arguments
        nested : bool, default=False
            Recursively traverse the inputs until we find matching dictionaries.
            Only mapkwargs are accepted if "nested"
        """
        super().__init__(shared='channels')
        self.mapargs = mapargs
        self.mapkwargs = mapkwargs
        self.nested = nested
        if nested and mapargs:
            raise ValueError('Cannot have both `nested` and positional transforms')

    def forward(self, *args, **kwargs):
        if args:
            a0 = args[0]
            if isinstance(a0, Args):
                return self.forward(*a0)
            if isinstance(a0, Kwargs):
                return self.forward(**a0)
            if isinstance(a0, ArgsAndKwargs):
                return self.forward(*a0.args, **a0.kwargs)
        default = self.forward if self.nested else (lambda x: x)
        if args:
            mapargs = self.mapargs
            mapargs += (default,) * max(0, len(args) - len(mapargs))
            args = tuple(f(a) if f else a for f, a in zip(mapargs, args))
        if kwargs:
            kwargs = {key: (self.mapkwargs.get(key, None) or default)(value)
                      for key, value in kwargs.items()}
        if args and kwargs:
            return ArgsAndKwargs(args, kwargs)
        elif kwargs:
            return Kwargs(kwargs)
        elif len(args) > 1:
            return Args(args)
        else:
            return args[0]

    def apply_transform(self, x, parameters):
        return x

    def __repr__(self):
        s = []
        for v in self.mapargs:
            s += [f'{v}']
        for k, v in self.mapkwargs.items():
            s += [f'{k}={v}']
        s = ', '.join(s)
        return f'{type(self).__name__}({s})'


def map(*mapargs, nested=False, **mapkwargs):
    return MappedTransform(*mapargs, nested=False, **mapkwargs)

class SplitChannels(Transform):
    """Unbind tensors across first dimension (without collapsing it)"""
    def __init__(self):
        super().__init__(shared='channels')

    def transform_tensor(self, x):
        return x.chunk(len(x))


class CatChannels(Transform):
    """
    Concatenate tensors across first dimension
    Assumes that the nested-most level in the structure is the one to
    concatenate.
    """
    def __init__(self):
        super().__init__(shared='channels')

    def forward(self, *args, **kwargs):
        args = tuple(_recursive_cat(x) for x in args)
        kwargs = {k: _recursive_cat(v) for k, v in kwargs.items()}
        if args and kwargs:
            return ArgsAndKwargs(args, kwargs)
        elif kwargs:
            return Kwargs(kwargs)
        elif len(args) > 1:
            return Args(args)
        else:
            return args[0]

