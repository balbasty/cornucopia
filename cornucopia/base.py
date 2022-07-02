import torch
from torch import nn
import random


__all__ = ['Transform', 'SequentialTransform', 'RandomizedTransform',
           'MaybeTransform', 'MappedTransform']


class Transform(nn.Module):
    """
    Base class for transforms.

    Transforms are parameter-free modules that usually act on tensors
    without a batch dimension (e.g., [C, *spatial])
    """

    def __init__(self, shared=False):
        """

        Parameters
        ----------
        shared : None or bool or int
            If False: independent across images and channels
            If True: shared across images and channels
        """
        super().__init__()
        if shared == 'channel': shared = 0
        self.shared = shared

    def get_first_element(self, x):
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

    def forward_with_parameters(self, *x, parameters=None):
        # DEV: this function should in general not be overloaded
        if len(x) > 1:
            return tuple(self.forward_with_parameters(elem, parameters=parameters)
                         for elem in x)
        x = x[0]
        if hasattr(x, 'items'):
            return {key: self.forward_with_parameters(value, parameters=parameters)
                    for key, value in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self.forward_with_parameters(elem, parameters=parameters)
                           for elem in x)
        return self.transform_with_parameters(x, parameters=parameters)

    def recursive_cat(self, x, **kwargs):
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

    def forward(self, *x):
        # DEV: this function should in general not be overloaded
        if self.shared is True:
            x0 = self.get_first_element(x)
            if torch.is_tensor(x0):
                x0 = x0[0, None]
            theta = self.get_parameters(x0)
            numel = len(x)
            x = self.forward_with_parameters(*x, parameters=theta)
            return tuple(x) if numel > 1 else x

        # not shared across images -> unfold
        if len(x) > 1:
            return tuple(self(elem) for elem in x)
        x = x[0]
        if hasattr(x, 'items'):
            return {key: self(value) for key, value in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self(elem) for elem in x)

        # now we're working with a single tensor (or str)
        if self.shared is False:
            channels = x.unbind(0)
            channels = [self.transform(c[None]) for c in channels]
            return self.recursive_cat(channels, dim=0)
        else:
            return self.transform(x)

    def transform(self, x):
        # DEV: this function can be overloaded if `shared` is not supported
        theta = self.get_parameters(x)
        return self.transform_with_parameters(x, parameters=theta)

    def transform_and_parameters(self, x):
        # DEV: This function should probably not be overloaded
        theta = self.get_parameters(x)
        return self.transform_with_parameters(x, parameters=theta), theta

    def get_parameters(self, x):
        # DEV: This function should be overloaded if `shared` is supported
        return None

    def transform_with_parameters(self, x, parameters):
        raise NotImplementedError("This function should be implemented "
                                  "in Transforms that handle `shared`.")

    def __add__(self, other):
        if isinstance(other, SequentialTransform):
            return other.__radd__(self)
        return SequentialTransform([self, other])

    def __radd__(self, other):
        if isinstance(other, SequentialTransform):
            return other.__add__(self)
        return SequentialTransform([other, self])


class SequentialTransform(Transform):
    """A sequence of transforms"""

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, *x):
        for transform in self.transforms:
            numel = len(x)
            x = transform(*x)
            x = (x,) if numel == 1 else x
        return x[0] if len(x) == 1 else tuple(x)

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
        return SequentialTransform([*self.transforms, other])

    def __radd__(self, other):
        if isinstance(other, SequentialTransform):
            return SequentialTransform([*other.transforms, *self.transforms])
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
        super().__init__(shared=shared)
        self.subtransform = transform
        self.prob = prob

    def transform_with_parameters(self, x, parameters):
        if parameters > 1 - self.prob:
            return self.subtransform(x)
        else:
            return x

    def get_parameters(self, x):
        return random.random()


class RandomizedTransform(Transform):
    """
    Transform generated by randomizing some parameters of another transform.
    """

    def __init__(self, transform, sample, shared=False):
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
        self.subtransform = transform

    def get_parameters(self, x):
        if isinstance(self.sample, (list, tuple)):
            return self.subtransform(*[f() for f in self.sample])
        if hasattr(self.sample, 'items'):
            return self.subtransform(**{k: f() for k, f in self.sample.items()})
        return self.subtransform(self.sample())

    def transform_with_parameters(self, x, parameters):
        return parameters(x)

    def __repr__(self):
        if type(self) is RandomizedTransform:
            try:
                if issubclass(self.subtransform, Transform):
                    return f'Randomized{self.subtransform.__name__}()'
            except TypeError:
                pass
        return super().__repr__()


class MappedTransform(Transform):
    """
    Transforms that are applied to specific keys if an input dictionary
    """

    def __init__(self, *maybe_map, **map):
        """

        Parameters
        ----------
        map : dict[key -> Transform]
        """
        super().__init__()
        if maybe_map:
            map.update(maybe_map)
        self.map = map

    def forward(self, *x):
        if len(x) > 1:
            return tuple(self(elem) for elem in x)
        x = x[0]
        if hasattr(x, 'items'):
            return {key: self.map[key](value) if key in self.map else
                    super(type(self), self).forward(value)
                    for key, value in x.items()}
        return super().forward(x)

    def transform_with_parameters(self, x, parameters):
        return x

    def __repr__(self):
        mapping = {key: value for key, value in self.map.items()}
        return f'{type(self).__name__}({mapping})'
