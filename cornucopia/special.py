__all__ = [
    'IdentityTransform',
    'SequentialTransform',
    'PerChannelTransform',
    'MaybeTransform',
    'SwitchTransform',
    'IncludeKeysTransform',
    'ExcludeKeysTransform',
    'SharedTransform',
    'ReturningTransform',
    'MappedTransform',
    'RandomizedTransform',
    'BatchedTransform',
    'SplitChannels',
    'CatChannels',
]
import torch
from torch import nn
from .base import (
    Transform,
    IdentityTransform,
    SequentialTransform,
    PerChannelTransform,
    MaybeTransform,
    SwitchTransform,
    IncludeKeysTransform,
    ExcludeKeysTransform,
    SharedTransform,
    ReturningTransform,
    MappedTransform,
    RandomizedTransform,
)
from .baseutils import Args, Kwargs, ArgsAndKwargs
from .baseutils import recursive_cat


class BatchedTransform(nn.Module):
    """Apply a transform to a batch

    !!! example
        Functional call:
        ```python
        batched_transform = cc.ctx.batch(transform)
        img, lab = batched_transform(img, lab)  # input shapes: [B, C, X, Y, Z]
        ```
        Object call:
        ```python
        batched_transform = cc.BatchedTransform(transform)
        img, lab = batched_transform(img, lab)  # input shapes: [B, C, X, Y, Z]
        ```
    """

    def __init__(self, transform):
        """
        Parameters
        ----------
        transform : Transform
            Transform to apply to a batched tensor or to a nested
            structure of batched tensors
        """
        super().__init__()
        self.transform = transform

    def forward(self, *args, **kwargs):

        class UnpackError(ValueError):
            pass

        def _unpack(x, i):
            if isinstance(x, (list, tuple)):
                return type(x)(_unpack(x1, i) for x1 in x)
            elif hasattr(x, 'items'):
                return {key: _unpack(val, i) for key, val in x.items()}
            elif torch.is_tensor(x):
                if len(x) <= i:
                    raise UnpackError(f'Trying to unpack index {i} of '
                                      f'tensor with shape {list(x.shape)}')
                return x[i]
            else:
                # let's assume it is a nontensor (None, number, etc)
                return x

        def unpack(x):
            i = 0
            while True:
                try:
                    yield _unpack(x, i)
                    i += 1
                except UnpackError:
                    return

        def pack(x):
            x0 = x[0]
            if isinstance(x0, (list, tuple)):
                return type(x0)(pack([x1[i] for x1 in x])
                                for i in range(len(x0)))
            elif hasattr(x0, 'items'):
                return {key: pack([x1[key] for x1 in x]) for key in x0.keys()}
            elif torch.is_tensor(x0):
                return torch.stack(x)
            else:
                raise TypeError(f'Don\'t know what to do with type {type(x0)}')

        batch = []
        for elem, kelem in zip(unpack(args), unpack(kwargs)):
            batch.append(self.transform(*elem, **kelem))
        batch = pack(batch)
        return batch

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


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
        args = tuple(recursive_cat(x) for x in args)
        valid_keys = kwargs.keys()
        if self._include is not None:
            valid_keys = [k for k in valid_keys if k in self._include]
        if self._exclude:
            valid_keys = [k for k in valid_keys if k not in self._exclude]
        kwargs = {k: recursive_cat(v) if k in valid_keys else v
                  for k, v in kwargs.items()}
        if args and kwargs:
            return ArgsAndKwargs(args, kwargs)
        elif kwargs:
            return Kwargs(kwargs)
        elif len(args) > 1:
            return Args(args)
        else:
            return args[0]
