__all__ = [
    'Transform',
    'FinalTransform',
    'NonFinalTransform',
    'SpecialMixin',
]
import torch
from torch import nn
import random
from .random import Sampler
from .utils.py import ensure_list, cumsum
from .baseutils import (
    Args, Kwargs, ArgsAndKwargs, Returned, VirtualTensor,
    get_first_element, prepare_output, unset, recursive_cat,
)


class Transform(nn.Module):
    """
    Base class for all transforms
    """

    def __init__(self, *,
                 returns=None,
                 append=False,
                 prefix=True,
                 include=None,
                 exclude=None,
                 ):
        """
        Parameters
        ----------
        returns : [list or dict of] str
            Which tensors to return.
            Most transforms accept `'input'` and `'output'` as valid
            returns.
        append : bool
            Append the (structure of) returned tensors to the parent
            structure.
        prefix : bool
            If `append` and parent is a dict, prefix the child's key with
            the parent key.
        include : str or list[str]
            List of keys to which the transform should apply
        exclude : str or list[str]
            List of keys to which the transform should not apply
        """
        super().__init__()
        self.returns = returns
        self.append = append
        self.prefix = prefix
        self.include = ensure_list(include) if include is not None else None
        self.exclude = ensure_list(exclude or tuple())

    @property
    def is_final(self):
        return False

    def get_prm(self):
        return dict(
            returns=self.returns,
            append=self.append,
            prefix=self.prefix,
            include=self.include,
            exclude=self.exclude,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __add__(self, other):
        return SequentialTransform([self, other])

    def __radd__(self, other):
        return SequentialTransform([other, self])

    def __iadd__(self, other):
        return SequentialTransform([self, other])

    def __mul__(self, prob):
        return MaybeTransform(self, prob)

    def __rmul__(self, prob):
        return MaybeTransform(self, prob)

    def __imul__(self, prob):
        return MaybeTransform(self, prob)

    def __or__(self, other):
        return SwitchTransform([self, other])

    def __ior__(self, other):
        return SwitchTransform([self, other])

    def __call__(self, *a, **k):
        out = super().__call__(*a, **k)
        if isinstance(out, Returned):
            out = out.obj
        return out

    def forward(self, *a, **k):
        """Apply the transform recursively.

        Parameters
        ----------
        x : [nested list or dict of] tensor
            Input tensors, with shape `(C, *shape)`

        Returns
        -------
        [nested list or dict of] tensor
            Output tensors. with shape `(C, *shape)`

        """
        if k:
            return self.forward(ArgsAndKwargs(a, k) if a else Kwargs(k))
        elif len(a) > 1:
            return self.forward(Args(a))
        elif not a:
            return None

        x = a[0]
        intype = type(x)

        def outtype(x):
            return (ArgsAndKwargs(*x) if intype is ArgsAndKwargs
                    else intype(x) if issubclass(intype, (Args, Kwargs))
                    else x)

        if isinstance(x, ArgsAndKwargs):
            x = [tuple(x.args), dict(x.kwargs)]
        elif isinstance(x, Args):
            x = tuple(x)
        elif isinstance(x, Kwargs):
            x = dict(x)

        # not shared across images -> unfold
        if isinstance(x, (list, tuple)):
            return outtype(self._forward_list(x, self.forward))
        if hasattr(x, 'items'):
            x = self._forward_dict(x, self.forward)
            if not isinstance(x, dict) and outtype in (Kwargs, ArgsAndKwargs):
                return Args(x)
            else:
                return x

        # now we're working with a single tensor (or str)
        y = self.apply(x)
        if not isinstance(y, Returned):
            if not isinstance(y, type(self.returns)):
                y = dict(input=x, output=y)
                y = prepare_output(y, self.returns).obj
            elif isinstance(self.returns, dict):
                for key, target in self.returns.items():
                    if target == 'input':
                        y[key] = x
            elif isinstance(self.returns, (list, tuple)):
                for i, target in enumerate(self.returns):
                    if target == 'input':
                        y[i] = x
            y = Returned(y)
        return y

    def _get_valid_keys(self, x):
        valid_keys = x.keys()
        if self.include is not None:
            valid_keys = [k for k in valid_keys if k in self.include]
        if self.exclude:
            valid_keys = [k for k in valid_keys if k not in self.exclude]
        return valid_keys

    def _forward_list(self, x: list, forward: callable = None):
        """Apply forward pass to elements of a list"""
        forward = forward or self.forward
        y = []
        for elem in x:
            elem = forward(elem)
            if isinstance(elem, Returned):
                elem = elem.obj
                if self.append:
                    if isinstance(elem, dict):
                        y.extend(elem.values())
                        continue
                    elif isinstance(elem, (list, tuple)):
                        y.extend(elem)
                        continue
            y.append(elem)
        return type(x)(y)

    def _forward_dict(self, x: dict, forward: callable = None):
        """Apply forward pass to elements of a dict"""
        forward = forward or self.forward
        valid_keys = self._get_valid_keys(x)
        y = {key: value for key, value in x.items() if key not in valid_keys}
        for key, value in x.items():
            if key not in valid_keys:
                continue
            value = forward(value)
            if isinstance(value, Returned):
                value = value.obj
                if self.append:
                    if isinstance(y, dict) and isinstance(value, dict):
                        if self.prefix:
                            value = {
                                key + '_' + child_key: child_value
                                for child_key, child_value in value.items()
                            }
                        y.update(value)
                        continue
                    elif isinstance(y, list) and isinstance(value, dict):
                        y.extend(value.values())
                        continue
                    elif isinstance(y, dict):
                        y = list(y.values())
                        if isinstance(value, (list, tuple)):
                            y.extend(value)
                            continue
            if isinstance(y, dict):
                y[key] = value
            else:
                y.append(value)
        if isinstance(y, dict):
            return type(x)(y)
        else:
            return y


class FinalTransform(Transform):
    """
    Mixin for determinstic transforms.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def is_final(self):
        return True

    def apply(self, x):
        """Apply the transform to a tensor

        Parameters
        ----------
        x : (C_inp, *spatial_inp) tensor

        Returns
        -------
        y : (C_out, *spatial_out) tensor
        """
        raise NotImplementedError()

    def make_inverse(self):
        """Generate the inverse transform"""
        return IdentityTransform()

    def inverse(self, *a, **k):
        """Apply the inverse transform recursively

        Parameters
        ----------
        x : [nested list or dict of] tensor
            Input tensors, with shape `(C, *shape)`

        Returns
        -------
        [nested list or dict of] tensor
            Output tensors. with shape `(C, *shape)`

        """
        return self.make_inverse()(*a, **k)

    def make_final(self, x, max_depth=float('inf')):
        return self


class IdentityTransform(FinalTransform):
    """Identity transform"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply(self, x):
        return x

    def make_inverse(self):
        return self


class SharedMixin:
    """
    Mixin for transforms that have parameters (e.g. random ones)
    that may be shared across tensors and/or channels or independent
    across tensors and/or channels.
    """

    @classmethod
    def _prepare_shared(cls, shared):
        if shared is True:
            shared = 'channels+tensors'
        if shared is False:
            shared = ''
        return shared

    def apply(self, x):
        if 'channels' in self.shared:
            xform = self.make_final(x[:1], max_depth=1)
        else:
            xform = self.make_final(x, max_depth=1)
        return xform.apply(x)

    def forward(self, *a, **k):
        return self._shared_forward(*a, **k)

    def _shared_forward(self, *a, _fallback=None, **k):
        _fallback = _fallback or super().forward
        if 'tensors' in self.shared:
            first_tensor = get_first_element(
                [a, k], include=self.include, exclude=self.exclude)
            if 'channels' in self.shared:
                transform = self.make_final(first_tensor[:1], max_depth=1)
            else:
                transform = self.make_final(first_tensor, max_depth=1)
            return transform(*a, **k)
        return _fallback(*a, **k)

    def make_per_channel(self, x, max_depth=float('inf'), *args, **kwargs):
        prm = dict(self.get_prm())
        prm.pop('shared', None)
        return PerChannelTransform([
            self.make_final(x[i:i+1], max_depth, *args, **kwargs)
            for i in range(len(x))
        ], **prm).make_final(x, max_depth-1)


class NonFinalTransform(SharedMixin, Transform):
    """
    Transforms whose parameters depend on features of the input
    transform (shape, dtype, etc)

    Parameters
    ----------
    shared : {'channels', 'tensors', 'channels+tensor', ''}

        - 'channel': the same transform is applied to all channels
            in a tensor, but different transforms are used in different
            tensors.
        - 'tensors': the same transform is applied to all tensors,
            but with a different transform for each channel.
        - 'channels+tensors' or True: the same transform is applied
            to all channels of all tensors.
        - '' or False: A different transform is applied to each
            channel and each tensor.
    """
    def __init__(self, *, shared=False, **kwargs):
        super().__init__(**kwargs)
        self.shared = self._prepare_shared(shared)

    def make_final(self, x, max_depth=float('inf'), *args, **kwargs):
        if self.is_final or max_depth == 0:
            return self
        return NotImplemented


class SpecialMixin:
    """Mixin for special transforms (i.e. transforms of transforms)"""

    def make_inverse(self):
        """Generate the inverse transform"""
        return IdentityTransform()

    def inverse(self, *a, **k):
        """Apply the inverse transform recursively

        Parameters
        ----------
        x : [nested list or dict of] tensor
            Input tensors, with shape `(C, *shape)`

        Returns
        -------
        [nested list or dict of] tensor
            Output tensors. with shape `(C, *shape)`

        """
        return self.make_inverse()(*a, **k)

    def make_final(self, x, max_depth=float('inf')):
        if x.is_final or max_depth == 0:
            return self
        return NotImplemented


class SequentialTransform(SpecialMixin, SharedMixin, Transform):
    """A sequence of transforms

    !!! example
        Sequences can be built explicitly, or simply by adding transforms
        together:
        ```python
        t1 = MultFieldTransform()
        t2 = GaussianNoiseTransform()
        seq = SequentialTransform([t1, t2])     # explicit
        seq = t1 + t2                           # implicit
        ```

        Sequences can also be extended by addition:
        ```python
        seq += SmoothTransform()
        ```

    """

    def __init__(self, transforms, **kwargs):
        """
        Parameters
        ----------
        transforms : list[Transform]
            A list of transforms to apply sequentially.

        Other Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensor', ''}

            - 'channel': the same sequence is applied to all channels
                in a tensor, but different transforms are used in different
                tensors.
            - 'tensors': the same transform is applied to all tensors,
                but with a different transform for each channel.
            - 'channels+tensors' or True: the same transform is applied
                to all channels of all tensors.
            - None or False: A different transform is applied to each
                channel and each tensor.
        include : str or list[str]
            List of keys to which the transform should apply
        exclude : str or list[str]
            List of keys to which the transform should not apply
        """
        shared = kwargs.pop('shared', False)
        super().__init__(**kwargs)
        self.shared = self._prepare_shared(shared)
        self.transforms = transforms

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        x = VirtualTensor.from_any(x, compute_stats=True)
        trf = []
        for t in self:
            t = t.make_final(x, max_depth=max_depth-1)
            x = t(x)
            trf.append(t)
        trf = SequentialTransform(trf, **self.get_prm())
        return trf

    @property
    def is_final(self):
        return all(t.is_final for t in self)

    def make_inverse(self):
        return SequentialTransform([t.make_inverse() for t in self])

    def forward(self, *a, **k):
        return self._shared_forward(*a, **k, _fallback=self._forward_impl)

    def _forward_impl(self, *args, **kwargs):
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
        for trf in self.transforms:
            with IncludeKeysTransform(trf, self.include), \
                 ExcludeKeysTransform(trf, self.exclude):
                x = trf(x)
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

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.transforms)})'


class PerChannelTransform(SpecialMixin, Transform):
    """Apply a different transform to each channel"""

    def __init__(self, transforms, **kwargs):
        """
        Parameters
        ----------
        transforms : list[Transform]
            A list of transforms to apply to each channel.
        """
        super().__init__(**kwargs)
        self.transforms = transforms

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        trf = []
        for i, t in enumerate(self.transforms):
            t = t.make_final(x[i:i+1], max_depth=max_depth-1)
            trf.append(t)
        prm = dict(self.get_prm())
        prm.pop('shared', None)
        trf = PerChannelTransform(trf, **prm)
        return trf

    def apply(self, x):
        results = []
        for i, t in enumerate(self.transforms):
            with ReturningTransform(t, self.returns), \
                 IncludeKeysTransform(t, self.include), \
                 ExcludeKeysTransform(t, self.exclude):
                results.append(t(x[i:i+1]))
        return Returned(recursive_cat(results))

    @property
    def is_final(self):
        return all(t.is_final for t in self.transforms)


class MaybeTransform(SpecialMixin, SharedMixin, Transform):
    """Randomly apply a transform

    !!! example "20% chance of adding noise"
        ```python
        import cornucopia as cc
        gauss = cc.GaussianNoiseTransform()
        ```
        Explicit call to the class:
        ```python
         img = cc.MaybeTransform(gauss, 0.2)(img)
        ```
        Implicit call using syntactic sugar:
        ```python
        img = (0.2 * gauss)(img)
        ```
        ```
    """

    def __init__(self, transform, prob=0.5, *, shared=False, **kwargs):
        """

        Parameters
        ----------
        transform : Transform
            A transform to randomly apply
        prob : float
            Probability to apply the transform
        shared : {'channels', 'tensors', 'channels+tensor', ''}
            Roll the dice once for all input tensors
        """
        super().__init__(**kwargs)
        self.shared = self._prepare_shared(shared)
        self.subtransform = transform
        self.prob = prob

    def throw_dice(self):
        return random.random() > 1 - self.prob

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if self.throw_dice():
            trf = self.subtransform
            with IncludeKeysTransform(trf, self.include), \
                 ExcludeKeysTransform(trf, self.exclude):
                return trf.make_final(x, max_depth=max_depth-1)
        else:
            return IdentityTransform()

    def __repr__(self):
        s = f'{repr(self.subtransform)}?'
        if self.prob != 0.5:
            s += f'[{self.prob}]'
        return s


class SwitchTransform(SpecialMixin, SharedMixin, Transform):
    """Randomly choose a transform to apply

    !!! example "Randomly apply either Gaussian or Chi noise"
        ```python
        import cornucopia as cc
        gauss = cc.GaussianNoiseTransform()
        chi = cc.ChiNoiseTransform()
        ```
        Explicit call to the class:
        ```python
        img = cc.SwitchTransform([gauss, chi])(img)
        ```
        Implicit call using syntactic sugar:
        ```python
        img = (gauss | chi)(img)
        ```
        Functional call:
        ```python
        img = cc.switch({gauss: 0.5, chi: 0.5})(img)
        ```
    """

    def __init__(self, transforms, prob=0, *, shared=False, **kwargs):
        """

        Parameters
        ----------
        transforms : list[Transform]
            A list of transforms to sample from
        prob : list[float]
            Probability of applying each transform
        shared : {'channels', 'tensors', 'channels+tensor', ''}
            Roll the dice once for all input tensors
        """
        super().__init__(**kwargs)
        self.shared = self._prepare_shared(shared)
        self.transforms = list(transforms)
        if not prob:
            prob = []
        self.prob = prob

    def _make_prob(self):
        prob = ensure_list(self.prob, len(self.transforms), default=0)
        sumprob = sum(prob)
        if not sumprob:
            prob = [1/len(self.transforms)] * len(self.transforms)
        else:
            prob = [x / sumprob for x in prob]
        return prob

    def throw_dice(self):
        prob = cumsum(self._make_prob())
        dice = random.random()
        for k in range(len(self.transforms)):
            if dice > 1 - prob[k]:
                return k
        return len(self.transforms) - 1

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        t = self.transforms[self.throw_dice()]
        t = t.make_final(x, max_depth=max_depth-1)
        with IncludeKeysTransform(t, self.include), \
             ExcludeKeysTransform(t, self.exclude):
            return t

    def __or__(self, other):
        if isinstance(other, SwitchTransform) \
                and not self.prob and not other.prob:
            return SwitchTransform([*self.transforms, *other.transforms])
        else:
            return SwitchTransform([self, other])

    def __ior__(self, other):
        if isinstance(other, SwitchTransform) \
                and not self.prob and not other.prob:
            self.transforms.append(other.transforms)
            return self
        else:
            return SwitchTransform([self, other])

    def __repr__(self):
        if self.prob:
            prob = self._make_prob()
            s = [f'{p} * {str(t)}' for p, t in zip(prob, self.transforms)]
        else:
            s = [str(t) for t in self.transforms]
        s = ' | '.join(s)
        s = f'({s})'
        return s

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class IncludeKeysTransform(SpecialMixin, Transform):
    """
    Context manager for keys to include

    !!! example "Use as a transform"
        ```python
        from cornucopia import IncludeKeysTransform
        newxform = IncludeKeysTransform(xform, "image)
        image, label = newxform(image=image, label=label)
        ```

    !!! example "Use as a context manager `with as`"
        ```python
        from cornucopia import IncludeKeysTransform
        with IncludeKeysTransform(xform, "image") as newxform:
            image, label = newxform(image=image, label=label)
        ```

    !!! example "Use as a context manager `with`"
        ```python
        from cornucopia import IncludeKeysTransform
        with IncludeKeysTransform(xform, "image"):
            image, label = xform(image=image, label=label)
        ```

    !!! example "Use as a context manager (alias)"
        ```python
        from cornucopia import context as ctx
        with ctx.include(xform, "image") as newxform:
            image, label = newxform(image=image, label=label)
        ```
    """

    def __init__(self, transform, keys, union=True):
        """
        Parameters
        ----------
        transform : Transform
            Transform to apply
        keys : [sequence of] str
            Keys to include
        union : bool
            Include the union of what was already included and `keys`
        """
        super().__init__()
        if keys and not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.transform = transform
        self.keys = list(keys) if keys else keys
        self.union = union

    def forward(self, *a, **k):
        with self as transform:
            return transform.forward(*a, **k)

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        with self as trf:
            final_trf = trf.make_final(x, max_depth)
            with IncludeKeysTransform(final_trf) as final_final_trf:
                return final_final_trf

    def make_inverse(self):
        with self as trf:
            inv_trf = trf.make_inverse()
            with IncludeKeysTransform(inv_trf) as final_inv_trf:
                return final_inv_trf

    def __enter__(self):
        self.include = self.transform.include
        self.transform.include = self.keys
        if self.union and self.include is not None:
            if self.transform.include is not None:
                self.transform.include = \
                    self.transform.include + self.include
            else:
                self.transform.include = self.include
        return self.transform

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.transform.include = self.include
        delattr(self, 'include')


class ExcludeKeysTransform(SpecialMixin, Transform):
    """
    Context manager for keys to exclude.
    Can also be used as a transform.

    !!! example "Use as a transform"
        ```python
        from cornucopia import ExcludeKeysTransform
        newxform = ExcludeKeysTransform(xform, "image)
        image, label = newxform(image=image, label=label)
        ```

    !!! example "Use as a context manager `with as`"
        ```python
        from cornucopia import ExcludeKeysTransform
        with ExcludeKeysTransform(xform, "image") as newxform:
            image, label = newxform(image=image, label=label)
        ```

    !!! example "Use as a context manager `with`"
        ```python
        from cornucopia import ExcludeKeysTransform
        with ExcludeKeysTransform(xform, "image"):
            image, label = xform(image=image, label=label)
        ```

    !!! example "Use as a context manager (alias)"
        ```python
        from cornucopia import context as ctx
        with ctx.exclude(xform, "image") as newxform:
            image, label = newxform(image=image, label=label)
        ```
    """

    def __init__(self, transform, keys, union=True):
        """
        Parameters
        ----------
        transform : Transform
            Transform to apply
        keys : [sequence of] str
            Keys to include
        union : bool
            Exclude the union of what was already excluded and `keys`
        """
        super().__init__()
        if keys and not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.transform = transform
        self.keys = list(keys) if keys else keys
        self.union = union

    def forward(self, *a, **k):
        with self as transform:
            return transform.forward(*a, **k)

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        with self as trf:
            final_trf = trf.make_final(x, max_depth)
            with ExcludeKeysTransform(final_trf) as final_final_trf:
                return final_final_trf

    def make_inverse(self):
        with self as trf:
            inv_trf = trf.make_inverse()
            with ExcludeKeysTransform(inv_trf) as final_inv_trf:
                return final_inv_trf

    def __enter__(self):
        self.exclude = self.transform.exclude
        self.transform.exclude = self.keys
        if self.union and self.exclude:
            if self.transform.exclude is not None:
                self.transform.exclude = \
                    self.transform.exclude + self.exclude
            else:
                self.transform.exclude = self.exclude
        return self.transform

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.transform.exclude = self.exclude
        delattr(self, 'exclude')


class SharedTransform(SpecialMixin, SharedMixin, Transform):
    """
    Context manager for sharing transforms across channels / tensors.
    Can also be used as a transform.

    !!! example "Use as a context manager (alias)"
        ```python
        from cornucopia import context as ctx
        with ctx.shared(xform, "channels") as newxform:
            image = newxform(image)
        ```
    """

    def __init__(self, transform, mode=unset):
        """
        Parameters
        ----------
        transform : Transform
            Transform to apply
        mode : {'channels', 'tensors', 'channels+tensor', ''}

        - 'channel': the same transform is applied to all channels
            in a tensor, but different transforms are used in different
            tensors.
        - 'tensors': the same transform is applied to all tensors,
            but with a different transform for each channel.
        - 'channels+tensors' or True: the same transform is applied
            to all channels of all tensors.
        - None or False: A different transform is applied to each
            channel and each tensor.

        """
        super().__init__()
        self.transform = transform
        self.mode = mode

    def forward(self, *a, **k):
        with self as transform:
            return transform.forward(*a, **k)

    def __enter__(self):
        self.hasattr = hasattr(self.transform, 'shared')
        self.saved_mode = getattr(self.transform, 'shared', None)
        if self.mode is not unset:
            self.transform.shared = self.mode
        return self.transform

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hasattr:
            self.transform.shared = self.saved_mode
        elif hasattr(self.transform, 'shared'):
            delattr(self.transform, 'shared')
        delattr(self, 'hasattr')
        delattr(self, 'saved_mode')


class ReturningTransform(SpecialMixin, Transform):
    """
    Context manager for sharing transforms across channels / tensors

    !!! example "Use as a context manager (alias)"
        ```python
        from cornucopia import context as ctx
        with ctx.returns(xform, "channels") as newxform:
            image = newxform(image)
        ```
    """

    def __init__(self, transform, returns=None):
        super().__init__()
        self.transform = transform
        self.returns = returns

    def __enter__(self):
        self.saved_returns = self.transform.returns
        self.transform.returns = self.returns
        return self.transform

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.transform.returns = self.saved_returns
        delattr(self, 'saved_returns')


class MappedTransform(SpecialMixin, Transform):
    """
    Transforms that are applied to specific positional or arguments

    !!! example
        ```python
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
        ```

    """

    def __init__(self, *mapargs, nested=False, default=None, **mapkwargs):
        """

        Parameters
        ----------
        mapargs : tuple[Transform]
            Transform to apply to positional arguments
        mapkwargs : dict[str, Transform]
            Transform to apply to keyword arguments
        nested : bool, default=False
            Recursively traverse the inputs until we find matching
            dictionaries. Only `mapkwargs` are accepted if "nested".
        default : Transform
            Transform to apply if nothing is specifically mapped
        """
        super().__init__(
            shared=mapkwargs.pop('shared', False),
            include=mapkwargs.pop('include', None),
            exclude=mapkwargs.pop('exclude', None),
        )
        self.mapargs = mapargs
        self.mapkwargs = mapkwargs
        self.nested = nested
        self.default = default
        if nested and mapargs:
            raise ValueError(
                'Cannot have both `nested` and positional transforms'
            )

    def forward(self, *args, **kwargs):
        if self.include is not None or self.exclude:
            def wrap(f):
                def ff(*a, **k):
                    with IncludeKeysTransform(f, self.include), \
                         ExcludeKeysTransform(f, self.exclude):
                        return f(*a, **k)
                return ff
        else:
            def wrap(f):
                return f

        if args:
            a0 = args[0]
            if isinstance(a0, Args):
                return self.forward(*a0)
            if isinstance(a0, Kwargs):
                return self.forward(**a0)
            if isinstance(a0, ArgsAndKwargs):
                return self.forward(*a0.args, **a0.kwargs)

        def default(x):
            if torch.is_tensor(x):
                return self.default(x) if self.default else x
            else:
                return self.forward(x) if self.nested else x

        if args:
            mapargs = tuple(wrap(f) if f else None for f in self.mapargs)
            mapargs += (default,) * max(0, len(args) - len(mapargs))
            args = tuple(f(a) if f else a for f, a in zip(mapargs, args))

        if kwargs:
            mapkwargs = {k: wrap(f) if f else None
                         for k, f in self.mapkwargs.items()}

            def func(key):
                return mapkwargs.get(key, default) or (lambda x: x)

            kwargs = {key: func(key)(value) for key, value in kwargs.items()}

        if args and kwargs:
            return ArgsAndKwargs(args, kwargs)
        elif kwargs:
            return Kwargs(kwargs)
        elif len(args) > 1:
            return Args(args)
        else:
            return args[0]

    def __repr__(self):
        s = []
        for v in self.mapargs:
            s += [f'{v}']
        for k, v in self.mapkwargs.items():
            s += [f'{k}={v}']
        s = ', '.join(s)
        return f'{type(self).__name__}({s})'


class RandomizedTransform(NonFinalTransform):
    """
    Transform generated by randomizing some parameters of another transform.

    !!! example "Gaussian noise with randomized variance"
        Object call
        ```python
        import cornucopia as cc
        hypernoise = RandomizedTransform(cc.GaussianNoise, [cc.Uniform(0, 10)])
        img = hypernoise(img)
        ```
        Functional call
        ```python
        import cornucopia as cc
        hypernoise = cc.randomize(cc.GaussianNoise)(cc.Uniform(0, 10))
        img = hypernoise(img)
        ```

    """

    def __init__(self, transform, sample, ksample=None,
                 *, shared=False, **kwargs):
        """

        Parameters
        ----------
        transform : callable(...) -> Transform
            A Transform subclass or a function that constructs a Transform.
        sample : [list or dict of] callable
            A collection of functions that generate parameter values provided
            to `transform`.

        Other Parameters
        ----------------
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Share random parameters across tensors and/or channels
        """
        super().__init__(shared=shared, **kwargs)
        self.sample = sample
        self.ksample = ksample
        self.subtransform = transform

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth)
        args = []
        kwargs = {}
        if isinstance(self.sample, (list, tuple)):
            args += [f() if isinstance(f, Sampler) else f for f in self.sample]
        elif hasattr(self.sample, 'items'):
            kwargs.update({k: f() if isinstance(f, Sampler) else f
                           for k, f in self.sample.items()})
        else:
            args += [self.sample() if isinstance(self.sample, Sampler)
                     else self.sample]
        if self.ksample:
            kwargs.update({k: f() if isinstance(f, Sampler) else f
                           for k, f in self.ksample.items()})
        xform = self.subtransform(*args, **kwargs)
        xform = xform.make_final(x, max_depth-1)
        return xform

    def __repr__(self):
        if type(self) is RandomizedTransform:
            try:
                if issubclass(self.subtransform, Transform):
                    return f'Randomized{self.subtransform.__name__}()'
            except TypeError:
                pass
        return super().__repr__()
