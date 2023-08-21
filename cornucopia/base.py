__all__ = ['Transform', 'SequentialTransform', 'RandomizedTransform',
           'MaybeTransform', 'MappedTransform', 'SwitchTransform',
           'BatchedTransform', 'MappedKeysTransform', 'MappedExceptKeysTransform',
           'randomize', 'map', 'switch', 'include', 'exclude', 'batch',
           'include_keys', 'exclude_keys']

import torch
from torch import nn
import random
from .utils.py import ensure_list, cumsum
from .random import Sampler


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
            return type(x[0])(**{k: _rec(*x2)
                                 for k in x[0].keys()
                                 for x2 in zip(x1[k] for x1 in x)})
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
    """Iterator across both args and kwargs"""
    def __init__(self, args, kwargs):
        self.args = Args(args)
        self.kwargs = Kwargs(kwargs)

    def __iter__(self):
        for a in self.args:
            yield a
        for v in self.kwargs:
            yield v


class Returned:
    """Internal object used to mark that this is an object returned
    by `transform_tensor` at the most nested level"""
    def __init__(self, obj):
        self.obj = obj


class include:
    """
    Context manager for keys to include

    !!! example
        ```python
        with include(xform, "image"):
            image, label = xform(image=image, label=label)
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
        if keys and not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.transform = transform
        self.keys = list(keys) if keys else keys
        self.union = union

    def __enter__(self):
        self.include = self.transform._include
        self.transform._include = self.keys
        if self.union and self.include is not None:
            if self.transform._include is not None:
                self.transform._include = self.transform._include + self.include
            else:
                self.transform._include = self.include

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.transform._include = self.include


class exclude:
    """
    Context manager for keys to exclude.

    !!! example
        ```python
        with exclude(xform, "image"):
            image, label = xform(image=image, label=label)
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
        if keys and not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.transform = transform
        self.keys = list(keys) if keys else keys
        self.union = union

    def __enter__(self):
        self.exclude = self.transform._exclude
        self.transform._exclude = self.keys
        if self.union and self.exclude:
            if self.transform._exclude is not None:
                self.transform._exclude = self.transform._exclude + self.exclude
            else:
                self.transform._exclude = self.exclude

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.transform._exclude = self.exclude


class Transform(nn.Module):
    """
    Base class for transforms.

    Transforms are parameter-free modules that usually act on tensors
    without a batch dimension (e.g., `[C, *spatial]`)

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
    transform would be defined as:
    ```python
    t = MaybeTransform(MultFieldTransform(shared=False), shared=True)
    ```

    Furthermore, the addition of two transforms implictly defines
    (or extends) a `SequentialTransform`:
    ```python
    t1 = MultFieldTransform()
    t2 = GaussianNoiseTransform()
    seq = t1 + t2
    ```

    """

    def __init__(self, *, returns=None, append=False, prefix=True, shared=False,
                 include=None, exclude=None):
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
        shared : bool or 'channels' or 'tensors'

            - If `True`: shared across tensors and channels
            - If `'channels'`: shared across channels
            - If `'tensors'`: shared across tensors
            - If `False`: independent across tensors and channels
        include : str or list[str]
            List of keys to which the transform should apply
        exclude : str or list[str]
            List of keys to which the transform should not apply
        """
        super().__init__()
        self.returns = returns
        self.append = append
        self.prefix = prefix
        self.shared = shared
        self._include = ensure_list(include) if include is not None else None
        self._exclude = ensure_list(exclude or tuple())

    def _apply_list(self, x, forward):
        """Apply forward pass to elements of a list"""
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
        print(y)
        return type(x)(y)

    def _get_valid_keys(self, x):
        valid_keys = x.keys()
        if self._include is not None:
            valid_keys = [k for k in valid_keys if k in self._include]
        if self._exclude:
            valid_keys = [k for k in valid_keys if k not in self._exclude]
        return valid_keys

    def _apply_dict(self, x, forward):
        """Apply forward pass to elements of a dict"""
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
                            value = {key + '_' + child_key: child_value
                                     for child_key, child_value in value.items()}
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
        # DEV: this function should in general not be overloaded
        out = self._forward(*a, **k)
        if isinstance(out, Returned):
            out = out.obj
        return out

    def _forward(self, *a, **k):
        if k:
            return self._forward(ArgsAndKwargs(a, k) if a else Kwargs(k))
        elif len(a) > 1:
            return self._forward(Args(a))
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
            x = self._forward_with_parameters(x, parameters=theta)
            return outtype(x)

        # not shared across images -> unfold
        if isinstance(x, (list, tuple)):
            return outtype(self._apply_list(x, self._forward))
        if hasattr(x, 'items'):
            x = self._apply_dict(x, self._forward)
            if not isinstance(x, dict) and outtype in (Kwargs, ArgsAndKwargs):
                return Args(x)
            else:
                return x

        # now we're working with a single tensor (or str)
        if self.shared is False:
            y = self.transform_tensor_perchannel(x)
        else:
            y = self.transform_tensor(x)
        if not isinstance(y, type(self.returns)):
            y = dict(input=x, output=y)
            y = prepare_output(y, self.returns)
        elif isinstance(self.returns, dict):
            for key, target in self.returns.items():
                if target == 'input':
                    y[key] = x
        elif isinstance(self.returns, (list, tuple)):
            for i, target in enumerate(self.returns):
                if target == 'input':
                    y[i] = x
        return Returned(y)

    def forward_with_parameters(self, x, parameters=None):
        """Apply the transform with pre-computed parameters

        Parameters
        ----------
        x : [nested list or dict of] tensor
            Input tensors, with shape `(C, *shape)`
        parameters : any
            Pre-computed parameters of the transform

        Returns
        -------
        [nested list or dict of] tensor
            Output tensors, with shape `(C, *shape)`

        """
        # DEV: this function should in general not be overloaded
        x = self._forward_with_parameters(x, parameters)
        if isinstance(x, Returned):
            x = x.obj
        return x

    def _forward_with_parameters(self, x, parameters=None):
        forward = lambda elem: self._forward_with_parameters(elem, parameters=parameters)
        if isinstance(x, (list, tuple)):
            return self._apply_list(x, forward)
        if hasattr(x, 'items'):
            return self._apply_dict(x, forward)
        x = self._apply_transform(x, parameters=parameters)
        return x

    def transform_tensor(self, x):
        """Apply the transform to a single tensor

        Parameters
        ----------
        x : tensor
            A single input tensor, with shape `(C, *shape)`.

        Returns
        -------
        x : tensor
            A single output tensor, with shape `(C, *shape)`.

        """
        # DEV: this function can be overloaded if `shared` is not supported
        theta = self.get_parameters(x)
        return self.apply_transform(x, parameters=theta)

    def _transform_tensor(self, x):
        # This is only for internal use
        return Returned(self.transform_tensor(x))

    def transform_tensor_perchannel(self, x):
        """Apply the transform to each channel of a single tensor

        Parameters
        ----------
        x : tensor
            A single input tensor, with shape `(C, *shape)`.

        Returns
        -------
        x : tensor
            A single output tensor, with shape `(C, *shape)`.

        """
        # DEV: This function should usually not be overloaded
        channels = x.unbind(0)
        channels = [self.transform_tensor(c[None]) for c in channels]
        return _recursive_cat(channels, dim=0)

    def _transform_tensor_perchannel(self, x):
        # This is only for internal use
        return Returned(self.transform_tensor_perchannel(x))

    def get_parameters(self, x):
        """Compute the parameters of a transform from an input tensor

        Parameters
        ----------
        x : tensor
            A single input tensor, with shape `(C, *shape)`.

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
        x : tensor
            A single input tensor, with shape `(C, *shape)`.
        parameters : any
            Precomputed parameters

        Returns
        -------
        x : tensor
            A single output tensor, with shape `(C, *shape)`.

        """
        raise NotImplementedError("This function should be implemented "
                                  "in Transforms that handle `shared`.")

    def _apply_transform(self, x, parameters):
        # This is only for internal use
        return Returned(self.apply_transform(x, parameters))

    def transform_tensor_and_get_parameters(self, x):
        """Apply the transform to a single tensor and return its parameters

        Parameters
        ----------
        x : tensor
            A single input tensor, with shape `(C, *shape)`.

        Returns
        -------
        x : tensor
            A single output tensor, with shape `(C, *shape)`.
        parameters : any
            Computed parameters.

        """

        # DEV: This function should probably not be overloaded
        theta = self.get_parameters(x)
        return self.apply_transform(x, parameters=theta), theta

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


class SequentialTransform(Transform):
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
        """
        super().__init__(**kwargs)
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
        for trf in self.transforms:
            with include(trf, self._include), exclude(trf, self._exclude):
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


class MaybeTransform(Transform):
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
        shared : bool
            Roll the dice once for all input tensors
        """
        super().__init__(shared=shared, **kwargs)
        self.subtransform = transform
        self.prob = prob

    def apply_transform(self, x, parameters):
        if parameters > 1 - self.prob:
            trf = self.subtransform
            with include(trf, self._include), exclude(trf, self._exclude):
                return trf(x)
        else:
            return x

    def get_parameters(self, x):
        return random.random()

    def __repr__(self):
        s = f'{repr(self.subtransform)}?'
        if self.prob != 0.5:
            s += f'[{self.prob}]'
        return s


class SwitchTransform(Transform):
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
        shared : bool
            Roll the dice once for all input tensors
        """
        super().__init__(shared=shared, **kwargs)
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

    def apply_transform(self, x, parameters):
        prob = cumsum(self._make_prob())
        for k, t in enumerate(self.transforms):
            if parameters > 1 - prob[k]:
                with include(t, self._include), exclude(t, self._exclude):
                    return t(x)
        return x

    def get_parameters(self, x):
        return random.random()

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


def switch(map):
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

    Parameters
    ----------
    map : dict(Transform -> float)
        A dictionary that maps transforms to probabilities of being chosen

    Returns
    -------
    SwitchTransform

    """
    return SwitchTransform(map.keys(), map.values())


class RandomizedTransform(Transform):
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

    def __init__(self, transform, sample, ksample=None, *, shared=False, **kwargs):
        """

        Parameters
        ----------
        transform : callable(...) -> Transform
            A Transform subclass or a function that constructs a Transform.
        sample : [list or dict of] callable
            A collection of functions that generate parameter values provided
            to `transform`.
        """
        super().__init__(shared=shared, **kwargs)
        self.sample = sample
        self.ksample = ksample
        self.subtransform = transform

    def get_parameters(self, x):
        args = []
        kwargs = {}
        if isinstance(self.sample, (list, tuple)):
            args += [f() if isinstance(f, Sampler) else f for f in self.sample]
        elif hasattr(self.sample, 'items'):
            kwargs.update({k: f() if isinstance(f, Sampler) else f
                           for k, f in self.sample.items()})
        else:
            args += [self.sample() if isinstance(self.sample, Sampler) else self.sample]
        if self.ksample:
            kwargs.update({k: f() if isinstance(f, Sampler) else f
                           for k, f in self.ksample.items()})
        return self.subtransform(*args, **kwargs)

    def apply_transform(self, x, parameters):
        with include(parameters, self._include), \
             exclude(parameters, self._exclude):
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
            Recursively traverse the inputs until we find matching dictionaries.
            Only `mapkwargs` are accepted if "nested"
        default : Transform
            Transform to apply if nothing is specifically mapped
        """
        super().__init__(shared='channels')
        self.mapargs = mapargs
        self.mapkwargs = mapkwargs
        self.nested = nested
        self.default = default
        if nested and mapargs:
            raise ValueError('Cannot have both `nested` and positional transforms')

    def forward(self, *args, **kwargs):
        if self._include is not None or self._exclude:
            def wrap(f):
                def ff(*a, **k):
                    with include(f, self._include), exclude(f, self._exclude):
                        return f(*a, **k)
                return ff
        else:
            wrap = lambda f: f

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
            func = lambda key: mapkwargs.get(key, default) or (lambda x: x)
            kwargs = {key: func(key)(value) for key, value in kwargs.items()}

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


class MappedKeysTransform(Transform):
    """Apply a transform to a set of keys"""

    def __init__(self, transform, keys):
        """

        Parameters
        ----------
        transform : Transform
            Transform to apply
        keys : [sequence of] str
            Keys to which to apply the transform
        """
        super().__init__()
        self.transform = transform
        self.keys = keys

    def forward(self, *a, **k):
        with include(self.transform, self.keys):
            return self.transform.forward(*a, **k)


class MappedExceptKeysTransform(MappedTransform):
    """Apply a transform to all but a set of keys"""

    def __init__(self, transform, keys):
        """

        Parameters
        ----------
        transform : Transform
            Transform to apply
        keys : [sequence of] str
            Keys to which *not* to apply the transform
        """
        super().__init__()
        self.transform = transform
        self.keys = keys

    def forward(self, *a, **k):
        with exclude(self.transform, self.keys):
            return self.transform.forward(*a, **k)


def map(*mapargs, nested=False, default=None, **mapkwargs):
    """Alias for MappedTransform

    !!! example
        ```python
        import cornucopia as cc
        import torch

        img = torch.randn([1, 32, 32])
        seg = torch.randn([3, 32, 32]).softmax(0)

        # positional variant
        trf = cc.map(GaussianNoise(), None)
        img, seg = trf(img, seg)

        # keyword variant
        trf = cc.map(image=GaussianNoise())
        img, seg = trf(image=img, label=seg)

        # alternative version
        dat = {'img': torch.randn([1, 32, 32]),
               'seg': torch.randn([3, 32, 32]).softmax(0)}
        dat = cc.map(img=GaussianNoise(), nested=True)(dat)
        ```
    """
    return MappedTransform(*mapargs, nested=nested, default=default, **mapkwargs)


def include_keys(transform, keys):
    """Alias for MappedKeysTransform

    !!! example
        Apply `geom` to all images, and apply `noise` to `img` only.
        ```python
        import cornucopia as cc

        geom = cc.RandomElasticTransform()
        noise = cc.GaussianNoiseTransform()
        trf = geom + cc.include_keys(noise, "image")
        img, lab = trf(image=img, label=lab)
        ```
    """
    return MappedKeysTransform(transform, keys)


def exclude_keys(transform, keys):
    """Alias for MappedExceptKeysTransform

    !!! example
        Apply `geom` to all images, and apply `noise` to all images
        except `lab`.
        ```python
        import cornucopia as cc

        geom = cc.RandomElasticTransform()
        noise = cc.GaussianNoiseTransform()
        trf = geom + cc.exclude_keys(noise, "label")
        img, lab = trf(image=img, label=lab)
        ```
    """
    return MappedExceptKeysTransform(transform, keys)


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
        valid_keys = kwargs.keys()
        if self._include is not None:
            valid_keys = [k for k in valid_keys if k in self._include]
        if self._exclude:
            valid_keys = [k for k in valid_keys if k not in self._exclude]
        kwargs = {k: _recursive_cat(v) if k in valid_keys else v
                  for k, v in kwargs.items()}
        if args and kwargs:
            return ArgsAndKwargs(args, kwargs)
        elif kwargs:
            return Kwargs(kwargs)
        elif len(args) > 1:
            return Args(args)
        else:
            return args[0]


class BatchedTransform(nn.Module):
    """Apply a transform to a batch

    !!! example
        Functional call:
        ```python
        batched_transform = cc.batch(transform)
        img, lab = batched_transform(img, lab)   # input shapes: [B, C, X, Y, Z]
        ```
        Object call:
        ```python
        batched_transform = c.BatchedTransform(transform)
        img, lab = batched_transform(img, lab)   # input shapes: [B, C, X, Y, Z]
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

    def forward(self, *args):

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
                raise TypeError(f'Don\'t know what to do with type {type(x)}')

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
                return type(x0)(pack([x1[i] for x1 in x]) for i in range(len(x0)))
            elif hasattr(x0, 'items'):
                return {key: pack([x1[key] for x1 in x]) for key in x0.keys()}
            elif torch.is_tensor(x0):
                return torch.stack(x)
            else:
                raise TypeError(f'Don\'t know what to do with type {type(x0)}')

        batch = []
        for elem in unpack(args):
            batch.append(self.transform(*elem))
        batch = pack(batch)
        return batch[0] if len(args) == 1 else tuple(batch)


def batch(transform):
    """Apply a transform to a batch

    !!! example
        Functional call:
        ```python
        batched_transform = cc.batch(transform)
        img, lab = batched_transform(img, lab)   # input shapes: [B, C, X, Y, Z]
        ```
        Object call:
        ```python
        batched_transform = c.BatchedTransform(transform)
        img, lab = batched_transform(img, lab)   # input shapes: [B, C, X, Y, Z]
        ```

    Parameters
    ----------
    transform : Transform
        Transform to apply to a batched tensor or to a nested
        structure of batched tensors

    Returns
    -------
    trf : BatchedTransform

    """
    return BatchedTransform(transform)


def prepare_output(results, returns):
    """Prepare object returned by `apply_transform`

    Parameters
    ----------
    results : dict[str, tensor]
        Named results
    returns : list[str] or dict[str, str] or str
        Structure describing which results should be returned.
        The results will be returned in an object of the same type, with
        the requested results associated to the same keys (if `dict`) or
        same position (if `list`). If a `str`, the requested tensor is
        returned.

    Returns
    -------
    requested_results : list[tensor] or dict[str, tensor] or tensor

    """
    if returns is None:
        if torch.is_tensor(results):
            return results
        else:
            return results.get('output', None)
    if isinstance(returns, dict):
        return type(returns)(
            **{key: results.get(target, None) for key, target in returns.items()})
    elif isinstance(returns, (list, tuple)):
        return type(returns)(
            [results.get(target, None) for target in returns])
    else:
        return results.get(returns, None)


def flatstruct(x):
    """Flatten a nested structure of tensors"""

    def _flatten(nested):
        if isinstance(nested, dict):
            flat = type(nested)()
            is_dict = True
            for key, elem in nested.items():
                elem = _flatten(elem)
                if isinstance(elem, dict):
                    for subkey, subelem in elem.items():
                        flat[subkey] = subelem
                elif not isinstance(elem, (dict, list, tuple)):
                    flat[key] = elem
                else:
                    is_dict = False
                    flat[key] = elem
            if not is_dict:
                flat, flatdict = [], flat
                for elem in flatdict.values():
                    if not isinstance(elem, (dict, list, tuple)):
                        flat.append(elem)
                    else:
                        flat.extend(elem)
            return flat
        elif isinstance(nested, (list, tuple)):
            flat = []
            for elem in nested:
                elem = _flatten(elem)
                if not isinstance(elem, (dict, list, tuple)):
                    flat.append(elem)
                elif isinstance(elem, dict):
                    flat.extend(elem.values())
                else:
                    flat.extend(elem)
            flat = type(nested)(flat)
            return flat
        else:
            return nested

    return _flatten(x)
