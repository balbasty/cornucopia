__all__ = [
    'Transform',
    'FinalTransform',
    'NonFinalTransform',
]
# stdlib
import inspect
import random
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Optional, Union
from math import inf

# dependencies
import torch
from torch import nn
from torch import Tensor

# internal
from .random import Sampler
from .utils.py import ensure_list, cumsum
from .baseutils import (
    Arguments, Arg, Args, Kwargs, ArgsAndKwargs, NoArguments, Returned, VirtualTensor,
    get_first_element, prepare_output, UNSET, recursive_cat,
)


class Transform(nn.Module, ABC):
    """Base class for all transforms."""

    def __init__(
        self, *,
        returns: Union[str, List[str], Dict[str, str], None] = None,
        append: bool = False,
        prefix: Union[bool, str] = True,
        include: Union[str, List[str], None] = None,
        exclude: Union[str, List[str], None] = None,
    ):
        """
        Parameters
        ----------
        returns : [list or dict of] str, optional
            Which tensors to return. Can be a nested structure.
            Most transforms accept `'input'` and `'output'` as valid
            returns. The default is `'output'`.
        append : bool, default=False
            Append the (structure of) returned tensors to the parent
            structure.
        prefix : bool or str, default=True
            If `append` and parent is a dict, prefix the returned key
            before inserting it in the output dictionary.
            If `True`, the prefix is the input key.
        include : str or list[str], optional
            List of keys to which the transform should apply.
            Default: all.
        exclude : str or list[str], optional
            List of keys to which the transform should not apply.
            Default: none.
        """
        super().__init__()
        self.returns = returns
        self.append = append
        self.prefix = prefix
        self.include = ensure_list(include) if include is not None else None
        self.exclude = ensure_list(exclude or tuple())

    @property
    def is_final(self) -> bool:
        """Whether the transform is final (i.e., deterministic) or not."""
        return False

    def get_prm(self) -> dict:
        """Get the parameters of the transform, for use in subtransforms."""
        return dict(
            returns=self.returns,
            append=self.append,
            prefix=self.prefix,
            include=self.include,
            exclude=self.exclude,
        )

    def __enter__(self) -> "Transform":
        # On most tranfsorms, this does nothing but return the transform
        # itself. However, some subclasses use this to act as context
        # managers that temporarily modify the transform's behavior.
        # See: `IncludeKeysTransform` and `ExcludeKeysTransform`.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def __add__(self, other: "Transform") -> "SequentialTransform":
        return SequentialTransform([self, other])

    def __radd__(self, other: "Transform") -> "SequentialTransform":
        return SequentialTransform([other, self])

    def __iadd__(self, other: "Transform") -> "SequentialTransform":
        return SequentialTransform([self, other])

    def __mul__(self, prob: float) -> "MaybeTransform":
        return MaybeTransform(self, prob)

    def __rmul__(self, prob: float) -> "MaybeTransform":
        return MaybeTransform(self, prob)

    def __imul__(self, prob: float) -> "MaybeTransform":
        return MaybeTransform(self, prob)

    def __or__(self, other: "Transform") -> "SwitchTransform":
        return SwitchTransform([self, other])

    def __ior__(self, other: "Transform") -> "SwitchTransform":
        return SwitchTransform([self, other])

    def __call__(self, *a, **k):
        # Use the torch machinery, although `Returned` objects get unwrapped.
        out = super().__call__(*a, **k)
        if isinstance(out, Returned):
            out = out.obj
        return out

    def forward(self, *a, **k) -> Returned:
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
        # We wrap positional and keywork arguments in special classes
        # to differentiate from inputs that are lists or dicts.
        x = args = Arguments(*a, **k)

        if not args:
            # If no input arguments, return None.
            # NOTE: Only `NoArguments()` reduces to `False`.
            return None

        # Arguments are passed to `_Forward.__init__` and `_Forward.__call__`.
        # The former is preserved as is in the `_Forward` object, and passed
        # to each `xform` or `make_final` call, while the latter is
        # recursively unwrapped and processed by `_Forward.__call__`.
        return self._Forward(self, args, **self.get_prm())(x)

    class _Forward:

        def __init__(self, transform, args: Arguments, **prm) -> None:
            self.transform = transform
            self.args = args
            self.prm = prm

        @property
        def include(self) -> Optional[List[str]]:
            return self.prm.get('include')

        @property
        def exclude(self) -> Optional[List[str]]:
            return self.prm.get('exclude')

        @property
        def append(self) -> bool:
            return self.prm.get('append', False)

        @property
        def prefix(self) -> Union[bool, str]:
            return self.prm.get('prefix', True)

        @property
        def returns(self) -> Optional[Union[str, List[str], Dict[str, str]]]:
            return self.prm.get('returns')

        def __call__(self, x: Union[Tensor, List, Dict, Arguments]):
            if isinstance(x, NoArguments):
                return None

            # At this point, there is a single positional argument
            # (which may be an Args, Kwargs, or ArgsAndKwargs object)
            # and no keyword arguments. We save the original input type
            # to be able to return the same kind of `Parameters` object.
            intype = type(x)

            # If the input is an `Arguments` object, convert it to list/dict
            if isinstance(x, Arguments):
                x = x.unwrap()

            def outtype(x):
                # Convert back to the original input type if needed
                if intype is ArgsAndKwargs:
                    args, kwargs = x
                elif intype is Args:
                    args, kwargs = x, {}
                elif intype is Kwargs:
                    args, kwargs = (), x
                else:
                    return x
                return Arguments(*args, **kwargs)

            # Not shared across tensors -> unfold
            if isinstance(x, (list, tuple)):
                return outtype(self._forward_list(x))

            if hasattr(x, 'items'):
                x = self._forward_dict(x)
                if not isinstance(x, dict) and intype in (Kwargs, ArgsAndKwargs):
                    return Args(*x)
                else:
                    return outtype(x)

            # ---- Now we're working with a single tensor (or str) ----

            # Apply the transform to the input tensor
            y = self.transform.safe_xform(x, args=self.args)

            # Most transforms return a well-formatted `Returned` object,
            # which contain all possible outputs of a transform, mapped
            # into a structure specified by the `returns` argument.
            if not isinstance(y, Returned):
                # When they do not, we have to build the `Returned`
                # object ourselves.
                if not isinstance(y, type(self.returns)):
                    # The transform returned a single output (likely a
                    # tensor), which we assign to the `output` key,
                    # while the input is assigned to the `input` key.
                    y = dict(input=x, output=y)
                    y = prepare_output(y, self.returns).obj
                else:
                    # `returns` and `y` have the same type, but `y` may
                    # have been obtained from a subtransform. We cannot
                    # guarantee that the keys of `y` are the same as
                    # those of `returns`, but we can insert the correct
                    # `input` tensor, it it was requested.
                    # NOTE: we cannot break early once `"input"` is
                    # encountered, because multiple outputs elements can
                    # contain the same target
                    # (e.g. `returns=['input', 'input']`).
                    if isinstance(self.returns, dict):
                        for key, target in self.returns.items():
                            if target == 'input':
                                y[key] = x
                    elif isinstance(self.returns, (list, tuple)):
                        for i, target in enumerate(self.returns):
                            if target == 'input':
                                y[i] = x
                # Wrap output in a `Returned` object, so that helpers
                # know how to handle it (e.g. when `append=True`).
                y = Returned(y)

            # ---- Now we're working with a `Returned` object ----
            return y

        def _get_valid_keys(self, x: Dict[str, str]) -> List[str]:
            valid_keys = x.keys()
            if self.include is not None:
                valid_keys = [k for k in valid_keys if k in self.include]
            if self.exclude:
                valid_keys = [k for k in valid_keys if k not in self.exclude]
            return valid_keys

        def _forward_list(
            self, x: list, forward: Optional[Callable] = None
        ) -> list:
            """Apply forward pass to elements of a list"""
            forward = forward or self
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

        def _forward_dict(
            self, x: dict, forward: Optional[Callable] = None
        ) -> Union[dict, list]:
            """Apply forward pass to elements of a dict"""
            forward = forward or self
            valid_keys = self._get_valid_keys(x)

            # Initialise output dictionary with input keys and values
            # that *will not* be transformed so that they are preserved.
            y = {key: value for key, value in x.items() if key not in valid_keys}

            # For each input item, apply the transform and save its outputs
            for key, value in x.items():
                if key not in valid_keys:
                    continue

                # Compute prefix
                prefix = self.prefix
                if prefix and not isinstance(self.prefix, str):
                    prefix = key

                # Apply transform to input value
                value = forward(value)

                # Deal with returned values.
                if isinstance(value, Returned):
                    value = value.obj

                    if self.append:

                        if isinstance(y, dict) and isinstance(value, dict):
                            # Insert the returned values in the output dictionary,
                            # using a new key, so that the input value is preserved.
                            if prefix:
                                value = {
                                    prefix + '.' + child_key: child_value
                                    for child_key, child_value in value.items()
                                }
                            y.update(value)
                            continue

                        if isinstance(y, dict):
                            # The transform did not return a dictionary, so
                            # we cannot insert its values in the output dict.
                            # We transform it into a list and append the
                            # returned values to it.
                            y = list(y.values())

                        if isinstance(value, (list, tuple)):
                            # Append the returned values to the output list.
                            y.extend(value)
                            continue

                        if isinstance(value, dict):
                            # The output dictionary was previously transformed
                            # into a list, so we cannot insert the returned
                            # (key, value) pairs. We append the values instead.
                            y.extend(value.values())
                            continue

                # We insert the returned value (whether it is a single tensor,
                # or a nested strucutre of tensors) it in the output dictionary,
                # in place of the input value.
                if isinstance(y, dict):
                    y[key] = value
                else:
                    y.append(value)

            if isinstance(y, dict):
                return type(x)(y)
            else:
                return y

    def safe_xform(
        self, x: Tensor, /,
        args: Arguments = NoArguments(),
    ) -> Returned:
        # Wrapper that calls `xform`, but only passes `args` if the
        # `xform` method accepts it, to avoid errors with legacy `xform`.
        if 'args' in inspect.signature(self.xform).parameters:
            return self.xform(x, args=args)
        else:
            return self.xform(x)

    def xform(
        self, x: Tensor, /,
        args: Arguments = NoArguments(),
    ) -> Returned:
        """Apply the transform to a tensor

        Parameters
        ----------
        x : (C_inp, *spatial_inp) tensor
            A single input tensor
        args: Arguments, optional
            The original inputs arguments to the transform, in case
            they are needed.

        Returns
        -------
        y : Returned | (C_out, *spatial_out) tensor
            A single output tensor, or a `Returned` object containing
            multiple output tensors and their corresponding keys.

        """
        raise NotImplementedError("This transform does not implement `xform`.")

    def safe_make_final(
        self, x: Tensor, /,
        max_depth: int = inf,
        args: Arguments = NoArguments(),
    ) -> Returned:
        # Wrapper that calls `make_final`, but only passes `args` if the
        # `make_final` method accepts it.
        if 'args' in inspect.signature(self.make_final).parameters:
            return self.make_final(x, max_depth, args=args)
        else:
            return self.make_final(x, max_depth)

    def make_final(
        self, x: Tensor, /,
        max_depth: int = inf,
        args: Arguments = NoArguments(),
    ) -> "Transform":
        """
        Generate a final (i.e., deterministic) version of the transform.

        Parameters
        ----------
        x : tensor
            A single input tensor, with shape `(C, *shape)`.
        max_depth : int | {inf}
            Maximum depth to apply `make_final` recursively.
            If not `inf`, the resulting transform may not be fully final.
            Default: no limit.
        args: Arguments, optional
            The original inputs arguments to the transform, in case
            they are needed.

        Returns
        -------
        Transform
            A final version of the transform.
        """
        if self.is_final or max_depth == 0:
            return self
        raise NotImplementedError(
            "This transform does not implement `make_final`."
        )

    def inverse(self, *a, **k) -> "FinalTransform":
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

    def make_inverse(self) -> "FinalTransform":
        """Generate the inverse transform"""
        # We fallback to the identity, rather than raising an error.
        return IdentityTransform()


class FinalTransform(Transform):
    """
    Base class for determinstic transforms.

    Final transforms *must* implement the `xform` method.
    """

    @property
    def is_final(self) -> bool:
        return True

    def make_final(
        self, x: Tensor, /,
        max_depth: int = inf,
        args: Arguments = NoArguments(),
    ) -> "FinalTransform":
        return self


class IdentityTransform(FinalTransform):
    """Identity transform"""

    def xform(
        self, x: Tensor, /, args: Arguments = NoArguments()
    ) -> Returned:
        return prepare_output(dict(input=x, output=x), self.returns)

    def make_inverse(self) -> "IdentityTransform":
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

    def xform(
        self, x: Tensor, /, args: Arguments = NoArguments(),
    ) -> Returned:
        template = x
        if 'channels' in self.shared:
            # Use the first channel only to compute the final transform
            template = template[:1]
        # Compute the next form of this transform
        # NOTE: we do not use `max_depth=inf` because the `shared` option
        # may differ across transformations in the hierarchy. For example,
        # the top-level parameters of a transformation may be shared
        # (e.g., the number of control points in a bias field), but not
        # the lower level ones (e.g., the values of the control points).
        transformation = self.safe_make_final(template, 1, args=args)
        if transformation is self:
            # Avoid infinite recursion. This should not happen.
            raise ValueError(
                f"The transform is not final, but calling `make_final` "
                f"returned itself. Transform: {self}"
            )
        # Apply the final transform to all channels
        return transformation.safe_xform(x, args=args)

    def forward(self, *a, **k) -> Returned:
        return self._shared_forward(*a, **k)

    def _shared_forward(self, *a, _fallback=None, **k) -> Returned:
        _fallback = _fallback or super().forward
        args = Arguments(*a, **k)
        a, k = args.to_args_kwargs()

        if 'tensors' in self.shared:
            # Get the first valid tensor across all inputs, and use it
            # as the template to compute the final transform.
            first_tensor = get_first_element(
                [a, k], include=self.include, exclude=self.exclude)
            if 'channels' in self.shared:
                # Get the first channel only, to compute the final transform.
                first_tensor = first_tensor[:1]
            # Compute the next form of this transform...
            transform = self.safe_make_final(first_tensor, 1, args=args)
            # ...and apply it to all tensors.
            return transform(*a, **k)

        # Else, we let `xform` deal with shared parameters across channels.
        return _fallback(*a, **k)

    def make_per_channel(
        self, x: Tensor, /,
        max_depth: int = float('inf'),
        args: Arguments = NoArguments(),
        **kwargs
    ) -> "PerChannelTransform":
        prm = dict(self.get_prm())
        prm.pop('shared', None)
        return PerChannelTransform([
            self.safe_make_final(x[i:i+1], max_depth, args=args, **kwargs)
            for i in range(len(x))
        ], **prm).safe_make_final(x, max_depth-1)


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
    def __init__(self, *, shared: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shared = self._prepare_shared(shared)


class SequentialTransform(SharedMixin, Transform):
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

    def __init__(self, transforms: List[Transform], **kwargs) -> None:
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

    def make_final(
        self, x: Tensor, /,
        max_depth: int = inf,
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        if self.is_final:
            return self
        # x = VirtualTensor.from_any(x, compute_stats=True)
        trf = []
        for t in self:
            t = t.safe_make_final(x, max_depth=max_depth-1, args=args)
            x = t(x)
            args = Arguments(x)
            trf.append(t)
        trf = SequentialTransform(trf, **self.get_prm())
        return trf

    @property
    def is_final(self) -> bool:
        return all(t.is_final for t in self)

    def make_inverse(self) -> Transform:
        return SequentialTransform([t.make_inverse() for t in reversed(self)])

    def forward(self, *a, **k) -> Returned:
        # If the entire sequence is shared across tensors, we use the
        # behavior from `SharedMixin`, which is to call `make_final` on
        # the first valid tensor, and apply the resulting transform to all
        # tensors.
        # Finalizing a sequence of transforms is a bit tricky, but sequences
        # are not shared by default, so this should rarely be used.
        # If the sequence is not shared (or onlt shared across channels),
        # we simply apply the transforms sequentially.
        return self._shared_forward(*a, **k, _fallback=self._forward_impl)

    def _forward_impl(self, *args, **kwargs):
        x = Arguments(*args, **kwargs)
        for trf in self:
            with IncludeKeysTransform(trf, self.include), \
                 ExcludeKeysTransform(trf, self.exclude):
                x = trf(x)
        return x

    def xform(
        self, x: Tensor, /, args: Arguments = NoArguments()
    ) -> Returned:
        # This should only be called when a Layer's `make_final` returns
        # a `SequentialTransform` (i.e., it is created implictly under
        # the hood, not explicitly by the user).
        # In such cases, `shared=False` and hopefully we can just fallback
        # to `forward()`.
        #
        # FIXME
        #   what happens if there's weird stuff in returns/include/exclude?
        return self(x)

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self) -> Iterator[Transform]:
        for t in self.transforms:
            yield t

    def __getitem__(self, item: Union[int, slice]) -> Transform:
        if isinstance(item, slice):
            return SequentialTransform(self.transforms[item])
        else:
            return self.transforms[item]

    def __repr__(self) -> str:
        return f'{type(self).__name__}({repr(self.transforms)})'


class PerChannelTransform(Transform):
    """Apply a different transform to each channel"""

    def __init__(self, transforms: List[Transform], **kwargs) -> None:
        """
        Parameters
        ----------
        transforms : list[Transform]
            A list of transforms to apply to each channel.
        """
        super().__init__(**kwargs)
        self.transforms = transforms

    def make_final(
        self, x: Tensor, /,
        max_depth: int = inf,
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        trf = []
        for i, t in enumerate(self.transforms):
            t = t.safe_make_final(x[i:i+1], max_depth-1, args=args)
            trf.append(t)
        prm = dict(self.get_prm())
        prm.pop('shared', None)
        trf = PerChannelTransform(trf, **prm)
        return trf

    def xform(
        self, x: Tensor, /, args: Arguments = NoArguments()
    ) -> Returned:
        results = []
        for i, t in enumerate(self.transforms):
            with (
                ReturningTransform(t, self.returns),
                IncludeKeysTransform(t, self.include),
                ExcludeKeysTransform(t, self.exclude)
            ):
                results.append(t(x[i:i+1]))
        return Returned(recursive_cat(results))

    @property
    def is_final(self) -> bool:
        return all(t.is_final for t in self.transforms)


class MaybeTransform(SharedMixin, Transform):
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

    !!! changedin "![v0.4](https://img.shields.io/badge/v0.4-yellow) \
                   Default for `shared` changed from `False` to `True`"
    """
    def __init__(
        self,
        transform: Transform,
        prob: float = 0.5,
        *,
        shared: bool = True,
        **kwargs
    ) -> None:
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

    def throw_dice(self) -> bool:
        return random.random() > 1 - self.prob

    def make_final(
        self, x: Tensor, /,
        max_depth: int = float('inf'),
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        if self.throw_dice():
            trf = self.subtransform
            with (
                IncludeKeysTransform(trf, self.include),
                ExcludeKeysTransform(trf, self.exclude)
            ):
                return trf.safe_make_final(x, max_depth-1, args=args)
        else:
            return IdentityTransform()

    def __repr__(self) -> str:
        s = f'{repr(self.subtransform)}?'
        if self.prob != 0.5:
            s += f'[{self.prob}]'
        return s


class SwitchTransform(SharedMixin, Transform):
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

    !!! changedin "![v0.4](https://img.shields.io/badge/v0.4-yellow) \
                   Default for `shared` changed from `False` to `True`"
    """

    def __init__(
        self,
        transforms: List[Transform],
        prob: Union[float, List[float]] = 0,
        *,
        shared: bool = True,
        **kwargs
    ) -> None:
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
        self.prob = prob or []

    def _make_prob(self) -> List[float]:
        prob = ensure_list(self.prob, len(self.transforms), default=0)
        sumprob = sum(prob)
        if not sumprob:
            prob = [1/len(self.transforms)] * len(self.transforms)
        else:
            prob = [x / sumprob for x in prob]
        return prob

    def throw_dice(self) -> int:
        prob = cumsum(self._make_prob())
        dice = random.random()
        for k in range(len(self.transforms)):
            if dice > 1 - prob[k]:
                return k
        return len(self.transforms) - 1

    def make_final(
        self, x: Tensor, /,
        max_depth: int = float('inf'),
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        t = self.transforms[self.throw_dice()]
        t = t.safe_make_final(x, max_depth-1, args=args)
        with (
            IncludeKeysTransform(t, self.include),
            ExcludeKeysTransform(t, self.exclude)
        ):
            return t

    def __or__(self, other: Transform) -> "SwitchTransform":
        if (isinstance(other, SwitchTransform)
            and not self.prob and not other.prob
        ):
            return SwitchTransform([*self.transforms, *other.transforms])
        else:
            return SwitchTransform([self, other])

    def __ior__(self, other: Transform) -> "SwitchTransform":
        if (isinstance(other, SwitchTransform)
            and not self.prob and not other.prob
        ):
            self.transforms.append(other.transforms)
            return self
        else:
            return SwitchTransform([self, other])

    def __repr__(self) -> str:
        if self.prob:
            prob = self._make_prob()
            s = [f'{p} * {str(t)}' for p, t in zip(prob, self.transforms)]
        else:
            s = [str(t) for t in self.transforms]
        s = ' | '.join(s)
        s = f'({s})'
        return s


class IncludeKeysTransform(Transform):
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

    def __init__(
        self,
        transform: Transform,
        keys: Union[List[str], str],
        union: bool = True
    ) -> None:
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

    def forward(self, *a, **k) -> Returned:
        with self as transform:
            return transform.forward(*a, **k)

    def make_final(
        self, x: Tensor, /,
        max_depth: int = float('inf'),
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        with self as trf:
            final_trf = trf.safe_make_final(x, max_depth, args=args)
            with IncludeKeysTransform(final_trf) as final_final_trf:
                return final_final_trf

    def make_inverse(self) -> Transform:
        with self as trf:
            inv_trf = trf.safe_make_inverse()
            with IncludeKeysTransform(inv_trf) as final_inv_trf:
                return final_inv_trf

    def __enter__(self) -> Transform:
        self.include = self.transform.include
        self.transform.include = self.keys
        if self.union and self.include is not None:
            if self.transform.include is not None:
                self.transform.include = \
                    self.transform.include + self.include
            else:
                self.transform.include = self.include
        return self.transform

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.transform.include = self.include
        delattr(self, 'include')


class ExcludeKeysTransform(Transform):
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

    def __init__(
        self,
        transform: Transform,
        keys: Union[List[str], str],
        union: bool = True
    ) -> None:
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

    def forward(self, *a, **k) -> Returned:
        with self as transform:
            return transform.forward(*a, **k)

    def make_final(
        self, x: Tensor, /,
        max_depth: int = float('inf'),
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        with self as trf:
            final_trf = trf.safe_make_final(x, max_depth, args=args)
            with ExcludeKeysTransform(final_trf) as final_final_trf:
                return final_final_trf

    def make_inverse(self) -> Transform:
        with self as trf:
            inv_trf = trf.make_inverse()
            with ExcludeKeysTransform(inv_trf) as final_inv_trf:
                return final_inv_trf

    def __enter__(self) -> Transform:
        self.exclude = self.transform.exclude
        self.transform.exclude = self.keys
        if self.union and self.exclude:
            if self.transform.exclude is not None:
                self.transform.exclude = \
                    self.transform.exclude + self.exclude
            else:
                self.transform.exclude = self.exclude
        return self.transform

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.transform.exclude = self.exclude
        delattr(self, 'exclude')


class SharedTransform(SharedMixin, Transform):
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

    def __init__(
        self, transform: Transform, mode: Union[str, bool] = UNSET
    ) -> None:
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

    def forward(self, *a, **k) -> Returned:
        with self as transform:
            return transform.forward(*a, **k)

    def __enter__(self) -> Transform:
        self.hasattr = hasattr(self.transform, 'shared')
        self.saved_mode = getattr(self.transform, 'shared', None)
        if self.mode is not UNSET:
            self.transform.shared = self.mode
        return self.transform

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.hasattr:
            self.transform.shared = self.saved_mode
        elif hasattr(self.transform, 'shared'):
            delattr(self.transform, 'shared')
        delattr(self, 'hasattr')
        delattr(self, 'saved_mode')


class ReturningTransform(Transform):
    """
    Context manager for sharing transforms across channels / tensors

    !!! example "Use as a context manager (alias)"
        ```python
        from cornucopia import context as ctx
        with ctx.returns(xform, "channels") as newxform:
            image = newxform(image)
        ```
    """

    def __init__(
        self,
        transform: Transform,
        returns: Union[str, List[str], Dict[str, str], None] = None
    ) -> None:
        super().__init__()
        self.transform = transform
        self.returns = returns

    def __enter__(self) -> Transform:
        self.saved_returns = self.transform.returns
        self.transform.returns = self.returns
        return self.transform

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.transform.returns = self.saved_returns
        delattr(self, 'saved_returns')


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

    def __init__(
        self,
        *mapargs,
        nested: bool = False,
        default: Optional[Transform] = None,
        **mapkwargs
    ) -> None:
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

    def forward(self, *args, **kwargs) -> Returned:

        if self.include is not None or self.exclude:
            def wrap(f: Callable) -> Callable:
                if not f:
                    return f
                def ff(*a, **k):
                    with IncludeKeysTransform(f, self.include), \
                         ExcludeKeysTransform(f, self.exclude):
                        return f(*a, **k)
                return ff
        else:
            def wrap(f: Callable) -> Callable:
                return f

        # If the input is a wrapped `Arguments`, unwrap it and recurse.
        arguments = Arguments(*args, **kwargs)
        if not arguments:
            return None
        args, kwargs = arguments.to_args_kwargs()

        def default(x):
            # Default transform to apply if nothing is mapped.
            if torch.is_tensor(x):
                return self.default(x) if self.default else x
            else:
                return self.forward(x) if self.nested else x

        if args:
            # Apply each transform to its corresponding positional argument
            mapargs = tuple(wrap(f) for f in self.mapargs)
            mapargs += (default,) * max(0, len(args) - len(mapargs))
            args = tuple(f(a) if f else a for f, a in zip(mapargs, args))

        if kwargs:
            # Apply each transform to its corresponding keyword argument
            mapkwargs = {k: wrap(f) for k, f in self.mapkwargs.items()}

            def func(key):
                return mapkwargs.get(key, default) or (lambda x: x)

            kwargs = {key: func(key)(value) for key, value in kwargs.items()}

        # If more than a single input argument, wrap them in `Arguments`
        if kwargs or len(args) > 1:
            return Arguments(*args, **kwargs)
        return args[0]

    def __repr__(self) -> str:
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

    !!! note "`ctx.randomize` is an alias for `RandomizedTransform`"

    !!! example "Gaussian noise with randomized variance"
        Object call
        ```python
        import cornucopia as cc
        hypernoise = cc.RandomizedTransform(cc.GaussianNoise, [cc.Uniform()])
        img = hypernoise(img)
        ```

        Delayed call
        ```python
        import cornucopia as cc
        MyRandomNoise = cc.randomize(cc.GaussianNoise)
        hypernoise = MyRandomNoise(cc.Uniform())
        img = hypernoise(img)
        ```

    """

    class Delayed:
        # Temproary parameter holder for delayed calls
        def __init__(self, transform: Transform, **kwargs) -> None:
            self.transform = transform
            self.kwargs = kwargs

        def __call__(self, *args, **kwargs) -> "RandomizedTransform":
            return RandomizedTransform(
                self.transform, args, kwargs, **self.kwargs)

    def __new__(cls, *args, **kwargs) -> "RandomizedTransform":
        if cls is RandomizedTransform:
            return cls._base_new(*args, **kwargs)
        return super().__new__(cls)

    @classmethod
    def _base_new(
        cls,
        transform: Transform,
        sample: tuple = tuple(),
        ksample: dict = dict(),
        *,
        shared: Union[bool, str] = False,
        **kwargs
    ) -> "RandomizedTransform":
        assert cls is RandomizedTransform
        if not sample and not ksample:
            # If no arguments are passed, it means that the user calls
            # this in "delayed/functional" mode. In that case, we return
            # a callable object that returns the constructed instance
            # using the call-time arguments.
            return cls.Delayed(transform, shared=shared, **kwargs)
        # Otherwise, we're in object mode and we instantiate the
        # randomized object.
        return super().__new__(cls)

    def __init__(
        self,
        transform: Transform,
        sample: tuple = tuple(),
        ksample: dict = dict(),
        *,
        shared: Union[bool, str] = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        transform : callable(...) -> Transform
            A Transform subclass or a function that constructs a Transform.
        sample : [list or dict of] callable
            A collection of functions that generate parameter values provided
            to `transform`. Can be args-like or kwargs-like arguments.
        ksample : dict[callable]
            Must be kwargs-like arguments.

        Other Parameters
        ----------------
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Share random parameters across tensors and/or channels
        """
        super().__init__(shared=shared, **kwargs)
        self.sample = sample
        self.ksample = ksample
        self.subtransform = transform

    def make_final(
        self, x: Tensor, /,
        max_depth: int = inf,
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth, args=args)
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
        xform = xform.safe_make_final(x, max_depth-1, args=args)
        return xform

    def __repr__(self) -> str:
        if type(self) is RandomizedTransform:
            xform = self.subtransform
            if isinstance(xform, type) and issubclass(xform, Transform):
                return f'Randomized{xform.__name__}()'
        return super().__repr__()
