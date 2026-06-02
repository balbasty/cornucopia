__all__ = [
    'Transform',
    'FinalTransform',
    'NonFinalTransform',
]
# stdlib
import inspect
import random
from abc import ABC
from copy import copy
from math import inf

# dependencies
import torch
import typing_extensions as tx
from torch import nn, Tensor

# internal
from .random import Sampler
from .utils.py import ensure_list, cumsum
from .baseutils import (
    Arguments, Args, Kwargs, ArgsAndKwargs, NoArguments, Returned,
    get_first_element, prepare_output, UNSET, recursive_cat,
)
from . import typing as cct


class Transform(nn.Module, ABC):
    """Base class for all transforms."""

    def __init__(
        self, *,
        returns: tx.Union[str, tx.Sequence[str], tx.Mapping[str, str], None] = None,
        append: tx.Union[bool, str] = False,
        prefix: tx.Union[bool, str] = True,
        include: tx.Optional[cct.IncludeT] = None,
        exclude: tx.Optional[cct.ExcludeT] = None,
        consume: tx.Optional[cct.ConsumeT] = None,
    ):
        """
        Parameters
        ----------
        returns : [list or dict of] str, optional
            Which tensors to return. Can be a nested structure.
            Most transforms accept `'input'` and `'output'` as valid
            returns. The default is `'output'`.
        append : bool | str
            Append the (structure of) returned tensors to the parent
            structure.

            !!! changedin "![v0.5](https://img.shields.io/badge/v0.5-yellow) \
                Can be a string since `v0.5`"
                If it is a `str` and parent is a `dict`, its value will be
                used as a separator between the prefix and the key.
                See `prefix`.
        prefix : bool | str
            If `append` and parent is a `dict`, prefix the returned key
            before inserting it in the output dictionary.

            If `True`, the prefix is the input key.

            !!! changedin "![v0.5](https://img.shields.io/badge/v0.5-yellow) \
                Can be a string since `v0.5`"
        include : [list of] str, optional
            List of keys to which the transform should apply.
            Default: all.
        exclude : [list of] str, optional
            List of keys to which the transform should not apply.
            Default: none.
        consume : [list of] str, optional
            List of keys to remove from the output after applying the
            transform. Default: none.

            !!! addedin "![v0.5](https://img.shields.io/badge/v0.5-green) \
                Added in `v0.5`"
        """
        super().__init__()
        self.returns = returns
        self.append = append
        self.prefix = prefix
        self.include = ensure_list(include) if include is not None else None
        self.exclude = ensure_list(exclude or tuple())
        self.consume = ensure_list(consume or tuple())

    @property
    def is_final(self) -> bool:
        """
        Returns
        -------
        bool
            Whether the transform is final (i.e., deterministic) or not.
        """
        return False

    def get_prm(self) -> dict:
        """Get the parameters of the transform, for use in subtransforms.

        Returns
        -------
        dict
            A dictionary containing the attributes
            `returns`, `append`, `prefix`, `include`, `exclude`, and
            `consume`.

        """
        return dict(
            returns=self.returns,
            append=self.append,
            prefix=self.prefix,
            include=self.include,
            exclude=self.exclude,
            consume=self.consume
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
        *a, **k : [nested list or dict of] tensor
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
        # to each `xform` or `unroll` call, while the latter is
        # recursively unwrapped and processed by `_Forward.__call__`.
        return self._Forward(self, args, **self.get_prm())(x)

    class _Forward:

        def __init__(self, transform, args: Arguments, **prm) -> None:
            self.transform = transform
            self.args = args
            self.prm = prm

        @property
        def include(self) -> tx.Optional[tx.Sequence[str]]:
            return self.prm.get('include')

        @property
        def exclude(self) -> tx.Optional[tx.Sequence[str]]:
            return self.prm.get('exclude')

        @property
        def consume(self) -> tx.Optional[tx.Sequence[str]]:
            return self.prm.get('consume')

        @property
        def append(self) -> bool:
            return self.prm.get('append', False)

        @property
        def prefix(self) -> tx.Union[bool, str]:
            return self.prm.get('prefix', True)

        @property
        def returns(self) -> tx.Optional[tx.Union[
            str, tx.Sequence[str], tx.Mapping[str, str]
        ]]:
            return self.prm.get('returns')

        def __call__(
            self, x: tx.Union[Tensor, tx.Sequence, tx.Mapping, Arguments]
        ) -> tx.Union[Tensor, tx.Sequence, tx.Mapping, Arguments]:
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
            y = self.transform.xform(x, args=self.args)

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

        def _get_valid_keys(self, x: tx.Mapping[str, str]) -> tx.Sequence[str]:
            valid_keys = x.keys()
            if self.include is not None:
                valid_keys = [k for k in valid_keys if k in self.include]
            if self.exclude:
                valid_keys = [k for k in valid_keys if k not in self.exclude]
            return valid_keys

        def _forward_list(
            self, x: list, forward: tx.Optional[tx.Callable] = None
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
            self, x: dict, forward: tx.Optional[tx.Callable] = None
        ) -> tx.Union[dict, list]:
            """Apply forward pass to elements of a dict"""
            forward = forward or self
            valid_keys = self._get_valid_keys(x)

            append = self.append
            if isinstance(append, str):
                sep, append = append, True
            elif append:
                sep = "."

            # Initialise output dictionary with input keys and values
            # that *will not* be transformed so that they are preserved.
            y = {
                key: value
                for key, value in x.items()
                if key not in valid_keys and key not in self.consume
            }

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

                    if append:

                        if isinstance(y, dict) and isinstance(value, dict):
                            # Insert the returned values in the output dictionary,
                            # using a new key, so that the input value is preserved.
                            if prefix:
                                value = {
                                    prefix + sep + child_key: child_value
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

    def xform(
        self, x: Tensor, /,
        args: Arguments = NoArguments(),
    ) -> Returned:
        """Apply the transform to a tensor.

        Non-final transforms do not implement this method in general.

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
        # Wrapper that calls `_xform`, but only passes `args` if the
        # method accepts it, to avoid errors with legacy implementations.
        if 'args' in inspect.signature(self._xform).parameters:
            return self._xform(x, args=args)
        else:
            return self._xform(x)

    def _xform(
        self, x: Tensor, /,
        args: Arguments = NoArguments(),
    ) -> Returned:
        raise NotImplementedError("This transform does not implement `xform`.")

    def final(
        self,
        x: Tensor, /,
        args: Arguments = NoArguments(),
        **kwargs
    ) -> "FinalTransform":
        """
        Generate the final version of the transform.

        Some transforms save the output type of this function in their
        `Final` attribute.

        !!! addedin "![v0.5](https://img.shields.io/badge/v0.5-green) \
            Added `final` method in `v0.5`."
            Before this, one had to use `make_final(x, max_depth=inf)`.

        Parameters
        ----------
        x : tensor
            A single input tensor, with shape `(C, *shape)`.
        args: Arguments, optional
            The original inputs arguments to the transform, in case
            they are needed.

        Returns
        -------
        FinalTransform
            A final version of the transform.
        """
        return self.unroll(x, max_depth=inf, args=args, **kwargs)

    def next(
        self,
        x: Tensor, /,
        args: Arguments = NoArguments(),
        **kwargs
    ) -> "FinalTransform":
        """
        Generate the next version of the transform.

        Some transforms save the output type of this function in their
        `Next` attribute.

        !!! addedin "![v0.5](https://img.shields.io/badge/v0.5-green) \
            Added `next` method in `v0.5`."
            Before this, one had to use `make_final(x, max_depth=1)`.

        Parameters
        ----------
        x : tensor
            A single input tensor, with shape `(C, *shape)`.
        args: Arguments, optional
            The original inputs arguments to the transform, in case
            they are needed.

        Returns
        -------
        Transform
            A more specialized version of the transform.
        """
        return self.unroll(x, max_depth=1, args=args, **kwargs)

    def unroll(
        self, x: Tensor, /,
        max_depth: int = inf,
        args: Arguments = NoArguments(),
        **kwargs
    ) -> "Transform":
        """
        Generate the next (i.e., more final) version(s) of the transform.

        * To completely finalize a transform,
          call `unroll(x, max_depth=inf)` or `final()`.
        * To get the the next version of a transform,
          call `unroll(x, max_depth=1)` or `next()`.

        !!! addedin "![v0.5](https://img.shields.io/badge/v0.5-green) \
            Added `unroll` method in `v0.5`."
            Before this, it was named `make_final`.

        Parameters
        ----------
        x : tensor
            A single input tensor, with shape `(C, *shape)`.
        max_depth : int | {inf}
            Maximum depth to apply `unroll` recursively.
            If not `inf`, the resulting transform may not be fully final.
            Default: no limit.
        args: Arguments, optional
            The original inputs arguments to the transform, in case
            they are needed.

        Returns
        -------
        Transform
            A more specialized version of the transform.
        """
        if max_depth <= 0:
            # This is always valid, so let's catch it
            return self
        # Wrapper that calls `_unroll`, but only passes `args` if the
        # method accepts it.
        if 'args' in inspect.signature(self._unroll).parameters:
            return self._unroll(x, max_depth, args=args, **kwargs)
        else:
            return self._unroll(x, max_depth, **kwargs)

    def make_final(
        self, x: Tensor, /,
        max_depth: int = inf,
        args: Arguments = NoArguments(),
        **kwargs
    ) -> "Transform":
        # Deprecated, but keep it for backward compatibility
        return self.unroll(x, max_depth=max_depth, args=args, **kwargs)

    def _unroll(
        self, x: Tensor, /,
        max_depth: int = inf,
        args: Arguments = NoArguments(),
    ) -> "Transform":
        if self.is_final or max_depth == 0:
            return self
        raise NotImplementedError("This transform does not implement `unroll`")

    def inverse(self, *a, **k) -> "FinalTransform":
        """Apply the inverse transform recursively

        Parameters
        ----------
        *a, **k : [nested list or dict of] tensor
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

    def _unroll(
        self, x: Tensor, /,
        max_depth: int = inf,
        args: Arguments = NoArguments(),
    ) -> "FinalTransform":
        return self


class _SharedMixin:
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

    def _xform(
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
        transformation = self.next(template, args=args)
        if transformation is self:
            # Avoid infinite recursion. This should not happen.
            raise ValueError(
                f"The transform is not final, but calling `next` "
                f"returned itself. Transform: {self}"
            )
        # Apply the final transform to all channels
        return transformation.xform(x, args=args)

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
            transform = self.next(first_tensor, args=args)
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
            self.unroll(x[i:i+1], max_depth, args=args, **kwargs)
            for i in range(len(x))
        ], **prm).unroll(x, max_depth-1)


class NonFinalTransform(_SharedMixin, Transform):
    """
    Transforms whose parameters depend on features of the input
    transform (shape, dtype, etc).

    Non-final transforms implement `unroll`, and do not implement
    `xform`. Their aim is to generate a more-specialized transform
    at call time.
    """
    def __init__(self, *, shared: bool = False, **kwargs) -> None:
        """
        Parameters
        ----------
        shared : {'channels', 'tensors', 'channels+tensor', ''} | bool

            - `'channel'`: the same transform is applied to all channels
                in a tensor, but different transforms are used in different
                tensors.
            - `'tensors'`: the same transform is applied to all tensors,
                but with a different transform for each channel.
            - `'channels+tensors'` or `True`: the same transform is applied
                to all channels of all tensors.
            - `''` or `False`: A different transform is applied to each
                channel and each tensor.
        """
        super().__init__(**kwargs)
        self.shared = self._prepare_shared(shared)


class SpecialTransform(Transform):
    """Base class for transforms that act on other transforms.

    Such transforms cannot be easily classified as "final" or "non-final",
    because this characeteristic depends on the transforms that they embed.

    They all implement `unroll`, but some may also implement a
    "fast-track" `xform` that is applied in simple cases (e.g., when
    the transform is not shared across tensors) for efficiency.

    !!! addedin "![v0.5](https://img.shields.io/badge/v0.5-green) \
        Added `SpecialTransform` class in `v0.5`."
        Before this, special transforms inherited directly from `Transform`.
    """
    ...


class IdentityTransform(FinalTransform):
    """Identity transform"""

    def _xform(
        self, x: Tensor, /, args: Arguments = NoArguments()
    ) -> Returned:
        return prepare_output(dict(input=x, output=x), self.returns)

    def make_inverse(self) -> "IdentityTransform":
        return self


class SequentialTransform(_SharedMixin, SpecialTransform):
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

    def __init__(self, transforms: tx.Sequence[Transform], **kwargs) -> None:
        """
        Parameters
        ----------
        transforms : list[Transform]
            A list of transforms to apply sequentially.

        Other Parameters
        ------------------
        shared
            See [`NonFinalTransform`][cornucopia.base.NonFinalTransform]
            for details.
        returns, append, prefix, include, exclude, consume
            See [`Transform`][cornucopia.base.Transform] for details.
        """
        shared = kwargs.pop('shared', False)
        super().__init__(**kwargs)
        self.shared = self._prepare_shared(shared)
        self.transforms = transforms

    def _unroll(
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
            t = t.unroll(x, max_depth=max_depth-1, args=args)
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
        # behavior from `SharedMixin`, which is to call `unroll` on
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
            # NOTE:
            #   I do not propagate `returns`, as I don't think it makes
            #   sense for sequences.
            with (
                IncludeKeysTransform(trf, self.include),
                ExcludeKeysTransform(trf, self.exclude),
                ConsumeKeysTransform(trf, self.consume)
            ):
                x = trf(x)
        return x

    def _xform(
        self, x: Tensor, /, args: Arguments = NoArguments()
    ) -> Returned:
        # This should only be called when a Layer's `unroll` returns
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

    def __iter__(self) -> tx.Iterator[Transform]:
        for t in self.transforms:
            yield t

    def __getitem__(self, item: tx.Union[int, slice]) -> Transform:
        if isinstance(item, slice):
            return SequentialTransform(self.transforms[item])
        else:
            return self.transforms[item]

    def __repr__(self) -> str:
        return f'{type(self).__name__}({repr(self.transforms)})'


class PerChannelTransform(SpecialTransform):
    """Apply a different transform to each channel"""

    def __init__(self, transforms: tx.Sequence[Transform], **kwargs) -> None:
        """
        Parameters
        ----------
        transforms : list[Transform]
            A list of transforms to apply to each channel.
        """
        super().__init__(**kwargs)
        self.transforms = transforms

    def _unroll(
        self, x: Tensor, /,
        max_depth: int = inf,
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        trf = []
        for i, t in enumerate(self.transforms):
            if (
                self.include is not None or
                self.exclude or self.consume or self.returns
            ):
                # NOTE
                #   We cannot use context managers because they exit on
                #   return. Instead, we make a shallow copy of the
                #   transform and change its options. It is not an issue
                #   in most cases, as `unroll` often creates a new
                #   transform, but can be one when `max_depth < 2`.
                t = copy(t)
                t.exclude = IncludeKeysTransform.combine(self.include, t.include)
                t.include = ExcludeKeysTransform.combine(self.exclude, t.exclude)
                t.consume = ConsumeKeysTransform.combine(self.consume, t.consume)
                if self.returns:
                    t.returns = self.returns
            t = t.unroll(x[i:i+1], max_depth-1, args=args)
            trf.append(t)
        prm = dict(self.get_prm())
        prm.pop('shared', None)
        trf = PerChannelTransform(trf, **prm)
        return trf

    def _xform(
        self, x: Tensor, /, args: Arguments = NoArguments()
    ) -> Returned:
        results = []
        for i, t in enumerate(self.transforms):
            with (
                ReturningTransform(t, self.returns),
                IncludeKeysTransform(t, self.include),
                ExcludeKeysTransform(t, self.exclude),
                ConsumeKeysTransform(t, self.consume)
            ):
                results.append(t(x[i:i+1]))
        return Returned(recursive_cat(results))

    @property
    def is_final(self) -> bool:
        return all(t.is_final for t in self.transforms)


class MaybeTransform(_SharedMixin, SpecialTransform):
    """Randomly apply a transform

    !!! note "[`ctx.maybe`][cornucopia.ctx.maybe] is an alias for [`MaybeTransform`][cornucopia.special.MaybeTransform]"

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
        shared : {'channels', 'tensors', 'channels+tensor', ''} | bool
            Roll the dice once for all input tensors
        """
        super().__init__(**kwargs)
        self.shared = self._prepare_shared(shared)
        self.subtransform = transform
        self.prob = prob

    def throw_dice(self) -> bool:
        return random.random() > 1 - self.prob

    def _unroll(
        self, x: Tensor, /,
        max_depth: int = float('inf'),
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        if self.throw_dice():
            trf = self.subtransform
            if self.include is not None or self.exclude or self.consume:
                # NOTE
                # * I do not use context managers as they exit on return.
                #   Context managers would work in most cases, as
                #   `unroll` often creates a new transform, but it
                #   can be a problem when `max_depth<2`. Better safe
                #   than sorry,
                # * I do not propagate `returns`. I think it should be
                #   dealt with by the subtransform.
                trf = copy(trf)
                trf.include = IncludeKeysTransform._combine(self.include, trf.include)
                trf.exclude = ExcludeKeysTransform._combine(self.exclude, trf.exclude)
                trf.consume = ConsumeKeysTransform._combine(self.consume, trf.consume)
            return trf.unroll(x, max_depth-1, args=args)
        else:
            return IdentityTransform(consume=self.consume)

    def __repr__(self) -> str:
        s = f'{repr(self.subtransform)}?'
        if self.prob != 0.5:
            s += f'[{self.prob}]'
        return s


class SwitchTransform(_SharedMixin, SpecialTransform):
    """Randomly choose a transform to apply

    !!! note "[`ctx.switch`][cornucopia.ctx.switch] is an alias for [`SwitchTransform`][cornucopia.special.SwitchTransform]"

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
        transforms: tx.Sequence[Transform],
        prob: cct.ScalarOrSequence[float] = 0,
        *,
        shared: cct.SharedT = True,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        transforms : list[Transform]
            A list of transforms to sample from
        prob : list[float]
            Probability of applying each transform
        shared : {'channels', 'tensors', 'channels+tensor', ''} | bool
            Roll the dice once for all input tensors
        """
        super().__init__(**kwargs)
        if isinstance(transforms, dict):
            if prob:
                raise ValueError(
                    "When `transforms` is a dict, `prob` should not be provided."
                )
            prob = list(transforms.values())
            transforms = list(transforms.keys())
        self.shared = self._prepare_shared(shared)
        self.transforms = list(transforms)
        self.prob = prob or []

    def _make_prob(self) -> tx.Sequence[float]:
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

    def _unroll(
        self, x: Tensor, /,
        max_depth: int = float('inf'),
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        t = self.transforms[self.throw_dice()]
        t = t.unroll(x, max_depth-1, args=args)
        if self.include is not None or self.exclude or self.consume:
            # NOTE
            #   We cannot use the context manager because it exits on
            #   return. Instead, we make a shallow copy of the transform
            #   and change its options.
            t = copy(t)
            t.include = IncludeKeysTransform._combine(self.include, t.include)
            t.exclude = ExcludeKeysTransform._combine(self.exclude, t.exclude)
            t.consume = ConsumeKeysTransform._combine(self.consume, t.consume)
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


class IncludeKeysTransform(SpecialTransform):
    """
    Context manager for keys to include

    !!! note "[`ctx.include`][cornucopia.ctx.include] is an alias for [`IncludeKeysTransform`][cornucopia.special.IncludeKeysTransform]"

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
        from cornucopia import ctx
        with ctx.include(xform, "image") as newxform:
            image, label = newxform(image=image, label=label)
        ```
    """

    def __init__(
        self,
        transform: Transform,
        keys: cct.IncludeT,
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
        if keys is not None:
            keys = ensure_list(keys)
        self.transform = transform
        self.keys = keys
        self.union = union

    def forward(self, *a, **k) -> Returned:
        with self as transform:
            return transform.forward(*a, **k)

    def _unroll(
        self, x: Tensor, /,
        max_depth: int = float('inf'),
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        with self as trf:
            final_trf = trf.unroll(x, max_depth, args=args)
            with IncludeKeysTransform(final_trf) as final_final_trf:
                return final_final_trf

    def make_inverse(self) -> Transform:
        with self as trf:
            inv_trf = trf.make_inverse()
            with IncludeKeysTransform(inv_trf) as final_inv_trf:
                return final_inv_trf

    @classmethod
    def _combine(self, *includes, union: bool = True) -> tx.Sequence[str]:
        new_include, *includes = includes
        if union:
            for include in includes:
                if include is not None:
                    if new_include is None:
                        new_include = []
                    new_include.extend(include)
        if new_include is not None:
            new_include = list(set(new_include))
        return new_include

    def __enter__(self) -> Transform:
        old_include = self.transform.include
        new_include = self.keys
        new_include = self._combine(new_include, old_include, union=self.union)
        self.transform.include, self.include = new_include, old_include
        return self.transform

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.transform.include = self.include
        delattr(self, 'include')


class ExcludeKeysTransform(SpecialTransform):
    """
    Context manager for keys to exclude.
    Can also be used as a transform.

    !!! note "[`ctx.exclude`][cornucopia.ctx.exclude] is an alias for [`ExcludeKeysTransform`][cornucopia.special.ExcludeKeysTransform]"

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
        from cornucopia import ctx
        with ctx.exclude(xform, "image") as newxform:
            image, label = newxform(image=image, label=label)
        ```
    """

    def __init__(
        self,
        transform: Transform,
        keys: cct.ExcludeT,
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
        if keys is not None:
            keys = ensure_list(keys)
        self.transform = transform
        self.keys = keys
        self.union = union

    def forward(self, *a, **k) -> Returned:
        with self as transform:
            return transform.forward(*a, **k)

    def _unroll(
        self, x: Tensor, /,
        max_depth: int = float('inf'),
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        with self as trf:
            final_trf = trf.unroll(x, max_depth, args=args)
            with ExcludeKeysTransform(final_trf) as final_final_trf:
                return final_final_trf

    def make_inverse(self) -> Transform:
        with self as trf:
            inv_trf = trf.make_inverse()
            with ExcludeKeysTransform(inv_trf) as final_inv_trf:
                return final_inv_trf

    @classmethod
    def _combine(self, *excludes, union: bool = True) -> tx.Sequence[str]:
        new_exclude, *excludes = excludes
        if union:
            for exclude in excludes:
                if exclude is not None:
                    if new_exclude is None:
                        new_exclude = []
                    new_exclude.extend(exclude)
        if new_exclude is not None:
            new_exclude = list(set(new_exclude))
        return new_exclude

    def __enter__(self) -> Transform:
        old_exclude = self.transform.exclude
        new_exclude = self.keys
        new_exclude = self._combine(new_exclude, old_exclude, union=self.union)
        self.transform.exclude, self.exclude = new_exclude, old_exclude
        return self.transform

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.transform.exclude = self.exclude
        delattr(self, 'exclude')


class ConsumeKeysTransform(SpecialTransform):
    """
    Context manager for keys to consume.
    Can also be used as a transform.

    !!! note "[`ctx.consume`][cornucopia.ctx.consume] is an alias for [`ConsumeKeysTransform`][cornucopia.special.ConsumeKeysTransform]"

    !!! example "Use as a transform"
        ```python
        from cornucopia import ConsumeKeysTransform
        newxform = ConsumeKeysTransform(xform, "image)
        label = newxform(image=image, label=label)
        ```

    !!! example "Use as a context manager `with as`"
        ```python
        from cornucopia import ConsumeKeysTransform
        with ConsumeKeysTransform(xform, "image") as newxform:
            label = newxform(image=image, label=label)
        ```

    !!! example "Use as a context manager `with`"
        ```python
        from cornucopia import ConsumeKeysTransform
        with ConsumeKeysTransform(xform, "image"):
            label = xform(image=image, label=label)
        ```

    !!! example "Use as a context manager (alias)"
        ```python
        from cornucopia import ctx
        with ctx.consume(xform, "image") as newxform:
            label = newxform(image=image, label=label)
        ```

    !!! addedin "![v0.5](https://img.shields.io/badge/v0.5-green) \
        Added in `v0.5`"
    """

    def __init__(
        self,
        transform: Transform,
        keys: cct.ConsumeT,
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
            Consume the union of what was already consumed and `keys`
        """
        super().__init__()
        if keys is not None:
            keys = ensure_list(keys)
        self.transform = transform
        self.keys = keys
        self.union = union

    def forward(self, *a, **k) -> Returned:
        with self as transform:
            return transform.forward(*a, **k)

    def _unroll(
        self, x: Tensor, /,
        max_depth: int = float('inf'),
        args: Arguments = NoArguments()
    ) -> Transform:
        if max_depth == 0:
            return self
        with self as trf:
            final_trf = trf.unroll(x, max_depth, args=args)
            with ConsumeKeysTransform(final_trf) as final_final_trf:
                return final_final_trf

    def make_inverse(self) -> Transform:
        with self as trf:
            inv_trf = trf.make_inverse()
            with ConsumeKeysTransform(inv_trf) as final_inv_trf:
                return final_inv_trf

    @classmethod
    def _combine(self, *consumes, union: bool = True) -> tx.Sequence[str]:
        new_consume, *consumes = consumes
        if union:
            for consume in consumes:
                if consume is not None:
                    if new_consume is None:
                        new_consume = []
                    new_consume.extend(consume)
        if new_consume is not None:
            new_consume = list(set(new_consume))
        return new_consume

    def __enter__(self) -> Transform:
        old_consume = self.transform.consume
        new_consume = self.keys
        new_consume = self._combine(new_consume, old_consume, union=self.union)
        self.transform.consume, self.consume = new_consume, old_consume
        return self.transform

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.transform.consume = self.consume
        delattr(self, 'consume')


class SharedTransform(_SharedMixin, SpecialTransform):
    """
    Context manager for sharing transforms across channels / tensors.
    Can also be used as a transform.

    !!! note "[`ctx.shared`][cornucopia.ctx.shared] is an alias for [`SharedTransform`][cornucopia.special.SharedTransform]"

    !!! example "Use as a context manager (alias)"
        ```python
        from cornucopia import ctx
        with ctx.shared(xform, "channels") as newxform:
            image = newxform(image)
        ```
    """

    def __init__(
        self, transform: Transform, mode: cct.SharedT = UNSET
    ) -> None:
        """
        Parameters
        ----------
        transform : Transform
            Transform to apply
        mode : {'channels', 'tensors', 'channels+tensor', ''} | bool

            - `'channel'`: the same transform is applied to all channels
                in a tensor, but different transforms are used in different
                tensors.
            - `'tensors'`: the same transform is applied to all tensors,
                but with a different transform for each channel.
            - `'channels+tensors'` or `True`: the same transform is applied
                to all channels of all tensors.
            - `''` or `False`: A different transform is applied to each
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


class ReturningTransform(SpecialTransform):
    """
    Context manager for sharing transforms across channels / tensors

    !!! note "[`ctx.returns`][cornucopia.ctx.returns] is an alias for [`ReturningTransform`][cornucopia.special.ReturningTransform]"

    !!! example "Use as a context manager (alias)"
        ```python
        from cornucopia import ctx
        with ctx.returns(xform, "channels") as newxform:
            image = newxform(image)
        ```
    """

    def __init__(
        self,
        transform: Transform,
        returns: tx.Optional[cct.ReturnsT] = None
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


class MappedTransform(SpecialTransform):
    """
    Transforms that are applied to specific positional or arguments

    !!! note "[`ctx.map`][cornucopia.ctx.map] is an alias for [`MappedTransform`][cornucopia.special.MappedTransform]"

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
        default: tx.Optional[Transform] = None,
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

        if self.include is not None or self.exclude or self.consume:
            def wrap(f: tx.Callable) -> tx.Callable:
                if not f:
                    return f
                def ff(*a, **k):
                    # NOTE
                    #   I do not propagate `returns`. I think it should
                    #   be dealt with by the subtransforms.
                    with (
                        IncludeKeysTransform(f, self.include),
                        ExcludeKeysTransform(f, self.exclude),
                        ConsumeKeysTransform(f, self.consume)
                    ):
                        return f(*a, **k)
                return ff
        else:
            def wrap(f: tx.Callable) -> tx.Callable:
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


class RandomizedTransform(SpecialTransform, NonFinalTransform):
    """
    Transform generated by randomizing some parameters of another transform.

    !!! note "[`ctx.randomize`][cornucopia.ctx.randomize] is an alias for [`RandomizedTransform`][cornucopia.special.RandomizedTransform]"

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
        shared: tx.Union[bool, str] = False,
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
        shared: tx.Union[bool, str] = False,
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
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Share random parameters across tensors and/or channels
        """
        super().__init__(shared=shared, **kwargs)
        self.sample = sample
        self.ksample = ksample
        self.subtransform = transform

    def _unroll(
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
            args += [
                f() if isinstance(f, Sampler) else f
                for f in self.sample
            ]

        elif hasattr(self.sample, 'items'):
            kwargs.update({
                k: f() if isinstance(f, Sampler) else f
                for k, f in self.sample.items()
            })

        else:
            args += [
                f() if isinstance(f, Sampler) else f
                for f in (self.sample,)
            ]

        if self.ksample:
            kwargs.update({
                k: f() if isinstance(f, Sampler) else f
                for k, f in self.ksample.items()
            })

        # Propagate general options (include, exclude),
        # unless they are already set by sample/ksample.
        for key, value in self.get_prm().items():
            kwargs.setdefault(key, value)

        # Build transform with fixed parameters, and recurse.
        xform = self.subtransform(*args, **kwargs)
        xform = xform.unroll(x, max_depth-1, args=args)
        return xform

    def __repr__(self) -> str:
        if type(self) is RandomizedTransform:
            xform = self.subtransform
            if isinstance(xform, type) and issubclass(xform, Transform):
                return f'Randomized{xform.__name__}()'
        return super().__repr__()
