# stdlib
from collections.abc import Mapping, Sequence

# dependencies
import torch

# internal
from .utils.indexing import guess_shape
from .utils.py import ensure_list


def get_first_element(x, include=None, exclude=None, types=None):
    """Return the fist element (tensor or string) in the nested structure"""
    types = ensure_list(types or [])

    def _recursive(x):
        if hasattr(x, 'items'):
            for k, v in x.items():
                if include and k not in include:
                    continue
                if exclude and k in exclude:
                    continue
                v, ok = _recursive(v)
                if ok:
                    return v, True
            return None, False
        if isinstance(x, (list, tuple)):
            for v in x:
                v, ok = _recursive(v)
                if ok:
                    return v, True
            return None, False
        if torch.is_tensor(x) or (types and isinstance(x, types)):
            return x, True
        return x, False

    return _recursive(x)[0]


def recursive_cat(x, **kwargs):
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
            pass
        else:
            results = results.get('output', None)
    elif isinstance(returns, dict):
        results = type(returns)(
            **{key: results.get(target, None)
               for key, target in returns.items()})
    elif isinstance(returns, (list, tuple)):
        results = type(returns)(
            [results.get(target, None) for target in returns])
    else:
        results = results.get(returns, None)
    return Returned(results)


def return_requires(returns) -> set[str]:
    """Return all requires fields in a flat structure"""
    if returns is None:
        return {'output'}
    returns = flatstruct(returns)
    if isinstance(returns, dict):
        return set(returns.values())
    elif isinstance(returns, (list, tuple, set)):
        return set(returns)
    else:
        return {returns}


def returns_find(flag, returned, returns):
    """Find tensor corresponding to flag in returned structure"""
    if returns is None:
        if flag == 'output':
            return returned
        else:
            return None
    if isinstance(returns, dict):
        return returned.get(flag, None)
    elif isinstance(returns, (list, tuple, set)):
        if flag in returns:
            return returned[returns.index(flag)]
        else:
            return None
    else:
        assert isinstance(returns, str)
        if returns == flag:
            return returned
        else:
            return None


def returns_update(value, flag, returned, returns):
    """Find tensor corresponding to flag in returned structure"""
    if returns is None:
        if flag == 'output':
            return value
        else:
            return None
    if isinstance(returns, dict):
        if flag in returns:
            returned[flag] = value
        return returned
    elif isinstance(returns, (list, tuple)):
        if flag in returns:
            returned[returns.index(flag)] = value
        return returned
    else:
        assert isinstance(returns, str)
        if returns == flag:
            return value
        else:
            return None


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


class Arguments:
    """Base class for wrapping arguments of a transform call.

    No instance of this class are ever created. Instead, it returns
    instances of one of its concrete subclasses:

    - `NoArguments`: when no arguments are passed
    - `Arg`: when a single argument is passed
    - `Args`: when only positional arguments are passed
    - `Kwargs`: when only keyword arguments are passed
    - `ArgsAndKwargs`: when both positional and keyword arguments are passed
    """

    def __new__(cls, *args, **kwargs):
        if cls is Arguments:
            if not kwargs and len(args) == 1:
                arg, = args
                if not isinstance(arg, Arguments):
                    arg = Arg(arg)
                return arg
            if args and kwargs:
                return ArgsAndKwargs(*args, **kwargs)
            elif args:
                return Args(*args)
            elif kwargs:
                return Kwargs(**kwargs)
            else:
                return NoArguments()
        return super().__new__(cls)

    def __str__(self):
        return repr(self)

class NoArguments(Arguments):
    """Wrapper when no arguments where passed."""

    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def __iter__(self):
        return

    def keys(self):
        return set()

    def values(self):
        return set()

    def items(self):
        return set()

    def unwrap(self):
        return None

    def to_args_kwargs(self):
        return (), {}

    def __repr__(self):
        return "NoArguments()"


class Arg(Arguments):
    """Single argument"""

    def __new__(cls, arg):
        if cls is Arg:
            if isinstance(arg, Arguments):
                return arg
            if isinstance(arg, Mapping):
                return DictArg(arg)
            if isinstance(arg, Sequence) and not isinstance(arg, str):
                return TupleArg(arg)
        return super().__new__(cls)

    def __init__(self, arg):
        if arg is self:
            # This can happen when calling `Arg` on an `Arg`.
            # Our `__new__` should return the input object instead of
            # creating a new one, but because this object _is_ an instance
            # of `Arg`, `__init__` is still called.
            return
        self.arg = arg

    def unwrap(self):
        return self.arg

    def to_args_kwargs(self):
        return (self.arg,), {}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.arg!r})"


class DictArg(Arg, Mapping):
    """Single argument that is a mapping."""

    def __len__(self):
        return len(self.arg)

    def __iter__(self):
        return iter(self.arg)

    def __getitem__(self, key):
        return self.arg[key]


class TupleArg(Arg, Sequence):
    """Single argument that is a sequence."""

    def __len__(self):
        return len(self.arg)

    def __getitem__(self, index):
        return self.arg[index]


class Args(tuple, Arguments):
    """Tuple of arguments: `*args`"""

    def __init__(self, *args):
        super().__init__(args)

    def unwrap(self):
        return tuple(self)

    def to_args_kwargs(self):
        return tuple(self), {}

    def __repr__(self):
        args = ", ".join(repr(a) for a in self)
        return f"{self.__class__.__name__}({args})"

    class Keys:
        def __init__(self, parent):
            self.parent = parent

        def __iter__(self):
            for i in range(len(self.parent)):
                yield i

        def __repr__(self):
            return f"Keys({list(self)!r})"

    class Values:
        def __init__(self, parent):
            self.parent = parent

        def __iter__(self):
            for a in self.parent:
                yield a

        def __repr__(self):
            return f"Values({list(self)!r})"

    class Items:
        def __init__(self, parent):
            self.parent = parent

        def __iter__(self):
            for i, a in enumerate(self.parent):
                yield (i, a)

        def __repr__(self):
            return f"Items({list(self)!r})"


class Kwargs(dict, Arguments):
    """Dict-like, except that unzipping works on values instead of keys"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def unwrap(self):
        return dict(self)

    def to_args_kwargs(self):
        return (), dict(self)

    def __repr__(self):
        kwargs = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"{self.__class__.__name__}({kwargs})"

    def __iter__(self):
        # Iterate across values instead of keys.
        # This allows `Kwargs` to act like a tuple of values.
        for v in self.values():
            yield v


class ArgsAndKwargs(Arguments):
    """Iterator across both args and kwargs"""

    def __init__(self, *args, **kwargs):
        self.args = Args(*args)
        self.kwargs = Kwargs(**kwargs)

    def to_args_kwargs(self):
        return tuple(self.args), dict(self.kwargs)

    def unwrap(self):
        return self.to_args_kwargs()

    def __repr__(self):
        args = ", ".join(repr(a) for a in self.args)
        kwargs = ", ".join(f"{k}={v!r}" for k, v in self.items())
        args_kwargs = ", ".join([args, kwargs])
        return f"{self.__class__.__name__}({args_kwargs})"

    def __iter__(self):
        # Iterate across values of both args and kwargs, in that order.
        # This allows `ArgsAndKwargs` to act like the concatenation of
        # an `Args` and a `Kwargs`.
        for a in self.args:
            yield a
        for v in self.kwargs:
            yield v

    def __len__(self):
        return len(self.args) + len(self.kwargs)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.args[index]
        else:
            return self.kwargs[index]

    def keys(self):
        return self.Keys(self)

    def values(self):
        return self.Values(self)

    def items(self):
        return self.Items(self)

    class Keys:
        def __init__(self, parent):
            self.parent = parent

        def __iter__(self):
            for i in range(len(self.parent.args)):
                yield i
            for k in self.parent.kwargs.keys():
                yield k

        def __repr__(self):
            return f"Keys({list(self)!r})"

    class Values:
        def __init__(self, parent):
            self.parent = parent

        def __iter__(self):
            for a in self.parent.args:
                yield a
            for v in self.parent.kwargs.values():
                yield v

        def __repr__(self):
            return f"Values({list(self)!r})"

    class Items:
        def __init__(self, parent):
            self.parent = parent

        def __iter__(self):
            for i, a in enumerate(self.parent.args):
                yield (i, a)
            for k, v in self.parent.kwargs.items():
                yield (k, v)

        def __repr__(self):
            return f"Items({list(self)!r})"


class Returned:
    """Internal object used to mark that this is an object returned
    by `transform_tensor` at the most nested level"""
    def __init__(self, obj):
        self.obj = obj


class VirtualTensor:
    """Virtual tensor used to recursively compute final transforms"""

    def __init__(self, shape, dtype=None, device=None,
                 vmin=None, vmax=None, vmean=None):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.vmin = vmin
        self.vmax = vmax
        self.vmean = vmean

    @classmethod
    def from_tensor(cls, x, compute_stats=False):
        if compute_stats:
            vmin = x.reshape([len(x), -1]).min(dim=-1).values
            vmax = x.reshape([len(x), -1]).max(dim=-1).values
            vmean = x.float().mean(dim=list(range(1, x.ndim)))
        else:
            vmin = vmax = vmean = None
        return VirtualTensor(x.shape, dtype=x.dtype, device=x.device,
                             vmin=vmin, vmax=vmax, vmean=vmean)

    @classmethod
    def from_virtual(cls, x):
        return VirtualTensor(x.shape, dtype=x.dtype, device=x.device,
                             vmin=x.vmin, vmax=x.vmax, vmean=x.vmean)

    @classmethod
    def from_any(cls, x, compute_stats=False):
        if torch.is_tensor(x):
            return cls.from_tensor(x, compute_stats)
        elif isinstance(x, VirtualTensor):
            return cls.from_virtual(x)
        elif isinstance(x, (list, tuple, torch.Size)):
            return VirtualTensor(x)
        else:
            raise TypeError(f"Don't know how to convert type {type(x)} "
                            f"to VirtualTensor")

    def __getitem__(self, index):
        outshape = guess_shape(index, self.shape)
        return VirtualTensor(
            outshape, self.dtype, self.device,
            self.vmin, self.vmax, self.vmean
        )


UNSET = object()
