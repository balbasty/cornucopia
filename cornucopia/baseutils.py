import torch
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
        if torch.is_tensor(x) or isinstance(x, types):
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


def return_requires(returns):
    """Return all requires fields in a flat structure"""
    if returns is None:
        return ['output']
    returns = flatstruct(returns)
    if isinstance(returns, dict):
        return list(returns.values())
    elif isinstance(returns, (list, tuple)):
        return list(returns)
    else:
        return [returns]


def returns_find(flag, returned, returns):
    """Find tensor corresponding to flag in returned structure"""
    if returns is None:
        if flag == 'output':
            return returned
        else:
            return None
    if isinstance(returns, dict):
        return returned.get(flag, None)
    elif isinstance(returns, (list, tuple)):
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
            vmin = x.min(dim=list(range(1, x.ndim)))
            vmax = x.max(dim=list(range(1, x.ndim)))
            vmean = x.mean(dim=list(range(1, x.ndim)))
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


class unset:
    pass
