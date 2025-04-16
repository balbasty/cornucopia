"""
These are "smart" inplace operators that only operate inplace if doing
so does not break the computational graph with respect to variables
that require grad.

Note that they should still be used carefully, as the overwritten tensors
may be needed when backpropagating through other operations. For example,
the following code would break the computational graph:

```python
x = torch.randn([])

y = torch.randn([])
y.requires_grad = True

a = x.mul(y)    # mul requires x to be saved to bakpropagate through y
b = x.add_(1)   # but we overwrite x here
c = a + b

c.backward()
```

whereas this would work

```python
x = torch.randn([])

y = torch.randn([])
y.requires_grad = True

a = x.add(y)     # add does not need anythong to backpropagate
b = x.add_(1)    # so we can overwrite x here
c = a + b

c.backward()
```

"""
import math
import torch

_abs = abs
_min = min
_max = max


def _shape_compat(x, y):
    if not torch.is_tensor(y):
        return True
    ndim = x.ndim
    if y.ndim > ndim:
        return False
    return all(dx >= dy for dx, dy in zip(x.shape[-ndim:], y.shape[-ndim:]))


def add_(x, y, **kwargs):
    # d(x+a*y)/dx = 1
    # d(x+a*y)/dy = a
    # d(x+a*y)/da = y
    # -> we can overwrite x
    if not torch.is_tensor(x):
        return x + y * kwargs.get('alpha', 1)
    if not _shape_compat(x, y):
        return x.add(y, **kwargs)
    return x.add_(y, **kwargs)


def add(x, y, **kwargs):
    if not torch.is_tensor(x):
        return x + y * kwargs.get('alpha', 1)
    return x.add(y, **kwargs)


def sub_(x, y, **kwargs):
    # d(x-a*y)/dx = 1
    # d(x-a*y)/dy = -a
    # d(x-a*y)/da = -y
    # -> we can overwrite x
    if not torch.is_tensor(x):
        return x - y * kwargs.get('alpha', 1)
    if not _shape_compat(x, y):
        return x.sub(y, **kwargs)
    return x.sub_(y, **kwargs)


def sub(x, y, **kwargs):
    if not torch.is_tensor(x):
        return x - y * kwargs.get('alpha', 1)
    return x.sub(y, **kwargs)


def mul_(x, y, **kwargs):
    # d(x*y)/dx = y
    # d(x*y)/dy = x
    # -> we can overwrite x if we do not backprop through y
    if not torch.is_tensor(x):
        return x * y
    if not _shape_compat(x, y):
        return x.mul(y, **kwargs)
    return (
        x.mul(y, **kwargs) if getattr(y, 'requires_grad', False) else
        x.mul_(y, **kwargs)
    )


def mul(x, y, **kwargs):
    if not torch.is_tensor(x):
        return x * y
    return x.mul(y, **kwargs)


def div_(x, y, **kwargs):
    # d(x/y)/dx = 1/y
    # d(x/y)/dy = -x/y**2
    # -> we can overwrite x if we do not backprop through y
    if not torch.is_tensor(x):
        return x / y
    if not _shape_compat(x, y):
        return x.div(y, **kwargs)
    return (
        x.div(y, **kwargs) if getattr(y, 'requires_grad', False) else
        x.div_(y, **kwargs)
    )


def div(x, y, **kwargs):
    if not torch.is_tensor(x):
        return x / y
    return x.div(y, **kwargs)


def pow_(x, y, **kwargs):
    # d(x**y)/dx = y * x**(y-1)
    # d(x**y)/dy = (x**y) * log(|x|) * sign(x)**y
    # -> we can overwrite x if we do not backprop through x or y
    if not torch.is_tensor(x):
        return x ** y
    if not _shape_compat(x, y):
        return x.pow(y, **kwargs)
    inplace = not (x.requires_grad or getattr(y, 'requires_grad', False))
    return x.pow(y, **kwargs) if not inplace else x.pow_(y, **kwargs)


def pow(x, y, **kwargs):
    if not torch.is_tensor(x):
        return x ** y
    return x.pow(y, **kwargs)


def square_(x, **kwargs):
    # d(x**2)/dx = 2*x
    # -> we can overwrite x if we do not backprop through x
    if not torch.is_tensor(x):
        return x * x
    return x.square(**kwargs) if x.requires_grad else x.square_(**kwargs)


def square(x, **kwargs):
    if not torch.is_tensor(x):
        return x * x
    return x.square(**kwargs)


def sqrt_(x, **kwargs):
    # d(x**0.5)/dx = 0.5*x
    # -> we can overwrite x if we do not backprop through x
    if not torch.is_tensor(x):
        return x ** 0.5
    return x.sqrt(**kwargs) if x.requires_grad else x.sqrt_(**kwargs)


def sqrt(x, **kwargs):
    # d(x**0.5)/dx = 0.5*x
    # -> we can overwrite x if we do not backprop through x
    if not torch.is_tensor(x):
        return x ** 0.5
    return x.sqrt(**kwargs)


def atan2_(x, y, **kwargs):
    if not torch.is_tensor(x) and not torch.is_tensor(y):
        return math.atan2(x, y)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=y.dtype, device=y.device)
    if not torch.is_tensor(y):
        y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
    if not _shape_compat(x, y):
        return x.atan2(y, **kwargs)
    inplace = not (x.requires_grad or y.requires_grad)
    return x.atan2_(y, **kwargs) if inplace else x.atan2(y, **kwargs)


def atan2(x, y, **kwargs):
    if not torch.is_tensor(x) and not torch.is_tensor(y):
        return math.atan2(x, y)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=y.dtype, device=y.device)
    if not torch.is_tensor(y):
        y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
    return x.atan2(y, **kwargs)


def neg_(x, **kwargs):
    if not torch.is_tensor(x):
        return -x
    return x.neg_(**kwargs)


def neg(x, **kwargs):
    if not torch.is_tensor(x):
        return -x
    return x.neg(**kwargs)


def reciprocal_(x, **kwargs):
    if not torch.is_tensor(x):
        return 1/x
    return (
        x.reciprocal(**kwargs) if x.requires_grad else
        x.reciprocal_(**kwargs)
    )


def reciprocal(x, **kwargs):
    if not torch.is_tensor(x):
        return 1/x
    return x.reciprocal(**kwargs)


def abs_(x, **kwargs):
    if not torch.is_tensor(x):
        return _abs(x)
    return x.abs(**kwargs) if x.requires_grad else x.abs_(**kwargs)


def abs(x, **kwargs):
    if not torch.is_tensor(x):
        return _abs(x)
    return x.abs(**kwargs)


def exp_(x, **kwargs):
    if not torch.is_tensor(x):
        return math.exp(x)
    return x.exp(**kwargs) if x.requires_grad else x.exp_(**kwargs)


def exp(x, **kwargs):
    if not torch.is_tensor(x):
        return math.exp(x)
    return x.exp(**kwargs)


def log_(x, **kwargs):
    if not torch.is_tensor(x):
        return math.log(x)
    return x.log(**kwargs) if x.requires_grad else x.log_(**kwargs)


def log(x, **kwargs):
    if not torch.is_tensor(x):
        return math.log(x)
    return x.log(**kwargs)


def atan_(x, **kwargs):
    if not torch.is_tensor(x):
        return math.atan(x)
    return x.atan(**kwargs) if x.requires_grad else x.atan_(**kwargs)


def atan(x, **kwargs):
    if not torch.is_tensor(x):
        return math.atan(x)
    return x.atan(**kwargs)


def min(x, y):
    if not torch.is_tensor(x) and not torch.is_tensor(y):
        return _min(x, y)
    elif torch.is_tensor(x) and torch.is_tensor(y):
        return torch.minimum(x, y)
    elif torch.is_tensor(x):
        return x.clamp_max(y)
    else:
        assert torch.is_tensor(y)
        return y.clamp_max(x)


def max(x, y):
    if not torch.is_tensor(x) and not torch.is_tensor(y):
        return _max(x, y)
    elif torch.is_tensor(x) and torch.is_tensor(y):
        return torch.maximum(x, y)
    elif torch.is_tensor(x):
        return x.clamp_min(y)
    else:
        assert torch.is_tensor(y)
        return y.clamp_min(x)


def gammaln(x):
    if torch.is_tensor(x):
        return math.lgamma(x)
    return torch.special.gammaln(x)


def gamma(x):
    # !!! Assumes x is positive
    return exp_(gammaln(x))


def floor(x, to=None):
    if torch.is_tensor(x):
        to = {
            int: torch.long,
            float: torch.float,
            complex: torch.complex32
        }.get(to, to)
        return x.floor().to(dtype=to)
    to = {
        torch.int: int,
        torch.long: int,
        torch.float: float,
        torch.double: float,
        torch.complex32: complex,
        torch.complex64: complex,
        None: (lambda x: x)
    }.get(to, to)
    return to(math.floor(x))


def ceil(x, to=None):
    if torch.is_tensor(x):
        to = {
            int: torch.long,
            float: torch.float,
            complex: torch.complex32
        }.get(to, to)
        return x.ceil().to(dtype=to)
    to = {
        torch.int: int,
        torch.long: int,
        torch.float: float,
        torch.double: float,
        torch.complex32: complex,
        torch.complex64: complex,
        None: (lambda x: x)
    }.get(to, to)
    return to(math.ceil(x))
