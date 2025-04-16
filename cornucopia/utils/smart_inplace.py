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


def add_(x, y, **kwargs):
    # d(x+a*y)/dx = 1
    # d(x+a*y)/dy = a
    # d(x+a*y)/da = y
    # -> we can overwrite x
    if not torch.is_tensor(x):
        return x + y * kwargs.get('alpha', 1)
    return x.add_(y, **kwargs)


def sub_(x, y, **kwargs):
    # d(x-a*y)/dx = 1
    # d(x-a*y)/dy = -a
    # d(x-a*y)/da = -y
    # -> we can overwrite x
    if not torch.is_tensor(x):
        return x - y * kwargs.get('alpha', 1)
    return x.sub_(y, **kwargs)


def mul_(x, y, **kwargs):
    # d(x*y)/dx = y
    # d(x*y)/dy = x
    # -> we can overwrite x if we do not backprop through y
    if not torch.is_tensor(x):
        return x * y
    return (
        x.mul(y, **kwargs) if getattr(y, 'requires_grad', False) else
        x.mul_(y, **kwargs)
    )


def div_(x, y, **kwargs):
    # d(x/y)/dx = 1/y
    # d(x/y)/dy = -x/y**2
    # -> we can overwrite x if we do not backprop through y
    if not torch.is_tensor(x):
        return x / y
    return (
        x.div(y, **kwargs) if getattr(y, 'requires_grad', False) else
        x.div_(y, **kwargs)
    )


def pow_(x, y, **kwargs):
    # d(x**y)/dx = y * x**(y-1)
    # d(x**y)/dy = (x**y) * log(|x|) * sign(x)**y
    # -> we can overwrite x if we do not backprop through x or y
    if not torch.is_tensor(x):
        return x ** y
    inplace = not (x.requires_grad or getattr(y, 'requires_grad', False))
    return x.pow(y, **kwargs) if not inplace else x.pow_(y, **kwargs)


def square_(x, **kwargs):
    # d(x**2)/dx = 2*x
    # -> we can overwrite x if we do not backprop through x
    if not torch.is_tensor(x):
        return x * x
    return x.square(**kwargs) if x.requires_grad else x.square_(**kwargs)


def sqrt_(x, **kwargs):
    # d(x**0.5)/dx = 0.5*x
    # -> we can overwrite x if we do not backprop through x
    if not torch.is_tensor(x):
        return x ** 0.5
    return x.sqrt(**kwargs) if x.requires_grad else x.sqrt_(**kwargs)


def atan2_(x, y, **kwargs):
    if not torch.is_tensor(x) and not torch.is_tensor(y):
        return math.atan2(x, y)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=y.dtype, device=y.device)
    if not torch.is_tensor(y):
        y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
    inplace = not (x.requires_grad or y.requires_grad)
    return x.atan2(y, **kwargs) if not inplace else x.atan2_(y, **kwargs)


def neg_(x, **kwargs):
    if not torch.is_tensor(x):
        return -x
    return x.neg_(**kwargs)


def reciprocal_(x, **kwargs):
    if not torch.is_tensor(x):
        return 1/x
    return (
        x.reciprocal(**kwargs) if x.requires_grad else
        x.reciprocal_(**kwargs)
    )


def abs_(x, **kwargs):
    if not torch.is_tensor(x):
        return abs(x)
    if torch.is_complex(x):
        # abs_ not supported for complex tensors
        return x.abs(**kwargs)
    return x.abs(**kwargs) if x.requires_grad else x.abs_(**kwargs)


def exp_(x, **kwargs):
    if not torch.is_tensor(x):
        return math.exp(x)
    return x.exp(**kwargs) if x.requires_grad else x.exp_(**kwargs)


def log_(x, **kwargs):
    if not torch.is_tensor(x):
        return math.log(x)
    return x.log(**kwargs) if x.requires_grad else x.log_(**kwargs)


def atan_(x, **kwargs):
    if not torch.is_tensor(x):
        return math.atan(x)
    return x.atan(**kwargs) if x.requires_grad else x.atan_(**kwargs)
