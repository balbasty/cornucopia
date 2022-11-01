from random import Random


def sym_range(x):
    if not isinstance(x, Random):
        if isinstance(x, (list, tuple)):
            x = (tuple(-x1 for x1 in x), x)
        else:
            x = (-x, x)
    return x


def upper_range(x, min=0):
    if not isinstance(x, Random):
        x = (min, x)
    return x


def lower_range(x, max=0):
    if not isinstance(x, Random):
        x = (x, max)
    return x