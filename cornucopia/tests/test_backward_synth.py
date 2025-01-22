import pytest
import random
import torch
from cornucopia import (
    IntensityTransform,
    SynthFromLabelTransform,
    SmoothLabelMap,
)

SEED = 12345678
SIZE = (1, 32, 32)


@pytest.mark.parametrize("order", [1, 3])
def test_backward_synth_intensity(order):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    g = torch.full([], 0.6, requires_grad=True)
    m = torch.full([], 3.0, requires_grad=True)
    r = torch.full([], 8.0, requires_grad=True)
    n = torch.full([], 10.0, requires_grad=True)
    y = IntensityTransform(
        gamma=g,
        motion_fwhm=m,
        resolution=r,
        snr=n,
        order=order,

    )(x)
    y.sum().backward()
    assert (
        (g.grad is not None) and
        (m.grad is not None) and
        (r.grad is not None) and
        (n.grad is not None)
    ), [
        k for k, v in {'g': g, 'm': m, 'r': r, 'n': n}.items()
        if v.grad is None
    ]


@pytest.mark.parametrize("patch", [None, 8])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("elastic_steps", [0, 7])
def test_backward_synth_fromlabel(patch, affine, elastic_steps):
    if not affine:
        affine = dict(rotation=0, shears=0, zooms=0)
    else:
        affine = {}

    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros([]).expand(SIZE)
    f = torch.full([], 10.0, requires_grad=True)
    d = torch.full([], 0.05, requires_grad=True)
    x = SmoothLabelMap(soft=True)(x)
    y, z = SynthFromLabelTransform(
        patch=patch,
        gmm_fwhm=f,
        elastic=d,
        elastic_steps=elastic_steps,
        **affine,
    )(x)
    y.sum().backward()
    assert (
        (f.grad is not None) and
        (d.grad is not None)
    ),  [
        k for k, v in {'f': f, 'd': d}.items()
        if v.grad is None
    ]
