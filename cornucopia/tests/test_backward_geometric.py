import pytest
import random
import torch
from cornucopia.geometric import (
    ElasticTransform,
    RandomElasticTransform,
    AffineTransform,
    RandomAffineTransform,
    AffineElasticTransform,
    RandomAffineElasticTransform,
)
from cornucopia import random as ccrand

SEED = 12345678
SIZE = [1, 32, 32]


@pytest.mark.parametrize("steps", (0, 8))
@pytest.mark.parametrize("order", (1, 3))
def test_backward_geom_elastic(steps, order):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    d = torch.full([], 0.1, requires_grad=True)
    y = ElasticTransform(dmax=d, steps=steps, order=order)(x)
    y.sum().backward()
    assert d.grad is not None


@pytest.mark.parametrize("steps", (0, 8))
@pytest.mark.parametrize("order", (1, 3))
def test_backward_geom_elastic_random(steps, order):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    d = torch.full([], 0.1, requires_grad=True)
    d_samp = ccrand.Uniform(d)
    y = RandomElasticTransform(dmax=d_samp, steps=steps, order=order)(x)
    y.sum().backward()
    assert d.grad is not None


def test_backward_geom_affine():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    t = torch.full([],  0.1, requires_grad=True)
    r = torch.full([],  1.0, requires_grad=True)
    z = torch.full([], -2.5, requires_grad=True)
    s = torch.full([],  0.1, requires_grad=True)
    x = torch.randn(SIZE)
    y = AffineTransform(
        translations=t,
        rotations=r,
        zooms=z.exp(),
        shears=s,
    )(x)
    y.sum().backward()
    assert (
        (t.grad is not None) and
        (r.grad is not None) and
        (z.grad is not None) and
        (s.grad is not None)
    ), [
        k for k, v in {'t': t, 'r': r, 'z': z, 's': s}.items()
        if v.grad is None
    ]


@pytest.mark.parametrize("iso", (True, False))
def test_backward_geom_affine_random(iso):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    t = torch.full([],  0.1,   requires_grad=True)
    r = torch.full([],  1.0,   requires_grad=True)
    z = torch.full([],  0.15,  requires_grad=True)
    s = torch.full([],  0.012, requires_grad=True)
    t_samp = ccrand.Uniform(-t, t)
    r_samp = ccrand.Uniform(-r, t)
    z_samp = ccrand.Uniform(1-z, 1+z)
    s_samp = ccrand.Uniform(-s, s)
    y = RandomAffineTransform(
        translations=t_samp,
        rotations=r_samp,
        zooms=z_samp,
        shears=s_samp,
        iso=iso
    )(x)
    y.sum().backward()
    assert (
        (t.grad is not None) and
        (r.grad is not None) and
        (z.grad is not None) and
        (s.grad is not None)
    ), [
        k for k, v in {'t': t, 'r': r, 'z': z, 's': z}.items()
        if v.grad is None
    ]


@pytest.mark.parametrize("steps", (0, 8))
@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("patch", (None, 8))
def test_backward_geom_affineelastic(steps, order, patch):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    t = torch.full([],  0.1,   requires_grad=True)
    r = torch.full([],  1.0,   requires_grad=True)
    z = torch.full([],  0.15,  requires_grad=True)
    s = torch.full([],  0.012, requires_grad=True)
    d = torch.full([],  15.0,  requires_grad=True)
    y = AffineElasticTransform(
        translations=t,
        rotations=r,
        zooms=z,
        shears=s,
        dmax=d,
        steps=steps,
        order=order,
        patch=patch,
    )(x)
    y.sum().backward()
    assert (
        (d.grad is not None) and
        (t.grad is not None) and
        (r.grad is not None) and
        (z.grad is not None) and
        (s.grad is not None)
    ), [
        k for k, v in {'d': d, 't': t, 'r': r, 'z': z, 's': z}.items()
        if v.grad is None
    ]


@pytest.mark.parametrize("steps", (0, 8))
@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("patch", (None, 8))
def test_backward_geom_affineelastic_random(steps, order, patch):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    t = torch.full([],  0.1,   requires_grad=True)
    r = torch.full([],  1.0,   requires_grad=True)
    z = torch.full([],  0.15,  requires_grad=True)
    s = torch.full([],  0.012, requires_grad=True)
    d = torch.full([],  15.0,  requires_grad=True)
    t_samp = ccrand.Uniform(-t, t)
    r_samp = ccrand.Uniform(-r, t)
    z_samp = ccrand.Uniform(1-z, 1+z)
    s_samp = ccrand.Uniform(-s, s)
    d_samp = ccrand.Uniform(d)
    y = RandomAffineElasticTransform(
        translations=t_samp,
        rotations=r_samp,
        zooms=z_samp,
        shears=s_samp,
        dmax=d_samp,
        steps=steps,
        order=order,
        patch=patch
    )(x)
    y.sum().backward()
    assert (
        (d.grad is not None) and
        (t.grad is not None) and
        (r.grad is not None) and
        (z.grad is not None) and
        (s.grad is not None)
    ), [
        k for k, v in {'d': d, 't': t, 'r': r, 'z': z, 's': z}.items()
        if v.grad is None
    ]
