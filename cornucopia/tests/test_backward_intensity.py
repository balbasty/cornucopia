import pytest
import random
import torch
from cornucopia import (
    AddValueTransform,
    MulValueTransform,
    AddMulTransform,
    FillValueTransform,
    ClipTransform,
    AddFieldTransform,
    MulFieldTransform,
    RandomAddFieldTransform,
    RandomMulFieldTransform,
    RandomSlicewiseMulFieldTransform,
    RandomMulTransform,
    RandomAddTransform,
    RandomAddMulTransform,
    GammaTransform,
    RandomGammaTransform,
    ZTransform,
    QuantileTransform,
)
import cornucopia.random as ccrand


SEED = 12345678
SIZE = (1, 32, 32)


def test_backward_intensity_add():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 1.0, requires_grad=True)
    y = AddValueTransform(a)(x)
    y.sum().backward()
    assert a.grad is not None


def test_backward_intensity_mul():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 2.0, requires_grad=True)
    y = MulValueTransform(a)(x)
    y.sum().backward()
    assert a.grad is not None


def test_backward_intensity_addmul():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 2.0, requires_grad=True)
    b = torch.full([], 1.0, requires_grad=True)
    y = AddMulTransform(a, b)(x)
    y.sum().backward()
    assert (a.grad is not None) and (b is not None)


def test_backward_intensity_fill():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 0.0, requires_grad=True)
    y = FillValueTransform(x < 0.5, a)(x)
    y.sum().backward()
    assert a.grad is not None


def test_backward_intensity_clip():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 0.25, requires_grad=True)
    b = torch.full([], 0.75, requires_grad=True)
    y = ClipTransform(a, b)(x)
    y.sum().backward()
    assert (a.grad is not None) and (b is not None)


@pytest.mark.parametrize("order", (1, 3))
def test_backward_intensity_addfield(order):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 0.0, requires_grad=True)
    b = torch.full([], 1.0, requires_grad=True)
    y = AddFieldTransform(vmin=a, vmax=b, order=order)(x)
    y.sum().backward()
    assert (a.grad is not None) and (b is not None)


@pytest.mark.parametrize("order", (1, 3))
def test_backward_intensity_mulfield(order):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 0.0, requires_grad=True)
    b = torch.full([], 1.0, requires_grad=True)
    y = MulFieldTransform(vmin=a, vmax=b, order=order)(x)
    y.sum().backward()
    assert (a.grad is not None) and (b is not None)


@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("shared", (True, False))
def test_backward_intensity_addfield_random(order, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 1.0, requires_grad=True)
    b = torch.full([], 2.0, requires_grad=True)
    a_samp = ccrand.Uniform(0, a)
    b_samp = ccrand.Uniform(1, b)
    y = RandomAddFieldTransform(
        vmin=a_samp, vmax=b_samp, order=order, shared=shared
    )(x)
    y.sum().backward()
    assert (a.grad is not None) and (b is not None)


@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("shared", (True, False))
def test_backward_intensity_mulfield_random(order, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 1.0, requires_grad=True)
    a_samp = ccrand.Uniform(0, a)
    y = RandomMulFieldTransform(
        vmax=a_samp, order=order, shared=shared
    )(x)
    y.sum().backward()
    assert (a.grad is not None)


@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("shared", (True, False))
def test_backward_intensity_mulfield_slicewise_random(order, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 1.0, requires_grad=True)
    a_samp = ccrand.Uniform(0, a)
    y = RandomSlicewiseMulFieldTransform(
        vmax=a_samp, order=order, shared=shared
    )(x)
    y.sum().backward()
    assert (a.grad is not None)


@pytest.mark.parametrize("shared", (True, False))
def test_backward_intensity_add_random(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 1.0, requires_grad=True)
    a_samp = ccrand.Uniform(0, a)
    y = RandomAddTransform(a_samp, shared=shared)(x)
    y.sum().backward()
    assert (a.grad is not None)


@pytest.mark.parametrize("shared", (True, False))
def test_backward_intensity_mul_random(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 1.0, requires_grad=True)
    a_samp = ccrand.Uniform(0, a)
    y = RandomMulTransform(a_samp, shared=shared)(x)
    y.sum().backward()
    assert (a.grad is not None)


@pytest.mark.parametrize("shared", (True, False))
def test_backward_intensity_addmul_random(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 1.0, requires_grad=True)
    a_samp = ccrand.Uniform(0, a)
    y = RandomAddMulTransform(a_samp, shared=shared)(x)
    y.sum().backward()
    assert (a.grad is not None)


def test_backward_intensity_gamma():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 0.5, requires_grad=True)
    y = GammaTransform(a)(x)
    y.sum().backward()
    assert (a.grad is not None)


@pytest.mark.parametrize("shared", (True, False))
def test_backward_intensity_gamma_random(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    a = torch.full([], 1.0, requires_grad=True)
    a_samp = ccrand.Uniform(0, a)
    y = RandomGammaTransform(a_samp, shared=shared)(x)
    y.sum().backward()
    assert (a.grad is not None)


@pytest.mark.parametrize("shared", (True, False))
def test_backward_intensity_ztransform(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    m = torch.full([], 0.0, requires_grad=True)
    s = torch.full([], 1.0, requires_grad=True)
    y = ZTransform(mu=m, sigma=s, shared=shared)(x)
    y.sum().backward()
    assert (m.grad is not None) and (s.grad is not None)


def test_backward_intensity_quantile():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    q0 = torch.full([], -1.0, requires_grad=True)
    q1 = torch.full([],  1.0, requires_grad=True)
    mn = torch.full([], 0.0, requires_grad=True)
    mx = torch.full([], 1.0, requires_grad=True)
    y = QuantileTransform(
        pmin=q0.sigmoid(), pmax=q1.sigmoid(), vmin=mn, vmax=mx
    )(x)
    y.sum().backward()
    assert (
        (q0.grad is not None) and
        (q1.grad is not None) and
        (mn.grad is not None) and
        (mx.grad is not None)
    ), [
        k for k, v in {'q0': q0, 'q1': q1, 'mn': mn, 'mx': mx}.items()
        if v.grad is None
    ]
