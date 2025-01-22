import pytest
import random
import torch
from cornucopia import (
    ArrayCoilTransform,
    SumOfSquaresTransform,
    IntraScanMotionTransform,
    SmallIntraScanMotionTransform,
)


SEED = 12345678
SIZE = (1, 32, 32)


def test_run_kspace_coils():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    f = torch.full([], 0.5, requires_grad=True)
    d = torch.full([], 0.8, requires_grad=True)
    j = torch.full([], 0.01, requires_grad=True)
    x = torch.randn(SIZE)
    y = ArrayCoilTransform(fwhm=f, diameter=d, jitter=j)(x)
    y.abs().sum().backward()
    assert (
        (f.grad is not None) and
        (d.grad is not None) and
        (j.grad is not None)
    )


def test_run_kspace_ssq():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    f = torch.full([], 0.5, requires_grad=True)
    d = torch.full([], 0.8, requires_grad=True)
    j = torch.full([], 0.01, requires_grad=True)
    y = ArrayCoilTransform(
        fwhm=f, diameter=d, jitter=j, returns='uncombined'
    )(x)
    z = SumOfSquaresTransform()(y)
    z.sum().backward()
    assert (
        (f.grad is not None) and
        (d.grad is not None) and
        (j.grad is not None)
    )


@pytest.mark.parametrize("pattern", ('sequential', 'random'))
@pytest.mark.parametrize("axis", (1, -1))
@pytest.mark.parametrize("freq", (True, False))
def test_run_kspace_motion(pattern, axis, freq):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    t = torch.full([], 0.1, requires_grad=True)
    r = torch.full([], 15.0, requires_grad=True)
    y = IntraScanMotionTransform(
        translations=t,
        rotations=r,
        pattern=pattern,
        axis=axis,
        freq=freq,
    )(x)
    y.sum().backward()
    assert (
        (t.grad is not None) and
        (r.grad is not None)
    )


def test_run_kspace_motion_coils():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    t = torch.full([], 0.1, requires_grad=True)
    r = torch.full([], 15.0, requires_grad=True)
    f = torch.full([], 0.5, requires_grad=True)
    d = torch.full([], 0.8, requires_grad=True)
    j = torch.full([], 0.01, requires_grad=True)
    c = ArrayCoilTransform(fwhm=f, diameter=d, jitter=j)
    y = IntraScanMotionTransform(
        translations=t,
        rotations=r,
        coils=c
    )(x)
    y.sum().backward()
    assert (
        (f.grad is not None) and
        (d.grad is not None) and
        (j.grad is not None) and
        (t.grad is not None) and
        (r.grad is not None)
    )


@pytest.mark.parametrize("axis", (1, -1))
def test_run_kspace_motion_small(axis):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(SIZE)
    t = torch.full([], 0.05, requires_grad=True)
    r = torch.full([], 5.0,  requires_grad=True)
    y = SmallIntraScanMotionTransform(
        translations=t,
        rotations=r,
        axis=axis,
    )(x)
    y.sum().backward()
    assert (
        (t.grad is not None) and
        (r.grad is not None)
    )
