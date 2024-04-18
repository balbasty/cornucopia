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


SEED = 12345678
sizes = ((1, 32, 32), (3, 32, 32), (1, 8, 8, 8))


@pytest.mark.parametrize("size", sizes)
def test_run_intensity_add(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = AddValueTransform(1)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_intensity_mul(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = MulValueTransform(2)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_intensity_addmul(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = AddMulTransform(2, 1)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_intensity_fill(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = FillValueTransform(x < 0.5, 0)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_intensity_clip(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = ClipTransform(0.25, 0.75)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("order", (1, 3))
def test_run_intensity_addfield(size, order):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = AddFieldTransform(order=order)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("order", (1, 3))
def test_run_intensity_mulfield(size, order):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = MulFieldTransform(order=order)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("shared", (True, False))
def test_run_intensity_addfield_random(size, order, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomAddFieldTransform(order=order, shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("shared", (True, False))
def test_run_intensity_mulfield_random(size, order, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomMulFieldTransform(order=order, shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("shared", (True, False))
def test_run_intensity_mulfield_slicewise_random(size, order, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomSlicewiseMulFieldTransform(order=order, shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", (True, False))
def test_run_intensity_add_random(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomAddTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", (True, False))
def test_run_intensity_mul_random(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomMulTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", (True, False))
def test_run_intensity_addmul_random(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomAddMulTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_intensity_gamma(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = GammaTransform(0.5)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", (True, False))
def test_run_intensity_gamma_random(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomGammaTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", (True, False))
def test_run_intensity_ztransform(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = ZTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_intensity_quantile(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = QuantileTransform()(x)
    assert True
