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
sizes_nochannels = ((1, 32, 32), (1, 8, 8, 8))
sizes = sizes_nochannels + ((3, 32, 32),)


@pytest.mark.parametrize("size", sizes_nochannels)
def test_run_kspace_coils(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = ArrayCoilTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
def test_run_kspace_ssq(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    y = ArrayCoilTransform(returns='uncombined')(x)
    _ = SumOfSquaresTransform()(y)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("pattern", ('sequential', 'random'))
@pytest.mark.parametrize("axis", (1, -1))
@pytest.mark.parametrize("freq", (True, False))
def test_run_kspace_motion(size, pattern, axis, freq):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = IntraScanMotionTransform(
        pattern=pattern,
        axis=axis,
        freq=freq,
    )(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
def test_run_kspace_motion_coils(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = IntraScanMotionTransform(coils=ArrayCoilTransform())(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("axis", (1, -1))
def test_run_kspace_motion_small(size, axis):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = SmallIntraScanMotionTransform(
        axis=axis,
    )(x)
    assert True
