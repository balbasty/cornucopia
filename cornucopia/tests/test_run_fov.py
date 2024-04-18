import pytest
import random
import torch
from cornucopia.fov import (
    FlipTransform,
    RandomFlipTransform,
    PermuteAxesTransform,
    RandomPermuteAxesTransform,
    PatchTransform,
    RandomPatchTransform,
    CropTransform,
    PadTransform,
    PowerTwoTransform,
)

SEED = 12345678
sizes = ((1, 32, 32), (3, 32, 32), (1, 8, 8, 8))


@pytest.mark.parametrize("size", sizes)
def test_run_fov_flip(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = FlipTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_fov_flip_random(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomFlipTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_fov_permute(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = PermuteAxesTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_fov_permute_random(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomPermuteAxesTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_fov_patch(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = PatchTransform(8)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_fov_patch_random(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomPatchTransform(8)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_fov_crop(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = CropTransform(8)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_fov_pad(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = PadTransform(2)(x)
    assert True


@pytest.mark.parametrize("size", ((1, 31, 31), (3, 31, 31), (1, 5, 5, 5)))
def test_run_fov_power2(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = PowerTwoTransform(3)(x)
    assert True
