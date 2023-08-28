import pytest
import random
import torch
from cornucopia import (
    SmoothTransform,
    RandomSmoothTransform,
    LowResTransform,
    RandomLowResTransform,
    LowResSliceTransform,
    RandomLowResSliceTransform,
    RandomChiNoiseTransform,
)


SEED = 12345678
sizes = ((1, 32, 32), (1, 8, 8, 8), (3, 32, 32),)


@pytest.mark.parametrize("size", sizes)
def test_run_psf_smooth(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = SmoothTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
def test_run_psf_smooth_random(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = RandomSmoothTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_psf_lowres(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = LowResTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_psf_lowres_noise(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    noise = RandomChiNoiseTransform()
    x = torch.zeros(size)
    _ = LowResTransform(noise=noise)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
def test_run_psf_lowres_random(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = RandomLowResTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
def test_run_psf_lowres_noise_random(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    noise = RandomChiNoiseTransform()
    x = torch.zeros(size)
    _ = RandomLowResTransform(noise=noise, shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_psf_slicelowres(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = LowResSliceTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_psf_slicelowres_noise(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    noise = RandomChiNoiseTransform()
    x = torch.zeros(size)
    _ = LowResSliceTransform(noise=noise)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
def test_run_psf_slicelowres_random(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = RandomLowResSliceTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
def test_run_psf_slicelowres_noise_random(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    noise = RandomChiNoiseTransform()
    x = torch.zeros(size)
    _ = RandomLowResSliceTransform(noise=noise, shared=shared)(x)
    assert True
