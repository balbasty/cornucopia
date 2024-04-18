import pytest
import random
import torch
from cornucopia import (
    GaussianNoiseTransform,
    RandomGaussianNoiseTransform,
    ChiNoiseTransform,
    RandomChiNoiseTransform,
    GammaNoiseTransform,
    RandomGammaNoiseTransform,
    GFactorTransform,
)


SEED = 12345678
sizes = ((1, 32, 32), (1, 8, 8, 8), (3, 32, 32),)


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
def test_run_noise_gaussian(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = GaussianNoiseTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("shared_noise", [True, False])
def test_run_noise_gaussian_random(size, shared, shared_noise):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = RandomGaussianNoiseTransform(
        shared=shared,
        shared_noise=shared_noise
    )(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
def test_run_noise_chi(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = ChiNoiseTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("shared_noise", [True, False])
def test_run_noise_chi_random(size, shared, shared_noise):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = RandomChiNoiseTransform(
        shared=shared,
        shared_noise=shared_noise
    )(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
def test_run_noise_gamma(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.ones(size)
    _ = GammaNoiseTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("shared_noise", [True, False])
def test_run_noise_gamma_random(size, shared, shared_noise):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = RandomGammaNoiseTransform(
        shared=shared,
        shared_noise=shared_noise
    )(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("shared_field", [True, False])
def test_run_noise_gaussian_gfactor(size, shared, shared_field):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    noise = GaussianNoiseTransform(shared=shared)
    x = torch.zeros(size)
    _ = GFactorTransform(noise, shared=shared_field)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("shared_field", [True, False])
def test_run_noise_gchi_gfactor(size, shared, shared_field):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    noise = ChiNoiseTransform(shared=shared)
    x = torch.zeros(size)
    _ = GFactorTransform(noise, shared=shared_field)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("shared_noise", [True, False])
@pytest.mark.parametrize("shared_field", [True, False])
def test_run_noise_gchi_random_gfactor(
    size, shared, shared_noise, shared_field
):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    noise = RandomChiNoiseTransform(shared=shared, shared_noise=shared_noise)
    x = torch.zeros(size)
    _ = GFactorTransform(noise, shared=shared_field)(x)
    assert True
