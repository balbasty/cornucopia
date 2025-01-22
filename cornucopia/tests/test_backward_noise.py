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
SIZE = (2, 32, 32)


@pytest.mark.parametrize("shared", [True, False])
def test_backward_noise_gaussian(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    s = torch.full([], 0.1, requires_grad=True)
    y = GaussianNoiseTransform(sigma=s, shared=shared)(x)
    y.sum().backward()
    assert s.grad is not None


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("shared_noise", [True, False])
def test_backward_noise_gaussian_random(shared, shared_noise):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    s = torch.full([], 0.2, requires_grad=True)
    y = RandomGaussianNoiseTransform(
        sigma=s,
        shared=shared,
        shared_noise=shared_noise
    )(x)
    y.sum().backward()
    assert s.grad is not None


@pytest.mark.parametrize("shared", [True, False])
def test_backward_noise_chi(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    s = torch.full([], 0.1, requires_grad=True)
    y = ChiNoiseTransform(sigma=s, shared=shared)(x)
    y.sum().backward()
    assert s.grad is not None


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("shared_noise", [True, False])
def test_backward_noise_chi_random(shared, shared_noise):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    s = torch.full([], 0.2, requires_grad=True)
    y = RandomChiNoiseTransform(
        sigma=s,
        shared=shared,
        shared_noise=shared_noise
    )(x)
    y.sum().backward()
    assert s.grad is not None


@pytest.mark.parametrize("shared", [True, False])
def test_backward_noise_gamma(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.ones(SIZE)
    s = torch.full([], 0.1, requires_grad=True)
    m = torch.full([], 1.0, requires_grad=True)
    y = GammaNoiseTransform(sigma=s, mean=m, shared=shared)(x)
    y.sum().backward()
    assert (s.grad is not None) and (m.grad is not None)


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("shared_noise", [True, False])
def test_backward_noise_gamma_random(shared, shared_noise):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    s = torch.full([], 0.2, requires_grad=True)
    m = torch.full([], 2.0, requires_grad=True)
    y = RandomGammaNoiseTransform(
        sigma=s,
        mean=m,
        shared=shared,
        shared_noise=shared_noise
    )(x)
    y.sum().backward()
    assert (s.grad is not None) and (m.grad is not None)


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("shared_field", [True, False])
def test_backward_noise_gaussian_gfactor(shared, shared_field):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    s = torch.full([], 0.1, requires_grad=True)
    mn = torch.full([], 0.5, requires_grad=True)
    mx = torch.full([], 1.5, requires_grad=True)
    noise = GaussianNoiseTransform(sigma=s, shared=shared)
    y = GFactorTransform(noise, vmin=mn, vmax=mx, shared=shared_field)(x)
    y.sum().backward()
    assert (
        (s.grad is not None) and
        (mn.grad is not None) and
        (mx.grad is not None)
    ), [
        k for k, v in {'s': s, 'mn': mn, 'mx': mx}.items()
        if v.grad is None
    ]


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("shared_field", [True, False])
def test_backward_noise_chi_gfactor(shared, shared_field):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    s = torch.full([], 0.1, requires_grad=True)
    mn = torch.full([], 0.5, requires_grad=True)
    mx = torch.full([], 1.5, requires_grad=True)
    noise = ChiNoiseTransform(sigma=s, shared=shared)
    y = GFactorTransform(noise, vmin=mn, vmax=mx, shared=shared_field)(x)
    y.sum().backward()
    assert (
        (s.grad is not None) and
        (mn.grad is not None) and
        (mx.grad is not None)
    ),  [
        k for k, v in {'s': s, 'mn': mn, 'mx': mx}.items()
        if v.grad is None
    ]


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("shared_noise", [True, False])
@pytest.mark.parametrize("shared_field", [True, False])
def test_backward_noise_chi_random_gfactor(shared, shared_noise, shared_field):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    s = torch.full([], 0.2, requires_grad=True)
    mn = torch.full([], 0.5, requires_grad=True)
    mx = torch.full([], 1.5, requires_grad=True)
    noise = RandomChiNoiseTransform(
        sigma=s, shared=shared, shared_noise=shared_noise
    )
    y = GFactorTransform(noise, vmin=mn, vmax=mx, shared=shared_field)(x)
    y.sum().backward()
    assert (
        (s.grad is not None) and
        (mn.grad is not None) and
        (mx.grad is not None)
    ), [
        k for k, v in {'s': s, 'mn': mn, 'mx': mx}.items()
        if v.grad is None
    ]
