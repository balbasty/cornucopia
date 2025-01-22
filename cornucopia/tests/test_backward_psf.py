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

# NOTE / TODO
#   The "low res" layers use torch.interpol under the hood, whose
#   scaling factor depends on the input and target sizes. There is
#   therefore no way to properly backpropagate through the resolution
#   (currently, we "cheat" by first smoothing the image by the resolution,
#   so there are some gradients flowing, but we're missing stuff).
#   We could fix that by reimplementing `interpol` using `sample_grid`,
#   and compute the flow field from the resolution parameter directly
#   (rather than go "resolution -> target size -> flow field", since
#   "resolution -> target_size" breaks the graph).


SEED = 12345678
SIZE = (2, 32, 32)


def test_backward_psf_smooth():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    f = torch.full([], 1.0, requires_grad=True)
    y = SmoothTransform(fwhm=f)(x)
    y.sum().backward()
    assert (f.grad is not None)


@pytest.mark.parametrize("shared", [True, False])
def test_backward_psf_smooth_random(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    f = torch.full([], 1.0, requires_grad=True)
    y = RandomSmoothTransform(fwhm=f, shared=shared)(x)
    y.sum().backward()
    assert (f.grad is not None)


def test_backward_psf_lowres():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    r = torch.full([], 5.0, requires_grad=True)
    y = LowResTransform(resolution=r)(x)
    y.sum().backward()
    assert (r.grad is not None)


def test_backward_psf_lowres_noise():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    noise = RandomChiNoiseTransform()
    x = torch.zeros(SIZE)
    r = torch.full([], 5.0, requires_grad=True)
    y = LowResTransform(resolution=r, noise=noise)(x)
    y.sum().backward()
    assert (r.grad is not None)


@pytest.mark.parametrize("shared", [True, False])
def test_backward_psf_lowres_random(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    r = torch.full([], 5.0, requires_grad=True)
    y = RandomLowResTransform(resolution=r, shared=shared)(x)
    y.sum().backward()
    assert (r.grad is not None)


@pytest.mark.parametrize("shared", [True, False])
def test_backward_psf_lowres_noise_random(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    noise = RandomChiNoiseTransform()
    x = torch.zeros(SIZE)
    r = torch.full([], 5.0, requires_grad=True)
    y = RandomLowResTransform(resolution=r, noise=noise, shared=shared)(x)
    y.sum().backward()
    assert (r.grad is not None)


def test_backward_psf_slicelowres():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    r = torch.full([], 5.0, requires_grad=True)
    t = torch.full([], 0.5, requires_grad=True)
    y = LowResSliceTransform(resolution=r, thickness=t)(x)
    print(x)
    y.sum().backward()
    assert (r.grad is not None) and (t.grad is not None)


def test_backward_psf_slicelowres_noise():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    noise = RandomChiNoiseTransform()
    x = torch.zeros(SIZE)
    r = torch.full([], 5.0, requires_grad=True)
    t = torch.full([], 0.5, requires_grad=True)
    y = LowResSliceTransform(resolution=r, thickness=t, noise=noise)(x)
    y.sum().backward()
    assert (r.grad is not None) and (t.grad is not None)


@pytest.mark.parametrize("shared", [True, False])
def test_backward_psf_slicelowres_random(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(SIZE)
    r = torch.full([], 5.0, requires_grad=True)
    t = torch.full([], 0.5, requires_grad=True)
    y = RandomLowResSliceTransform(resolution=r, thickness=t, shared=shared)(x)
    y.sum().backward()
    assert (r.grad is not None) and (t.grad is not None)


@pytest.mark.parametrize("shared", [True, False])
def test_backward_psf_slicelowres_noise_random(shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    noise = RandomChiNoiseTransform()
    x = torch.zeros(SIZE)
    r = torch.full([], 5.0, requires_grad=True)
    t = torch.full([], 0.5, requires_grad=True)
    y = RandomLowResSliceTransform(
        resolution=r, thickness=t, noise=noise, shared=shared
    )(x)
    y.sum().backward()
    assert (r.grad is not None) and (t.grad is not None)
