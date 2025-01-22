import random
import torch
from cornucopia import random as ccrand

SEED = 12345678


def test_backward_random_fixed():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    a = torch.full([], 0.0, requires_grad=True)
    r = ccrand.Fixed(a)
    r().backward()
    assert (a.grad is not None)


def test_backward_random_uniform():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    a = torch.full([], 0.0, requires_grad=True)
    b = torch.full([], 1.0, requires_grad=True)
    r = ccrand.Uniform(a, b)
    r().backward()
    assert (a.grad is not None) and (b.grad is not None)


def test_backward_random_normal():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    m = torch.full([], 0.0, requires_grad=True)
    s = torch.full([], 1.0, requires_grad=True)
    r = ccrand.Normal(m, s)
    r().backward()
    assert (m.grad is not None) and (s.grad is not None)


def test_backward_random_lognormal():
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    m = torch.full([], 0.0, requires_grad=True)
    s = torch.full([], 1.0, requires_grad=True)
    r = ccrand.LogNormal(m, s)
    r().backward()
    assert (m.grad is not None) and (s.grad is not None)
