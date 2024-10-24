import torch
from cornucopia import GammaTransform


def test_gamma_nonan():
    x = torch.ones([2, 16, 16])
    t = GammaTransform()
    y = t(x)
    assert torch.allclose(x, y)
