import pytest
import random
import torch
from cornucopia.geometric import (
    RandomElasticTransform,
)

SEED = 12345678
sizes = ((1, 32, 32), (3, 32, 32), (1, 8, 8, 8))


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("unit", ('fov', 'vox'))
@pytest.mark.parametrize("steps", (0, 8))
@pytest.mark.parametrize("order", (1, 3))
def test_geom_elastic_zerocenter(size, unit, steps, order):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    f = RandomElasticTransform(
        unit=unit, steps=steps, order=order,
        zero_center=True, returns='flow'
    )(x)
    m = f.reshape(f.shape[:2] + (-1,)).mean(-1)
    m = m.reshape(f.shape[:2] + (1,) * f.shape[1])
    assert m.allclose(torch.zeros([len(size)]), atol=float('inf')), m.squeeze()
