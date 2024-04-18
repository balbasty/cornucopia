import pytest
import random
import torch
from cornucopia.contrast import (
    ContrastLookupTransform,
    ContrastMixtureTransform,
)

SEED = 12345678
sizes = ((1, 32, 32), (3, 32, 32), (1, 8, 8, 8))


@pytest.mark.parametrize("size", sizes)
def test_run_contrast_lookup(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = ContrastLookupTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_contrast_mixture(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = ContrastMixtureTransform()(x)
    assert True
