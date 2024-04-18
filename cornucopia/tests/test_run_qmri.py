import pytest
import random
import torch
from cornucopia import (
    RandomSusceptibilityMixtureTransform,
    SusceptibilityToFieldmapTransform,
    ShimTransform,
    OptimalShimTransform,
    RandomShimTransform,
    HertzToPhaseTransform,
    GradientEchoTransform,
    RandomGMMGradientEchoTransform,
    SmoothLabelMap,
)


SEED = 12345678
sizes_nochannels = ((1, 32, 32), (1, 8, 8, 8))
sizes = sizes_nochannels + ((3, 32, 32),)


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
def test_run_qmri_susceptibility(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros([]).expand(size)
    x = SmoothLabelMap()(x)
    _ = RandomSusceptibilityMixtureTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_qmri_tofieldmap(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = SusceptibilityToFieldmapTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_qmri_tophase(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    x = SusceptibilityToFieldmapTransform()(x)
    x = HertzToPhaseTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_qmri_shim(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros([]).expand(size)
    x = SmoothLabelMap()(x)
    x = RandomSusceptibilityMixtureTransform()(x)
    x = SusceptibilityToFieldmapTransform()(x)
    if x.ndim == 3:
        linear = torch.randn([]).tolist()
        quad = torch.randn([]).tolist()
    else:
        assert x.ndim == 4
        linear = torch.randn([3]).tolist()
        quad = torch.randn([2]).tolist()
    _ = ShimTransform(linear, quad)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_qmri_shim_optimal(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros([]).expand(size)
    x = SmoothLabelMap()(x)
    x = RandomSusceptibilityMixtureTransform()(x)
    x = SusceptibilityToFieldmapTransform()(x)
    _ = OptimalShimTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("shared", [True, False])
def test_run_qmri_shim_random(size, shared):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros([]).expand(size)
    x = SmoothLabelMap()(x)
    x = RandomSusceptibilityMixtureTransform()(x)
    x = SusceptibilityToFieldmapTransform()(x)
    _ = RandomShimTransform(shared=shared)(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
def test_run_qmri_gre(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn([5, *size[1:]]).abs_()
    x[-2] /= x[-2].max()
    x[-1] *= 0.05 / x[-1].max()
    _ = GradientEchoTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
def test_run_qmri_gre_random(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros([]).expand(size)
    x = SmoothLabelMap()(x)
    _ = RandomGMMGradientEchoTransform()(x)
    assert True
