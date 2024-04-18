import pytest
import random
import torch
from cornucopia import (
    OneHotTransform,
    ArgMaxTransform,
    RelabelTransform,
    GaussianMixtureTransform,
    RandomGaussianMixtureTransform,
    SmoothLabelMap,
    RandomSmoothLabelMap,
    ErodeLabelTransform,
    RandomErodeLabelTransform,
    DilateLabelTransform,
    RandomDilateLabelTransform,
    SmoothMorphoLabelTransform,
    RandomSmoothMorphoLabelTransform,
    SmoothShallowLabelTransform,
    RandomSmoothShallowLabelTransform,
    BernoulliTransform,
    SmoothBernoulliTransform,
    BernoulliDiskTransform,
    SmoothBernoulliDiskTransform,
)


SEED = 12345678
sizes_nochannels = ((1, 32, 32), (1, 8, 8, 8))
sizes = sizes_nochannels + ((3, 32, 32),)


@pytest.mark.parametrize("size", sizes_nochannels)
def test_run_onehot(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randint(0, 4, size)
    _ = OneHotTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
def test_run_argmax(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.rand([4, *size[1:]])
    _ = ArgMaxTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
def test_run_relabel(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randint(0, 4, size)
    _ = RelabelTransform([0, 4, 3, 2, 1])(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
@pytest.mark.parametrize("fwhm", [0, 1])
@pytest.mark.parametrize("background", [True, False])
def test_run_gmm(size, fwhm, background):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randint(0, 4, size)
    mu = torch.randn([5])
    sig = torch.rand([5])
    _ = GaussianMixtureTransform(
        mu, sig,
        fwhm=fwhm,
        background=background,
    )(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
@pytest.mark.parametrize("fwhm", [0, 1])
@pytest.mark.parametrize("background", [True, False])
def test_run_gmm_random(size, fwhm, background):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randint(0, 4, size)
    _ = RandomGaussianMixtureTransform(
        fwhm=fwhm,
        background=background,
    )(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
@pytest.mark.parametrize("soft", [True, False])
def test_run_smoothmap(size, soft):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = SmoothLabelMap(soft=soft)(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
@pytest.mark.parametrize("soft", [True, False])
def test_run_smoothmap_random(size, soft):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros(size)
    _ = RandomSmoothLabelMap(soft=soft)(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
@pytest.mark.parametrize("method", ['conv', 'l2'])
@pytest.mark.parametrize("new_labels", [True, False])
def test_run_smoothmap_erode(size, method, new_labels):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = SmoothLabelMap()(torch.zeros(size))
    _ = ErodeLabelTransform(method=method, new_labels=new_labels)(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
@pytest.mark.parametrize("method", ['conv', 'l2'])
def test_run_smoothmap_erode_random(size, method):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = SmoothLabelMap()(torch.zeros(size))
    _ = RandomErodeLabelTransform(method=method)(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
@pytest.mark.parametrize("method", ['conv', 'l2'])
def test_run_smoothmap_dilate(size, method):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = SmoothLabelMap()(torch.zeros(size))
    _ = DilateLabelTransform(method=method)(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
@pytest.mark.parametrize("method", ['conv', 'l2'])
def test_run_smoothmap_dilate_random(size, method):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = SmoothLabelMap()(torch.zeros(size))
    _ = RandomDilateLabelTransform(method=method)(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
@pytest.mark.parametrize("method", ['conv', 'l2'])
def test_run_smoothmap_morpho(size, method):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = SmoothLabelMap()(torch.zeros(size))
    _ = SmoothMorphoLabelTransform(method=method)(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
@pytest.mark.parametrize("method", ['conv', 'l2'])
def test_run_smoothmap_morpho_random(size, method):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = SmoothLabelMap()(torch.zeros(size))
    _ = RandomSmoothMorphoLabelTransform(method=method)(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
def test_run_smoothmap_shallow(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = SmoothLabelMap()(torch.zeros(size))
    _ = SmoothShallowLabelTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
def test_run_smoothmap_shallow_random(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = SmoothLabelMap()(torch.zeros(size))
    _ = RandomSmoothShallowLabelTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
def test_run_smoothmap_bernoulli(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = SmoothLabelMap()(torch.zeros(size))
    _ = BernoulliTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
def test_run_smoothmap_bernoulli_smooth(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = SmoothLabelMap()(torch.zeros(size))
    _ = SmoothBernoulliTransform()(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
@pytest.mark.parametrize("method", ['conv', 'l2'])
def test_run_smoothmap_bernoulli_disk(size, method):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = SmoothLabelMap()(torch.zeros(size))
    _ = BernoulliDiskTransform(method=method)(x)
    assert True


@pytest.mark.parametrize("size", sizes_nochannels)
@pytest.mark.parametrize("method", ['conv', 'l2'])
def test_run_smoothmap_bernoulli_disk_smooth(size, method):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = SmoothLabelMap()(torch.zeros(size))
    _ = SmoothBernoulliDiskTransform(method=method)(x)
    assert True
