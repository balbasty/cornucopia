import pytest
import random
import torch
from cornucopia import (
    IntensityTransform,
    SynthFromLabelTransform,
    SmoothLabelMap,
)

SEED = 12345678
sizes_nochannels = ((1, 32, 32), (1, 8, 8, 8))
sizes = sizes_nochannels + ((3, 32, 32),)


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("bias", [False, 7])
@pytest.mark.parametrize("gamma", [False, 0.6])
@pytest.mark.parametrize("motion", [False, 3])
@pytest.mark.parametrize("resolution", [False, 8])
@pytest.mark.parametrize("snr", [False, 10])
@pytest.mark.parametrize("gfactor", [False, 5])
@pytest.mark.parametrize("order", [1, 3])
def test_run_synth_intensity(
    size, bias, gamma, motion, resolution, snr, gfactor, order
):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = IntensityTransform(
        bias=bias,
        gamma=gamma,
        motion_fwhm=motion,
        resolution=resolution,
        snr=snr,
        gfactor=gfactor,
        order=order,

    )(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("patch", [None, 8])
@pytest.mark.parametrize("gmm_fwhm", [0, 10])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("elastic", [False, 0.05])
@pytest.mark.parametrize("elastic_steps", [0, 7])
def test_run_synth_fromlabel(
    size, patch, gmm_fwhm, affine, elastic, elastic_steps,
):
    if not affine:
        affine = dict(rotation=0, shears=0, zooms=0)
    else:
        affine = {}

    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.zeros([]).expand(size)
    x = SmoothLabelMap()(x)
    _ = SynthFromLabelTransform(
        patch=patch,
        gmm_fwhm=gmm_fwhm,
        elastic=elastic,
        elastic_steps=elastic_steps,
        **affine,
    )(x)
    assert True
