import pytest
import random
import torch
from cornucopia.geometric import (
    ElasticTransform,
    RandomElasticTransform,
    AffineTransform,
    RandomAffineTransform,
    AffineElasticTransform,
    RandomAffineElasticTransform,
    MakeAffinePair,
    SlicewiseAffineTransform,
    RandomSlicewiseAffineTransform,
)

SEED = 12345678
sizes = ((1, 32, 32), (3, 32, 32), (1, 8, 8, 8))


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("unit", ('fov', 'vox'))
@pytest.mark.parametrize("steps", (0, 8))
@pytest.mark.parametrize("order", (1, 3))
def test_run_geom_elastic(size, unit, steps, order):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = ElasticTransform(unit=unit, steps=steps, order=order)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("unit", ('fov', 'vox'))
@pytest.mark.parametrize("steps", (0, 8))
@pytest.mark.parametrize("order", (1, 3))
def test_run_geom_elastic_random(size, unit, steps, order):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomElasticTransform(unit=unit, steps=steps, order=order)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("unit", ('fov', 'vox'))
def test_run_geom_affine(size, unit):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = AffineTransform(
        unit=unit,
        translations=0.1,
        rotations=1,
        zooms=0.1,
        shears=0.1,
    )(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("unit", ('fov', 'vox'))
@pytest.mark.parametrize("iso", (True, False))
def test_run_geom_affine_random(size, unit, iso):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomAffineTransform(unit=unit, iso=iso)(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("unit", ('fov', 'vox'))
@pytest.mark.parametrize("steps", (0, 8))
@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("patch", (None, 8))
def test_run_geom_affineelastic(size, unit, steps, order, patch):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = AffineElasticTransform(
        translations=0.1,
        rotations=1,
        zooms=0.1,
        shears=0.1,
        unit=unit,
        steps=steps,
        order=order,
        patch=patch,
    )(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("unit", ('fov', 'vox'))
@pytest.mark.parametrize("steps", (0, 8))
@pytest.mark.parametrize("order", (1, 3))
@pytest.mark.parametrize("patch", (None, 8))
def test_run_geom_affineelastic_random(size, unit, steps, order, patch):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomAffineElasticTransform(
        unit=unit,
        steps=steps,
        order=order,
        patch=patch
    )(x)
    assert True


@pytest.mark.parametrize("size", sizes)
def test_run_geom_affinepair(size):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = MakeAffinePair()(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("unit", ('fov', 'vox'))
@pytest.mark.parametrize("spacing", (1, 2, 4))
@pytest.mark.parametrize("slice", (0, -1))
def test_run_geom_slicewise(size, unit, spacing, slice):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = SlicewiseAffineTransform(
        translations=[0.1]*x[0].shape[slice],
        rotations=[1]*x[0].shape[slice],
        zooms=[0.1]*x[0].shape[slice],
        shears=[0.1]*x[0].shape[slice],
        unit=unit,
        spacing=spacing,
        slice=slice
    )(x)
    assert True


@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("unit", ('fov', 'vox'))
@pytest.mark.parametrize("spacing", (1, 2, 4))
@pytest.mark.parametrize("slice", (0, -1))
@pytest.mark.parametrize("shots", (1, 2))
@pytest.mark.parametrize("nodes", (0, 1, 2, 4))
def test_run_geom_slicewise_random(size, unit, spacing, slice, shots, nodes):
    random.seed(SEED)
    torch.random.manual_seed(SEED)
    x = torch.randn(size)
    _ = RandomSlicewiseAffineTransform(
        unit=unit,
        spacing=spacing,
        slice=slice,
        shots=shots,
        nodes=nodes,
    )(x)
    assert True
