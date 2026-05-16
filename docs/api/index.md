---
icon: fontawesome/brands/python
---

# API

## [`cc.special`](special/): Meta transforms:

Meta-transforms act on other transforms:

??? quote "<code>[cc.SequentialTransform](special/#cornucopia.special.SequentialTransform)(list[[Transform][cornucopia.base.Transform]])</code> <br />Apply multiple transforms in sequence"
    ::: cornucopia.special.SequentialTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomizedTransform](special/#cornucopia.special.RandomizedTransform)(type[[Transform][cornucopia.base.Transform]], tuple[[Sampler][cornucopia.random.Sampler]], dict[str, [Sampler][cornucopia.random.Sampler]]])</code> <br/>Randomize the input parameters of a transform"
    ::: cornucopia.special.RandomizedTransform
        options:
            heading_level: 3

??? quote "<code>[cc.SwitchTransform](special/#cornucopia.special.SwitchTransform)(list[[Transform][cornucopia.base.Transform]], prob: float = 0, shared: bool | str = False)</code> <br/>Apply one out of a set of transforms"
    ::: cornucopia.special.SwitchTransform
        options:
            heading_level: 3

??? quote "<code>[cc.MaybeTransform](special/#cornucopia.special.MaybeTransform)([Transform](base/#cornucopia.base.Transform), prob: float = 0.5, shared: bool | str = False)</code> <br/>Apply a transform with some probability"
    ::: cornucopia.special.MaybeTransform
        options:
            heading_level: 3

??? quote "<code>[cc.MappedTransform](special/#cornucopia.special.MappedTransform)(&ast;tuple[[Transform](base/#cornucopia.base.Transform)], &ast;&ast;dict[str, [Transform](base/#cornucopia.base.Transform)], nested: bool = False)</code> <br/>Apply different transforms to different tensors"
    ::: cornucopia.special.MappedTransform
        options:
            heading_level: 3

??? quote "<code>[cc.IncludeKeysTransform](special/#cornucopia.special.IncludeKeysTransform)([Transform](base/#cornucopia.base.Transform), keys: str | list[str])</code> <br/>Only apply a transform to a set of keys"
    ::: cornucopia.special.IncludeKeysTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ExcludeKeysTransform](special/#cornucopia.special.IncludeKeysTransform)([Transform](base/#cornucopia.base.Transform), keys: str | list[str])</code> <br/>Do not apply a transform to a set of keys"
    ::: cornucopia.special.ExcludeKeysTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ConsumeKeysTransform](special/#cornucopia.special.ConsumeKeysTransform)([Transform](base/#cornucopia.base.Transform), keys: str | list[str])</code> <br/>Consume (= do not return) a set of keys"
    ::: cornucopia.special.ConsumeKeysTransform
        options:
            heading_level: 3

??? quote "<code>[cc.SplitChannels](special/#cornucopia.special.SplitChannels)()</code> <br/>Transform multi-channel tensors into tuples of single-channel tensors"
    ::: cornucopia.special.SplitChannels
        options:
            heading_level: 3

??? quote "<code>[cc.CatChannels](special/#cornucopia.special.CatChannels)()</code> <br/>Concatenate tensors along the channel dimension"
    ::: cornucopia.special.CatChannels
        options:
            heading_level: 3

## [`cc.ctx`](ctx/): Context managers

Most meta-transforms can be used as context managers.
We define aliases for these meta-transforms under `cc.ctx`:

??? quote "<code>[cc.ctx.randomize](cts/#cornucopia.ctx.randomize) is [cc.RandomizedTransform](special/#cornucopia.special.RandomizedTransform)</code>"
    ::: cornucopia.ctx.randomize
        options:
            heading_level: 3

??? quote "<code>[cc.ctx.switch](cts/#cornucopia.ctx.switch) is [cc.SwitchTransform](special/#cornucopia.special.SwitchTransform)</code>"
    ::: cornucopia.ctx.switch
        options:
            heading_level: 3

??? quote "<code>[cc.ctx.maybe](cts/#cornucopia.ctx.maybe) is [cc.MaybeTransform](special/#cornucopia.special.MaybeTransform)</code>"
    ::: cornucopia.ctx.maybe
        options:
            heading_level: 3

??? quote "<code>[cc.ctx.map](cts/#cornucopia.ctx.map) is [cc.MappedTransform](special/#cornucopia.special.MappedTransform)</code>"
    ::: cornucopia.ctx.map
        options:
            heading_level: 3

??? quote "<code>[cc.ctx.include](cts/#cornucopia.ctx.include) is [cc.IncludeKeysTransform](special/#cornucopia.special.IncludeKeysTransform)</code>"
    ::: cornucopia.ctx.include
        options:
            heading_level: 3

??? quote "<code>[cc.ctx.exclude](cts/#cornucopia.ctx.exclude) is [cc.RandomizedTransform](special/#cornucopia.special.RandomizedTransform)</code>"
    ::: cornucopia.ctx.exclude
        options:
            heading_level: 3

??? quote "<code>[cc.ctx.consume](cts/#cornucopia.ctx.consume) is [cc.ExcludeKeysTransform](special/#cornucopia.special.ExcludeKeysTransform)</code>"
    ::: cornucopia.ctx.consume
        options:
            heading_level: 3

??? quote "<code>[cc.ctx.shared](cts/#cornucopia.ctx.shared) is [cc.RandomizedTransform](special/#cornucopia.special.RandomizedTransform)</code>"
    ::: cornucopia.ctx.shared
        options:
            heading_level: 3

??? quote "<code>[cc.ctx.returns](cts/#cornucopia.ctx.returns) is [cc.SharedTransform](special/#cornucopia.special.SharedTransform)</code>"
    ::: cornucopia.ctx.returns
        options:
            heading_level: 3

??? quote "<code>[cc.ctx.batch](cts/#cornucopia.ctx.batch) is [cc.BatchedTransform](special/#cornucopia.special.BatchedTransform)</code>"
    ::: cornucopia.ctx.batch
        options:
            heading_level: 3

## [`cc.io`](io/): Data loaders and converters

```python
cc.ToTensorTransform(ndim=None, dtype=None, device=None)
cc.LoadTransform(ndim=None, dtype=None, device=None)
```

## [`cc.fov`](fov/): Modify the field-of-view

```python
cc.FlipTransform(axis=None, shared=True)
cc.PatchTransform(shape=64, center=0, bound='dct2', shared=True)
cc.RandomPatchTransform(patch_size: int or list[int], bound='dct2', shared=True)
cc.CropTransform(cropping: int or float or list[int or float], unit='vox', side='both', shared=True)
cc.PadTransform(padding: int or float or list[int or float], unit='vox', side='both', shared=True)
cc.PowerTwoTransform(exponent=1, bound='dct2', shared='channels')
```

## [`cc.noise`](noise/): Inject noise

```python
cc.GaussianNoiseTransform(sigma=0.1, shared=False)
cc.ChiNoiseTransform(sigma=0.1, nb_channels=2, shared=False)
cc.GammaNoiseTransform(mean=1, sigma=0.1, shared=False)
cc.GFactorTransform(noise: Transform, shape=5, vmin=1, vmax=4)

# randomized
cc.RandomGaussianNoiseTransform(sigma=0.1, shared=False)
cc.RandomChiNoiseTransform(sigma=0.1, nb_channels=8, shared=False)
cc.RandomGammaNoiseTransform(mean=2, sigma=0.1, shared=False)
```

## [`cc.intensity`](intensity/): Modify image intensities

```python
cc.MultFieldTransform(shape=5, vmin=0, vmax=1, shared=False)
cc.AddFieldTransform(shape=5, vmin=0, vmax=1, shared=False)
cc.GlobalMultTransform(value=1, shared=False)
cc.GlobalAdditiveTransform(value=0, shared=False)
cc.GammaTransform(gamma=1, vmin=None, vmax=None, shared=False)
cc.ZTransform(shared=False)
cc.QuantileTransform(pmin=0.01, pmax=0.99, vmin=0, vmax=1, clamp=False, shared=False)

# randomized
cc.RandomMultFieldTransform(shape=8, vmax=2, shared=False)
cc.RandomAddFieldTransform(shape=8, vmin=-1, vmax=1, shared=False)
cc.RandomGlobalMultTransform(value=(0.5, 2), shared=True)
cc.RandomGlobalAdditiveTransform(value=1, shared=True)
cc.RandomGammaTransform(gamma=(0.5, 2), vmin=None, vmax=None, shared=False)
```

## [`cc.contrast`](contrast/): Modify image contrast

```python
cc.ContrastMixtureTransform(nk=16, keep_background=True, shared='channels')
cc.ContrastLookupTransform(nk=16, keep_background=True, shared=False)
```

## [`cc.psf`](psf/): Modify point-spread function (or resolution)

```python
cc.SmoothTransform(fwhm=1)
cc.LowResTransform(resolution=2, noise: Transform = None)
cc.LowResSliceTransform(resolution=3, thickness=0.8, axis=-1, noise: Transform = None)

# randomized
cc.RandomSmoothTransform(fwhm=2)
cc.RandomLowResTransform(resolution=2, noise: Transform = None)
cc.RandomLowResSliceTransform(resolution=3, thickness=0.1, axis=None, noise: Transform = None)
```

## [`cc.geometric`](geometric/): Geometric transformations

```python
cc.ElasticTransform(dmax=0.1, unit='fov', shape=5, bound='border', steps=0, shared=True)
cc.AffineTransform(translations=0, rotations=0, shears=0, zooms=0,
                   unit='fov', bound='border', shared=True)
cc.AffineElasticTransform(dmax=0.1, shape=5, steps=0,
                          translations=0, rotations=0, shears=0, zooms=0,
                          unit='fov', bound='border', patch=None, shared=True)
cc.Slicewise3DAffineTransform(translations=0, rotations=0, shears=0, zooms=0,
                              slice=-1, unit='fov', bound='border', shared=True)

# randomized
cc.RandomElasticTransform(dmax=0.15, shape=10, unit='fov', bound='border', steps=0, shared=True)
cc.RandomAffineTransform(translations=0.1, rotations=15, shears=0.012, zooms=0.15,
                         unit='fov', bound='border', shared=True)
cc.RandomAffineElasticTransform(dmax=0.15, shape=10, steps=0,
                                translations=0.1, rotations=15, shears=0.012, zooms=0.15,
                                unit='fov', bound='border', patch=None, shared=True):
cc.RandomSlicewise3DAffineTransform(translations=0.1, rotations=15,
                                    shears=0, zooms=0, slice=-1, shots=2, nodes=8,
                                    unit='fov', bound='border', shared=True)
```

## [`cc.labels`](labels/): Transforms that act on label maps

```python
cc.OneHotTransform(label_map=None, label_ref=None, keep_background=True, dtype=None)
cc.ArgMaxTransform()
cc.GaussianMixtureTransform(mu=None, sigma=None, fwhm=0, background=None, shared=False)
cc.SmoothLabelMap(nb_classes=2, shape=5, soft=False, shared=False)
cc.ErodeLabelMap(labels=tuple(), radius=3, method='conv')
cc.DilateLabelMap(labels=tuple(), radius=3, method='conv')
cc.SmoothMorphoLabelTransform(labels=tuple(), min_radius=-3, max_radius=3, shape=5, method='conv')
cc.SmoothShallowLabelTransform(labels=tuple(), max_width=5, min_width=1, shape=5,
                               background_labels=tuple(), method='l2', shared=False)
cc.BernoulliTransform(prob=0.1, shared=False)
cc.SmoothBernoulliTransform(prob=0.1, shape=5, shared=False)

# randomized
cc.RandomGaussianMixtureTransform(mu=255, sigma=16, fwhm=2, background=None, shared='channels')
cc.RandomErodeLabelTransform(labels=0.5, radius=3, method='conv', shared=False)
cc.RandomDilateLabelTransform(labels=0.5, radius=3, method='conv', shared=False)
cc.RandomSmoothMorphoLabelTransform(labels=0.5, min_radius=-3, max_radius=3,
                                    shape=5, method='conv', shared=False)
cc.RandomSmoothShallowLabelTransform(labels=0.5, max_width=5, min_width=1, shape=5,
                                     background_labels=tuple(), method='l2', shared=False)

```

## [`cc.kspace`](kspace/): Transforms that act on k-space (Fourier domain)
```python
cc.ArrayCoilTransform(ncoils=8, fwhm=0.5, diameter=0.8, jitter=0.01,
                      unit='fov', shape=4, sos=True, shared=True)
cc.SumOfSquaresTransform()
cc.IntraScanMotionTransform(shots=4, axis=-1, freq=True, pattern='sequential',
                 translations=0.1, rotations=15, sos=True, coils=None, shared='channels')
cc.SmallIntraScanMotionTransform(translations=0.05, rotations=5, axis=-1, shared='channels')
```

## [`cc.synth`](synth/): Synthesize images (domain randomization)

```python
cc.SynthContrastTransform(...)
cc.SynthFromLabelTransform(patch=None, from_disk=False, one_hot=False,
                           synth_labels=None, synth_labels_maybe=None, target_labels=None,
                           ...)
```
