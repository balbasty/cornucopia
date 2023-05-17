# API overview

### Meta transforms
```python
cc.SequentialTransform(transforms: List[Transform])
cc.MaybeTransform(transform: Transform, prob=0.5, shared=False)
cc.SwitchTransform(transforms: List[Transform], prob=0, shared=False)
cc.RandomizedTransform(transform_class, sample, ksample=None, shared=False)
cc.MappedTransform(*transforms, **mapped_transforms, nested=False, default=None)
cc.MappedKeysTransform(transform: Transform, keys: str or list[str])
cc.MappedExceptKeysTransform(transform: Transform, keys: str or list[str])
cc.SplitChannels()
cc.CatChannels()

# helpers
cc.switch(dict[Transform -> float]) -> SwitchTransform
cc.randomize(transform_class, shared=False)(*args, **kwargs) -> RandomizedTransform
cc.map(*args: Transforms, **kwargs: Transforms, nested=False, default=None) -> MappedTransform
cc.include_keys(transform: Transform, keys: str or list[str]) -> MappedKeysTransform
cc.exclude_keys(transform: Transform, keys: str or list[str]) -> MappedExceptKeysTransform
cc.batch(transform: Transform) -> BatchedTransform
```

### I/O
```python
cc.ToTensorTransform(dim=None, dtype=None, device=None)
cc.LoadTransform(dim=None, dtype=None, device=None)
```

### Field-of-view
```python
cc.FlipTransform(axis=None, shared=True)
cc.PatchTransform(shape=64, center=0, bound='dct2', shared=True)
cc.RandomPatchTransform(patch_size: int or list[int], bound='dct2', shared=True)
cc.CropTransform(cropping: int or float or list[int or float], unit='vox', side='both', shared=True)
cc.PadTransform(padding: int or float or list[int or float], unit='vox', side='both', shared=True)
cc.PowerTwoTransform(exponent=1, bound='dct2', shared='channels')
```

### Noise
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

### Intensity
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

### Contrast
```python
cc.ContrastMixtureTransform(nk=16, keep_background=True, shared='channels')
cc.ContrastLookupTransform(nk=16, keep_background=True, shared=False)
```

### Point-spread function
```python
cc.SmoothTransform(fwhm=1)
cc.LowResTransform(resolution=2, noise: Transform = None)
cc.LowResSliceTransform(resolution=3, thickness=0.8, axis=-1, noise: Transform = None)

# randomized
cc.RandomSmoothTransform(fwhm=2)
cc.RandomLowResTransform(resolution=2, noise: Transform = None)
cc.RandomLowResSliceTransform(resolution=3, thickness=0.1, axis=None, noise: Transform = None)
```

### Geometric
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

### Labels
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

### k-space
```python
cc.ArrayCoilTransform(ncoils=8, fwhm=0.5, diameter=0.8, jitter=0.01,
                      unit='fov', shape=4, sos=True, shared=True)
cc.SumOfSquaresTransform()
cc.IntraScanMotionTransform(shots=4, axis=-1, freq=True, pattern='sequential',
                 translations=0.1, rotations=15, sos=True, coils=None, shared='channels')
cc.SmallIntraScanMotionTransform(translations=0.05, rotations=5, axis=-1, shared='channels')
```

### Synth
```python
cc.SynthContrastTransform(...)
cc.SynthFromLabelTransform(patch=None, from_disk=False, one_hot=False,
                           synth_labels=None, synth_labels_maybe=None, target_labels=None,
                           ...)
```