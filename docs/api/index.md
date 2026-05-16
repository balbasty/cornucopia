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

??? quote "<code>[cc.RandomizedTransform](special/#cornucopia.special.RandomizedTransform)(type[[Transform][cornucopia.base.Transform]], tuple[[Sampler][cornucopia.random.Sampler]], dict[str, [Sampler][cornucopia.random.Sampler]])</code> <br/>Randomize the input parameters of a transform"
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

??? quote "<code>[cc.ToTensorTransform](io/#cornucopia.io.ToTensorTransform)(ndim: int | None, dtype: [dtype][torch.dtype] | None, device: [device][torch.device] | None)</code> <br/>Ensure that an array is a tensor (with required properties)"
    ::: cornucopia.io.ToTensorTransform
        options:
            heading_level: 3

??? quote "<code>[cc.LoadTransform](io/#cornucopia.io.LoadTransform)(ndim: int | None, dtype: [dtype][torch.dtype] | None, *, device: [device][torch.device] | None)</code> <br/>Load a tensor from disk"
    ::: cornucopia.io.LoadTransform
        options:
            heading_level: 3

## [`cc.fov`](fov/): Modify the field-of-view

### Deterministic transforms

??? quote "<code>[cc.FlipTransform](fov/#cornucopia.fov.FlipTransform)(axis: int | list[int] | None = None)</code> <br/>Flip one or more axes"
    ::: cornucopia.fov.FlipTransform
        options:
            heading_level: 3

??? quote "<code>[cc.PermuteAxesTransform](fov/#cornucopia.fov.PermuteAxesTransform)(permutation: int | list[int] | None = None)</code> <br/>Permute axes"
    ::: cornucopia.fov.PermuteAxesTransform
        options:
            heading_level: 3

??? quote "<code>[cc.Rot90Transform](fov/#cornucopia.fov.Rot90Transform)(axis: int | list[int] = 0, negative: bool | list[bool] = False)</code> <br/>Apply a 90 rotation along one or several axes"
    ::: cornucopia.fov.Rot90Transform
        options:
            heading_level: 3

??? quote "<code>[cc.Rot180Transform](fov/#cornucopia.fov.Rot180Transform)(axis: int | list[int] = 0)</code> <br/>Apply a 180 rotation along one or several axes"
    ::: cornucopia.fov.Rot180Transform
        options:
            heading_level: 3

??? quote "<code>[cc.CropPadTransform](fov/#cornucopia.fov.CropPadTransform)(crop: list[slice] = (), pad: list[int] = (), ...)</code> <br/>Crop and/or pad a tensor"
    ::: cornucopia.fov.CropPadTransform
        options:
            heading_level: 3

??? quote "<code>[cc.CropTransform](fov/#cornucopia.fov.CropTransform)(cropping: number | list[number], unit: {'fov','vox'} = 'vox', side: {'pre','post','both'} = 'both')</code> <br/>Crop a tensor"
    ::: cornucopia.fov.CropTransform
        options:
            heading_level: 3

??? quote "<code>[cc.PadTransform](fov/#cornucopia.fov.PadTransform)(padding: number | list[number], unit: {'fov','vox'} = 'vox', side: {'pre','post','both'} = 'both', ...)</code> <br/>Pad a tensor"
    ::: cornucopia.fov.PadTransform
        options:
            heading_level: 3

??? quote "<code>[cc.PatchTransform](fov/#cornucopia.fov.PatchTransform)(shape: int | list[int] = 64, center: float | list[float] = 0, ...)</code> <br/>Extract a patch from the tensor"
    ::: cornucopia.fov.PatchTransform
        options:
            heading_level: 3

??? quote "<code>[cc.PowerTwoTransform](fov/#cornucopia.fov.PowerTwoTransform)(exponent: int | list[int] = 1, ...)</code> <br/>Pad the volume such that the tensor shape can be divided by 2**x"
    ::: cornucopia.fov.PowerTwoTransform
        options:
            heading_level: 3

### Random transforms


??? quote "<code>[cc.RandomFlipTransform](fov/#cornucopia.fov.RandomFlipTransform)(axes: [Sampler](random/cornucopia.random.Sampler) | int | list[int] | None = None, *, shared: bool | str = True)</code> <br/>Randomly flip one or more axes"
    ::: cornucopia.fov.RandomFlipTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomPermuteAxesTransform](fov/#cornucopia.fov.RandomPermuteAxesTransform)(permutation: list[int] | None = None, *, shared: bool | str = True)</code> <br/>Randomly permute axes"
    ::: cornucopia.fov.RandomPermuteAxesTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomRot90Transform](fov/#cornucopia.fov.RandomRot90Transform)(axes: int | list[int] | None = None, max_rot: [Sampler](random/cornucopia.random.Sampler) | int = 2, negative: bool | list[bool] = False, *, shared: bool | str = True)</code> <br/>Random set of 90 transforms"
    ::: cornucopia.fov.RandomRot90Transform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomPatchTransform](fov/#cornucopia.fov.RandomPatchTransform)(shape: int | list[int], *, shared: bool | str = True)</code> <br/>Extract a random patch from the tensor"
    ::: cornucopia.fov.RandomPatchTransform
        options:
            heading_level: 3

## [`cc.noise`](noise/): Inject noise

### Deterministic parameters

??? quote "<code>[cc.GaussianNoiseTransform](noise/#cornucopia.noise.GaussianNoiseTransform)(sigma: float | list[float] = 0.1, *, shared: bool | str = False)</code> <br/>Inject Gaussian noise"
    ::: cornucopia.noise.GaussianNoiseTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ChiNoiseTransform](noise/#cornucopia.noise.ChiNoiseTransform)(sigma: float | list[float] = 0.1, nb_channels: int = 2, *, shared: bool | str = False)</code> <br/>Inject non-central Chi noise"
    ::: cornucopia.noise.ChiNoiseTransform
        options:
            heading_level: 3

??? quote "<code>[cc.GammaNoiseTransform](noise/#cornucopia.noise.GammaNoiseTransform)(sigma: float | list[float] = 0.1, mean: float | list[float] = 1, *, shared: bool | str = False)</code> <br/>Inject Gamma noise"
    ::: cornucopia.noise.GammaNoiseTransform
        options:
            heading_level: 3

??? quote "<code>[cc.GFactorTransform](noise/#cornucopia.noise.GFactorTransform)(noise: [Transform](base/#cornucopia.base.Transform), shape: int = 5, vmin: float = 0.5, vmax: float = 1.5, order: int = 3, *, shared: bool | str = False)</code> <br/>Inject noise with spatially varying variance"
    ::: cornucopia.noise.GFactorTransform
        options:
            heading_level: 3

### Random parameters

??? quote "<code>[cc.RandomGaussianNoiseTransform](noise/#cornucopia.noise.RandomGaussianNoiseTransform)(sigma: [Sampler](random/#cornucopia.random.Sampler) | float = 0.1, *, shared: bool | str = False)</code> <br/>Inject Gaussian noise with random parameters"
    ::: cornucopia.noise.RandomGaussianNoiseTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomChiNoiseTransform](noise/#cornucopia.noise.RandomChiNoiseTransform)(sigma: [Sampler](random/#cornucopia.random.Sampler) | float = 0.1, nb_channels: [Sampler](random/#cornucopia.random.Sampler) | int = 8, *, shared: bool | str = False)</code> <br/>Inject non-central Chi noise with random parameters"
    ::: cornucopia.noise.RandomChiNoiseTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomGammaNoiseTransform](noise/#cornucopia.noise.RandomGammaNoiseTransform)(sigma: [Sampler](random/#cornucopia.random.Sampler) | float = 0.1, mean: [Sampler](random/#cornucopia.random.Sampler) | float = [Fixed](random/#cornucopia.random.Fixed)(1), *, shared: bool | str = False)</code> <br/>Inject Gamma noise with random parameters"
    ::: cornucopia.noise.RandomGammaNoiseTransform
        options:
            heading_level: 3

## [`cc.intensity`](intensity/): Modify image intensities

### Deterministic transforms

??? quote "<code>[cc.AddValueTransform](intensity/#cornucopia.intensity.AddValueTransform)(value: float | list[float] | tensor)</code> <br/>Add a constant value"
    ::: cornucopia.intensity.AddValueTransform
        options:
            heading_level: 3

??? quote "<code>[cc.MulValueTransform](intensity/#cornucopia.intensity.MulValueTransform)(value: float | list[float] | tensor)</code> <br/>Multiply by a constant value"
    ::: cornucopia.intensity.MulValueTransform
        options:
            heading_level: 3

??? quote "<code>[cc.FillValueTransform](intensity/#cornucopia.intensity.FillValueTransform)(mask: tensor, value: float | list[float] | tensor)</code> <br/>Fill with a constant value"
    ::: cornucopia.intensity.FillValueTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ReturnValueTransform](intensity/#cornucopia.intensity.ReturnValueTransform)(value: float | list[float] | tensor)</code> <br/>Return a constant value"
    ::: cornucopia.intensity.ReturnValueTransform
        options:
            heading_level: 3

??? quote "<code>[cc.AddMulTransform](intensity/#cornucopia.intensity.AddMulTransform)(slope: float | list[float] | tensor = 1, offset: float | list[float] | tensor = 0)</code> <br/>Element wise affine transform"
    ::: cornucopia.intensity.AddMulTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ClipTransform](intensity/#cornucopia.intensity.ClipTransform)(vmin: float | list[float] | tensor | None = None, vmax: float | list[float] | tensor | None = None)</code> <br/>Clip extremum values"
    ::: cornucopia.intensity.ClipTransform
        options:
            heading_level: 3

??? quote "<code>[cc.SplineUpsampleTransform](intensity/#cornucopia.intensity.SplineUpsampleTransform)(order: int = 3, prefilter: bool = False)</code> <br/>Upsample a field using spline interpolation"
    ::: cornucopia.intensity.SplineUpsampleTransform
        options:
            heading_level: 3

??? quote "<code>[cc.GammaTransform](intensity/#cornucopia.intensity.GammaTransform)(gamma: float = 1, vmin: float = 0, vmax: float = 1)</code> <br/>Gamma correction"
    ::: cornucopia.intensity.GammaTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ZTransform](intensity/#cornucopia.intensity.ZTransform)(mu: float = 0, sigma: float = 1, *, shared: bool | str = False)</code> <br/>Gamma correction"
    ::: cornucopia.intensity.ZTransform
        options:
            heading_level: 3

??? quote "<code>[cc.QuantileTransform](intensity/#cornucopia.intensity.QuantileTransform)(pmin: float = 0.01, pmax: float = 0.99, vmin: float = 0, vmax: float = 1, clip: bool = False, *, shared: bool | str = False)</code> <br/>Match lower and upper quantiles to (0, 1)"
    ::: cornucopia.intensity.QuantileTransform
        options:
            heading_level: 3

??? quote "<code>[cc.MinMaxTransform](intensity/#cornucopia.intensity.MinMaxTransform)(vmin: float = 0, vmax: float = 1, clip: bool = False, *, shared: bool | str = False)</code> <br/>Match min and max values to (0, 1)"
    ::: cornucopia.intensity.MinMaxTransform
        options:
            heading_level: 3

### Transforms with fixed parameters (but random coefficients)

??? quote "<code>[cc.AddFieldTransform](intensity/#cornucopia.intensity.AddFieldTransform)(shape: int | list[int] = 5, vmin: float = 0, vmax: float = 1, order: int = 3, ...)</code> <br/>Smooth additive field"
    ::: cornucopia.intensity.AddFieldTransform
        options:
            heading_level: 3

??? quote "<code>[cc.MulFieldTransform](intensity/#cornucopia.intensity.MulFieldTransform)(shape: int | list[int] = 5, vmin: float = 0, vmax: float = 1, order: int = 3, ...)</code> <br/>Smooth multiplicative field"
    ::: cornucopia.intensity.MulFieldTransform
        options:
            heading_level: 3

### Transforms with random parameters

??? quote "<code>[cc.RandomAddTransform](intensity/#cornucopia.intensity.RandomAddTransform)(value: [Sampler](random/#cornucopia.random.Sampler) | float | tuple[float, float] = 1, *, shared: bool = False)</code> <br/>Add a random value"
    ::: cornucopia.intensity.RandomAddTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomMulTransform](intensity/#cornucopia.intensity.RandomMulTransform)(value: [Sampler](random/#cornucopia.random.Sampler) | float | tuple[float, float] = (0.5, 2), *, bool | str: bool = False)</code> <br/>Multiply by a random value"
    ::: cornucopia.intensity.RandomMulTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomAddMulTransform](intensity/#cornucopia.intensity.RandomAddMulTransform)(slope: [Sampler](random/#cornucopia.random.Sampler) | float | tuple[float, float] = 1, offset: [Sampler](random/#cornucopia.random.Sampler) | float | tuple[float, float] = 0.5, *, shared: bool | str = False)</code> <br/>Random element-wise affine transform"
    ::: cornucopia.intensity.RandomAddMulTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomAddFieldTransform](intensity/#cornucopia.intensity.RandomAddFieldTransform)(shape: [Sampler](random/#cornucopia.random.Sampler) | int = 8, vmin: [Sampler](random/#cornucopia.random.Sampler) | float = -1, vmax: [Sampler](random/#cornucopia.random.Sampler) | float = 1, order: int = 3, *, shared: bool | str = False)</code> <br/>Random smooth multiplicative field"
    ::: cornucopia.intensity.RandomAddFieldTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomMulFieldTransform](intensity/#cornucopia.intensity.RandomMulFieldTransform)(shape: [Sampler](random/#cornucopia.random.Sampler) | int = 8, vmax: [Sampler](random/#cornucopia.random.Sampler) | float = 1, order: int = 3, symmetric: bool | float = False, *, shared: bool | str = False)</code> <br/>Random smooth multiplicative field"
    ::: cornucopia.intensity.RandomMulFieldTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomSlicewiseMulFieldTransform](intensity/#cornucopia.intensity.RandomSlicewiseMulFieldTransform)(shape: [Sampler](random/#cornucopia.random.Sampler) | int = 8, vmax: [Sampler](random/#cornucopia.random.Sampler) | float = 1, order: int = 3, slice: int | None = None, thickness: [Sampler](random/#cornucopia.random.Sampler) | int = 32,shape_through: [Sampler](random/#cornucopia.random.Sampler) | int | None = None, *, shared: bool | str = False)</code> <br/>Random smooth slicewise multiplicative field"
    ::: cornucopia.intensity.RandomSlicewiseMulFieldTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomGammaTransform](intensity/#cornucopia.intensity.RandomGammaTransform)(gamma: [Sampler](random/#cornucopia.random.Sampler) | float | tuple[float, float] = (0.5, 2), *, shared: bool | str = False)</code> <br/>Gamma correction"
    ::: cornucopia.intensity.RandomGammaTransform
        options:
            heading_level: 3

## [`cc.contrast`](contrast/): Modify image contrast

??? quote "<code>[cc.ContrastMixtureTransform](intensity/#cornucopia.contrast.ContrastMixtureTransform)(nk: int = 16, keep_background: bool = True, *, shared: bool = False)</code> <br/>Change the means and covariances of intensity modes"
    ::: cornucopia.contrast.ContrastMixtureTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ContrastLookupTransform](intensity/#cornucopia.contrast.ContrastMixtureTransform)(nk: int = 16, keep_background: bool = True, *, shared: bool = False)</code> <br/>Segment intensities into equidistant bins and change their mean value"
    ::: cornucopia.contrast.ContrastMixtureTransform
        options:
            heading_level: 3

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
