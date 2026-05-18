---
icon: fontawesome/brands/python
---

# API

## [`cc.special`](special/): Meta transforms: {#special}

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

## [`cc.ctx`](ctx/): Context managers {#ctx}

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

## [`cc.io`](io/): Data loaders and converters {#io}

??? quote "<code>[cc.ToTensorTransform](io/#cornucopia.io.ToTensorTransform)(ndim: int | None, dtype: [dtype][torch.dtype] | None, device: [device][torch.device] | None)</code> <br/>Ensure that an array is a tensor (with required properties)"
    ::: cornucopia.io.ToTensorTransform
        options:
            heading_level: 3

??? quote "<code>[cc.LoadTransform](io/#cornucopia.io.LoadTransform)(ndim: int | None, dtype: [dtype][torch.dtype] | None, *, device: [device][torch.device] | None)</code> <br/>Load a tensor from disk"
    ::: cornucopia.io.LoadTransform
        options:
            heading_level: 3

## [`cc.fov`](fov/): Modify the field-of-view {#fov}

### Deterministic transforms {#fov-deterministic}

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

### Random transforms {#fov-random}


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

## [`cc.noise`](noise/): Inject noise {#noise}

### Deterministic parameters {#noise-deterministic}

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

### Random parameters {#noise-random}

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

## [`cc.intensity`](intensity/): Modify image intensities {#intensity}

### Deterministic transforms {#intensity-deterministic}

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

### Transforms with fixed parameters (but random coefficients) {#intensity-fixed}

??? quote "<code>[cc.AddFieldTransform](intensity/#cornucopia.intensity.AddFieldTransform)(shape: int | list[int] = 5, vmin: float = 0, vmax: float = 1, order: int = 3, ...)</code> <br/>Smooth additive field"
    ::: cornucopia.intensity.AddFieldTransform
        options:
            heading_level: 3

??? quote "<code>[cc.MulFieldTransform](intensity/#cornucopia.intensity.MulFieldTransform)(shape: int | list[int] = 5, vmin: float = 0, vmax: float = 1, order: int = 3, ...)</code> <br/>Smooth multiplicative field"
    ::: cornucopia.intensity.MulFieldTransform
        options:
            heading_level: 3

### Transforms with random parameters {#intensity-random}

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

## [`cc.contrast`](contrast/): Modify image contrast {#contrast}

??? quote "<code>[cc.ContrastMixtureTransform](intensity/#cornucopia.contrast.ContrastMixtureTransform)(nk: int = 16, keep_background: bool = True, *, shared: bool = False)</code> <br/>Change the means and covariances of intensity modes"
    ::: cornucopia.contrast.ContrastMixtureTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ContrastLookupTransform](intensity/#cornucopia.contrast.ContrastMixtureTransform)(nk: int = 16, keep_background: bool = True, *, shared: bool = False)</code> <br/>Segment intensities into equidistant bins and change their mean value"
    ::: cornucopia.contrast.ContrastMixtureTransform
        options:
            heading_level: 3

## [`cc.psf`](psf/): Modify point-spread function (or resolution) {#psf}

### Deterministic transforms {#psf-deterministic}

??? quote "<code>[cc.SmoothTransform](intensity/#cornucopia.psf.SmoothTransform)(fwhm: float | list[float] = 1)</code> <br/>Apply Gaussian smoothing"
    ::: cornucopia.psf.SmoothTransform
        options:
            heading_level: 3

??? quote "<code>[cc.LowResSliceTransform](intensity/#cornucopia.psf.LowResSliceTransform)(resolution: float = 3, thickness: float = 0.8, axis: int = -1, noise: [Transform](base/#cornucopia.base.Transform) | None = None)</code> <br/>Model a low-resolution slice direction, with Gaussian profile"
    ::: cornucopia.psf.LowResSliceTransform
        options:
            heading_level: 3

??? quote "<code>[cc.LowResTransform](intensity/#cornucopia.psf.LowResSliceTransform)(resolution: float | list[float] = 2, noise: [Transform](base/#cornucopia.base.Transform) | None = None)</code> <br/>Model a lower-resolution image"
    ::: cornucopia.psf.LowResSliceTransform
        options:
            heading_level: 3

### Random transforms {#psf-random}

??? quote "<code>[cc.RandomSmoothTransform](intensity/#cornucopia.psf.RandomSmoothTransform)(fwhm: [Sampler](random/#cornucopia.random.Sampler) | float = 2, *, shared: bool | str = False)</code> <br/>Apply Gaussian smoothing with random width"
    ::: cornucopia.psf.RandomSmoothTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomLowResSliceTransform](intensity/#cornucopia.psf.RandomLowResSliceTransform)(resolution: [Sampler](random/#cornucopia.random.Sampler) | float = 3, thickness: [Sampler](random/#cornucopia.random.Sampler) | float = 0.1, axis: [Sampler](random/#cornucopia.random.Sampler) | int | None = None, noise: [Transform](base/#cornucopia.base.Transform) | None = None, *, shared: bool | str = False)</code> <br/>Random low-resolution slice direction"
    ::: cornucopia.psf.RandomLowResSliceTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomLowResTransform](intensity/#cornucopia.psf.RandomLowResTransform)(resolution: [Sampler](random/#cornucopia.random.Sampler) | float = 2, noise: [Transform](base/#cornucopia.base.Transform) | None = None, *, shared: bool | str = False)</code> <br/>Random lower-resolution image"
    ::: cornucopia.psf.RandomLowResTransform
        options:
            heading_level: 3

## [`cc.geometric`](geometric/): Geometric transformations {#geometric}

### Deterministic transforms {#geometric-deterministic}

??? quote "<code>[cc.ApplyAffineTransform](intensity/#cornucopia.geometric.ApplyAffineTransform)(matrix: tensor | None = None, flow: tensor | None = None, ...)</code> <br/>Apply an affine matrix (or precomputed flow field)"
    ::: cornucopia.geometric.ApplyAffineTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ApplySlicewiseAffineTransform](intensity/#cornucopia.geometric.ApplySlicewiseAffineTransform)(matrix: tensor, flow: tensor | None = None, ...)</code> <br/>Apply a slice-wise affine transformation"
    ::: cornucopia.geometric.ApplySlicewiseAffineTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ApplyElasticTransform](intensity/#cornucopia.geometric.ApplyElasticTransform)(flow: tensor | None = None, controls: tensor | None = None, ...)</code> <br/>Apply an elastic tranform"
    ::: cornucopia.geometric.ApplyElasticTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ApplyAffineElasticTransform](intensity/#cornucopia.geometric.ApplyAffineElasticTransform)(flow: tensor | None = None, controls: tensor | None = None, affine: tensor | None = None, ...)</code> <br/>Apply an affine + elastic tranform"
    ::: cornucopia.geometric.ApplyAffineElasticTransform
        options:
            heading_level: 3

??? quote "<code>[cc.AffineTransform](intensity/#cornucopia.geometric.AffineTransform)(translations: vector_like[float] = 0, rotations: vector_like[float] = 0, shears: vector_like[float] = 0, zooms: vector_like[float] = 0)</code> <br/>Apply an affine transformation, encoded by its parameters"
    ::: cornucopia.geometric.AffineTransform
        options:
            heading_level: 3

??? quote "<code>[cc.SlicewiseAffineTransform](intensity/#cornucopia.geometric.SlicewiseAffineTransform)(translations: list[float | list[float]] = 0, rotations: list[float | list[float]] = 0, shears: list[float | list[float]] = 0, zooms: list[float | list[float]] = 0)</code> <br/>Apply a slice-wise affine transformation"
    ::: cornucopia.geometric.SlicewiseAffineTransform
        options:
            heading_level: 3

### Transforms with fixed parameters (but random coefficients) {#geometric-fixed}

??? quote "<code>[cc.ElasticTransform](intensity/#cornucopia.geometric.ElasticTransform)(dmax: vector_like[float] = 0.1, unit: {'fov','vox'} = 'fov', shape: int | list[int] = 5, ...)</code> <br/>Sample a smooth elastic deformation."
    ::: cornucopia.geometric.ElasticTransform
        options:
            heading_level: 3

??? quote "<code>[cc.AffineElasticTransform](intensity/#cornucopia.geometric.AffineElasticTransform)(dmax: vector_like[float] = 0.1, shape: int | list[int] = 5, steps: int = 0, translations: vector_like[float] = 0, rotations: vector_like[float] = 0, shears: vector_like[float] = 0, zooms: vector_like[float] = 0, ...)</code> <br/>Apply an affine + elastic transformation."
    ::: cornucopia.geometric.AffineElasticTransform
        options:
            heading_level: 3

### Transforms with random parameters {#geometric-random}

??? quote "<code>[cc.RandomAffineTransform](intensity/#cornucopia.geometric.RandomAffineTransform)(translations: [Sampler](random/#cornucopia.random.Sampler) | vector_like[float] = 0.1, rotations: [Sampler](random/#cornucopia.random.Sampler) | vector_like[float] = 15, shears: [Sampler](random/#cornucopia.random.Sampler) | vector_like[float] = 0.012, [Sampler](random/#cornucopia.random.Sampler) | zooms: vector_like[float] = 0.15, ..., *, shared: bool | str = True)</code> <br/>Apply a random affine transformation"
    ::: cornucopia.geometric.RandomAffineTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomSlicewiseAffineTransform](intensity/#cornucopia.geometric.RandomSlicewiseAffineTransform)(translations: list[float | list[float]] = 0, rotations: list[float | list[float]] = 0, shears: list[float | list[float]] = 0, zooms: list[float | list[float]] = 0)</code> <br/>Apply a random slice-wise affine transformation"
    ::: cornucopia.geometric.RandomSlicewiseAffineTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomElasticTransform](intensity/#cornucopia.geometric.RandomElasticTransform)(dmax: [Sampler](random/#cornucopia.random.Sampler) | vector_like[float] = 0.1, shape: [Sampler](random/#cornucopia.random.Sampler) | int | list[int] = 5, unit: {'fov','vox'} = 'fov', ...)</code> <br/>Apply a random elsstic transformation"
    ::: cornucopia.geometric.RandomElasticTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomAffineElasticTransform](intensity/#cornucopia.geometric.RandomAffineElasticTransform)([Sampler](random/#cornucopia.random.Sampler) | dmax: vector_like[float] = 0.1, shape: [Sampler](random/#cornucopia.random.Sampler) | int | list[int] = 5, steps: int = 0, translations: [Sampler](random/#cornucopia.random.Sampler) | vector_like[float] = 0, rotations: [Sampler](random/#cornucopia.random.Sampler) | vector_like[float] = 0, shears: [Sampler](random/#cornucopia.random.Sampler) | vector_like[float] = 0, zooms: [Sampler](random/#cornucopia.random.Sampler) | vector_like[float] = 0, ...)</code> <br/>Apply an affine + elastic transformation."
    ::: cornucopia.geometric.RandomAffineElasticTransform
        options:
            heading_level: 3

## [`cc.labels`](labels/): Transforms that act on label maps {#labels}

### Deterministic transforms {#labels-deterministic}

??? quote "<code>[cc.OneHotTransform](intensity/#cornucopia.labels.OneHotTransform)(label_map: list[int | str | list[int | str]] | None = None, label_ref: dict[str, int] | None = None, ...)</code> <br/>Transform a volume of integer labels into a one-hot representation."
    ::: cornucopia.labels.OneHotTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ArgMaxTransform](intensity/#cornucopia.labels.ArgMaxTransform)()</code> <br/>Take the argmax along the channel dimension."
    ::: cornucopia.labels.ArgMaxTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RelabelTransform](intensity/#cornucopia.labels.RelabelTransform)(labels: list[int | list[int]] | None = None)</code> <br/>Relabel a label map."
    ::: cornucopia.labels.RelabelTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ErodeLabelTransform](intensity/#cornucopia.labels.ErodeLabelTransform)(labels: int | list[int] = (), radius: int | list[int] = 3, method: {'conv','l1','l2'} = 'conv', ...)</code> <br/>Morphological erosion."
    ::: cornucopia.labels.ErodeLabelTransform
        options:
            heading_level: 3

??? quote "<code>[cc.DilateLabelTransform](intensity/#cornucopia.labels.DilateLabelTransform)(labels: int | list[int] = (), radius: int | list[int] = 3, method: {'conv','l1','l2'} = 'conv', ...)</code> <br/>Morphological erosion."
    ::: cornucopia.labels.DilateLabelTransform
        options:
            heading_level: 3

### Transforms with fixed parameters (but random coefficients) {#labels-fixed}

??? quote "<code>[cc.GaussianMixtureTransform](intensity/#cornucopia.labels.GaussianMixtureTransform)(mu: vector_like[float] | None = None, sigma: vector_like[float] | None = None, fwhm: vector_like[float] = 0, ...)</code> <br/>Sample from a Gaussian mixture."
    ::: cornucopia.labels.GaussianMixtureTransform
        options:
            heading_level: 3

??? quote "<code>[cc.SmoothLabelMap](intensity/#cornucopia.labels.SmoothLabelMap)(nb_classes: int = 2, shape: int | list[int] = 5, soft: bool = True, *, shared: bool | str = False)</code> <br/>Generate a random label map."
    ::: cornucopia.labels.SmoothLabelMap
        options:
            heading_level: 3

??? quote "<code>[cc.SmoothMorphoLabelTransform](intensity/#cornucopia.labels.SmoothMorphoLabelTransform)(labels: int | list[int] = (), min_radius: int | list[int] = -3, max_radius: int | list[int] = 3, shape: int | list[int] = 5, conv: {'conv','l1','l2'} = 'conv')</code> <br/>Morphological erosion/dilation with spatially varying radius."
    ::: cornucopia.labels.SmoothMorphoLabelTransform
        options:
            heading_level: 3

??? quote "<code>[cc.SmoothShallowLabelTransform](intensity/#cornucopia.labels.SmoothShallowLabelTransform)(labels: int | list[int] = (), max_width: int | list[int] = 5, min_width: int | list[int] = 1, shape: int | list[int] = 5, ...)</code> <br/>Make labels "empty", with a border of a given size."
    ::: cornucopia.labels.SmoothShallowLabelTransform
        options:
            heading_level: 3

??? quote "<code>[cc.BernoulliTransform](intensity/#cornucopia.labels.BernoulliTransform)(prob: float = 0.1)</code> <br/>Randomly mask voxels."
    ::: cornucopia.labels.BernoulliTransform
        options:
            heading_level: 3

??? quote "<code>[cc.SmoothBernoulliTransform](intensity/#cornucopia.labels.SmoothBernoulliTransform)(prob: float = 0.1, shape: int | list[int] = 5, *, shared: bool | str = False)</code> <br/>Randomly mask voxels."
    ::: cornucopia.labels.SmoothBernoulliTransform
        options:
            heading_level: 3

??? quote "<code>[cc.BernoulliDiskTransform](intensity/#cornucopia.labels.BernoulliDiskTransform)(prob: float = 0.1, radius: Sampler | float = 2)</code> <br/>Randomly mask voxels in balls at random locations."
    ::: cornucopia.labels.BernoulliDiskTransform
        options:
            heading_level: 3

??? quote "<code>[cc.SmoothBernoulliDiskTransform](intensity/#cornucopia.labels.SmoothBernoulliDiskTransform)(prob: float = 0.1, radius: float | tuple[float, float] = 2, shape: int | list[int] = 5, ..., *, shared: bool | str = False)</code> <br/>Randomly mask voxels in balls at random locations."
    ::: cornucopia.labels.SmoothBernoulliDiskTransform
        options:
            heading_level: 3

### Transforms with random parameters {#labels-random}

!!! bug "TODO"

## [`cc.kspace`](kspace/): Transforms that act on k-space (Fourier domain) {#kspace}

### Deterministic transforms {#kspace-deterministic}

??? quote "<code>[cc.SumOfSquaresTransform](intensity/#cornucopia.kspace.SumOfSquaresTransform)()</code> <br/>Compute the sum-of-squares across coils/channels."
    ::: cornucopia.kspace.SumOfSquaresTransform
        options:
            heading_level: 3

### Transforms with fixed parameters (but random coefficients) {#kspace-fixed}

??? quote "<code>[cc.ArrayCoilTransform](intensity/#cornucopia.kspace.ArrayCoilTransform)(ncoils: int = 8, fwhm: float = 0.5, diameter: float = 0.8, jitter: float = 0.01, ...)</code> <br/>Generate and apply random coil sensitivities (real or complex)."
    ::: cornucopia.kspace.ArrayCoilTransform
        options:
            heading_level: 3

### Transforms with random parameters {#kspace-random}

??? quote "<code>[cc.IntraScanMotionTransform](intensity/#cornucopia.kspace.IntraScanMotionTransform)(shots: int = 4, axis: int = -1, freq: bool = True, ...)</code> <br/>Model intra-scan motion."
    ::: cornucopia.kspace.IntraScanMotionTransform
        options:
            heading_level: 3

??? quote "<code>[cc.SmallIntraScanMotionTransform](intensity/#cornucopia.kspace.SmallIntraScanMotionTransform)(translations: [Sampler](random/#cornucopia.random.Sampler) | float = 0.05, rotations: [Sampler](random/#cornucopia.random.Sampler) | float = 5, axis: int = -1)</code> <br/>Model small intra-scan motion."
    ::: cornucopia.kspace.SmallIntraScanMotionTransform
        options:
            heading_level: 3

## [`cc.qmri`](qmri/): Quantitative MRI {#qmri}

### Deterministic transforms {#qmri-deterministic}

??? quote "<code>[cc.SusceptibilityToFieldmapTransform](intensity/#cornucopia.qmri.SusceptibilityToFieldmapTransform)(axis: int | vector_like[float] = -1, ...)</code> <br/>Convert a susceptibiity map (in ppm) into a field map (in Hz)."
    ::: cornucopia.qmri.SusceptibilityToFieldmapTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ShimTransform](intensity/#cornucopia.qmri.ShimTransform)(linear: vector_like[float] = 0, quadratic[float] = 0, isocenter: vector_like[float] | None = None)</code> <br/>Apply a shim field to the input field map."
    ::: cornucopia.qmri.ShimTransform
        options:
            heading_level: 3

??? quote "<code>[cc.OptimalShimTransform](intensity/#cornucopia.qmri.OptimalShimTransform)(max_order: int = 2, lam_abs: float = 1, lam_grad: float = 10)</code> <br/>Compute an optimal shim field for the input field map."
    ::: cornucopia.qmri.OptimalShimTransform
        options:
            heading_level: 3

??? quote "<code>[cc.HertzToPhaseTransform](intensity/#cornucopia.qmri.HertzToPhaseTransform)(te: float = 0)</code> <br/>Converts a ΔB0 field (in Hz) into a Phase shift field Δφ (in rad)."
    ::: cornucopia.qmri.HertzToPhaseTransform
        options:
            heading_level: 3

??? quote "<code>[cc.HertzToVoxelShiftTransform](intensity/#cornucopia.qmri.HertzToVoxelShiftTransformHertzToVoxelShiftTransform)(te: float = 0)</code> <br/>Converts a ΔB0 field (in Hz) into a voxel shift field Δv."
    ::: cornucopia.qmri.HertzToVoxelShiftTransform
        options:
            heading_level: 3

??? quote "<code>[cc.ApplyB0DistortionTransform](intensity/#cornucopia.qmri.ApplyB0DistortionTransform)(flow: tensor | str | None = None, vdm: tensor | str | None = None, controls: tensor | str | None = None, ...)</code> <br/>Apply a pre-compute B0 distortion field."
    ::: cornucopia.qmri.ApplyB0DistortionTransform
        options:
            heading_level: 3

??? quote "<code>[cc.GradientEchoTransform](intensity/#cornucopia.qmri.GradientEchoTransform)(tr: float = 25e-3, te: float = 7e-3, alpha: float = 20, pd: float | None = None, t1: float | None = None, t2: float | None = None, b1: float | None = 1, mt: float | None = 0)</code> <br/>Spoiled Gradient Echo forward model."
    ::: cornucopia.qmri.GradientEchoTransform
        options:
            heading_level: 3

### Transforms with fixed parameters (but random coefficients) {#qmri-fixed}

??? quote "<code>[cc.B0DistortionTransform](intensity/#cornucopia.qmri.B0DistortionTransform)(dmax: float = 0.1, unit: {'fov','vox'} = 'fov', shape: int | list[int] = 5, ...)</code> <br/>Compute and apply a B0 distortion field.."
    ::: cornucopia.qmri.B0DistortionTransform
        options:
            heading_level: 3

### Transforms with random parameters {#qmri-random}

??? quote "<code>[cc.RandomSusceptibilityMixtureTransform](intensity/#cornucopia.qmri.RandomSusceptibilityMixtureTransform)(...)</code> <br/>A RandomGaussianMixtureTransform tailored to susceptibility maps."
    ::: cornucopia.qmri.RandomSusceptibilityMixtureTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomShimTransform](intensity/#cornucopia.qmri.RandomShimTransform)(...)</code> <br/>Apply a random shim field to an input field."
    ::: cornucopia.qmri.RandomShimTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomB0DistortionTransform](intensity/#cornucopia.qmri.B0DistortionTransform)(dmax: [Sampler](random/#cornucopia.random.Sampler) | float = 0.1, shape: [Sampler](random/#cornucopia.random.Sampler) | int = 5, ...)</code> <br/>Apply a random B0 distortion field."
    ::: cornucopia.qmri.B0DistortionTransform
        options:
            heading_level: 3

??? quote "<code>[cc.RandomGMMGradientEchoTransform](intensity/#cornucopia.qmri.RandomGMMGradientEchoTransform)(tr: [Sampler](random/#cornucopia.random.Sampler) | float = 50E-3, te: [Sampler](random/#cornucopia.random.Sampler) | float = 50E-3, alpha: [Sampler](random/#cornucopia.random.Sampler) | float = 90, ...)</code> <br/>A RandomGaussianMixtureTransform tailored to quantitative MRI maps, followed by a GRE forward model."
    ::: cornucopia.qmri.RandomGMMGradientEchoTransform
        options:
            heading_level: 3

## [`cc.synth`](synth/): Synthesize images (domain randomization) {#synth}

??? quote "<code>[cc.IntensityTransform](intensity/#cornucopia.synth.IntensityTransform)(...)</code> <br/>Common intensity augmentation for MRI and related images."
    ::: cornucopia.synth.IntensityTransform
        options:
            heading_level: 3

??? quote "<code>[cc.SynthFromLabelTransform](intensity/#cornucopia.synth.SynthFromLabelTransform)(...)</code> <br/>Synthesize an MRI from an existing label map."
    ::: cornucopia.synth.IntensityTransform
        options:
            heading_level: 3
