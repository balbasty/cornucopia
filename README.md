# Cornucopia

The `cornucopia` package provides a generic framework for preprocessing,
augmentation, and domain randomization; along with an abundance of specific layers,
mostly targeted at (medical) imaging. `cornucopia` is written using a PyTorch
backend, and therefore runs **on the CPU or GPU**.

Cornucopia is *intended* to be used on the GPU for on-line augmentation.
A quick [benchmark](examples/benchmark.ipynb) of affine and elastic augmentation
shows that while cornucopia is slower than [TorchIO](https://github.com/fepegar/torchio)
on the CPU (~ 3s vs 1s), it is greatly accelerated on the GPU (~ 50ms).

Since gradients are not expected to backpropagate through its layers, it can
theoretically be used within any dataloader pipeline,
independent of the downstream learning framework (pytorch, tensorflow, jax, ...).

## Installation

```shell
pip install git+https://github.com/balbasty/cornucopia
```

## Usage

Let's start with importing cornucopia:
```python
import cornucopia as cc
```

Transforms are simply Pytorch modules that expect tensors with a channel but
no batch dimension (e.g., `[C, X, Y, Z]`).
To make this clear, the name of (almost) all transforms implemented has the
suffix `Transform`. For example, here's how to add random noise to an image:
```python
# Load an MRI and ensure that it is reshaped as [C, X, Y, Z]
img = cc.LoadTransform(dtype='float32')('path/to/mri.nii.gz')

# Add random noise
img = cc.GaussianNoiseTransform()(img)
```

Sometimes, the exact same transform must be applied to multiple images
(say, a geometric transform). In this case, multiple images can be provided:
```python
img = cc.LoadTransform(dtype='float32')('path/to/mri.nii.gz')
lab = cc.LoadTransform(dtype='long')('path/to/labels.nii.gz')

# Apply a random elastic transform
img, lab = cc.ElasticTransform()(img, lab)
```

Note that if one wants to have different random parameters applied ot different
channels of an image, the keyword `shared=False` can be used:
```python
img = cc.ElasticTransform(shared=False)(img)
```
The default value for `shared` can differ from transform to transform.
For example, the default for `ElasticTransform` is `True` (since we expect
that most people want to apply the same deformation to different channels),
whereas the default for `GaussianNoiseTransform` is `False` (since we want
to resample noise in each channel).

We offer utilities to randomly activate the application of a transform,
or randomly choose a transform to apply from a set of transforms:
```python
gauss = cc.GaussianNoiseTransform()
chi = cc.ChiNoiseTransform()

# 20% chance of adding noise
img = cc.MaybeTransform(gauss, 0.2)(img)
# randomly apply either Gaussian or Chi noise
img = cc.SwitchTransform([gauss, chi])(img)  

# syntactic sugar
img = 0.2 * gauss                           # -> MaybeTransform
img = gauss | chi                           # -> SwitchTransform
img = cc.switch({gauss: 0.5, chi: 0.5})     # -> SwitchTransform
```

Transforms can be composed together using the `SequentialTransform` class,
or by simply adding them together:
```python
# programatic instantiation of a sequence
seq = cc.SequentialTransform([cc.ElasticTransform(), cc.GaussianNoiseTransform()])
# syntactic sugar
seq = cc.ElasticTransform() + cc.GaussianNoiseTransform()

img = seq(img)
```

Better augmentation can be obtained if the parameters of a random transform
(_e.g._, Gaussian noise variance) are themselves sampled from a prior
distribution (_e.g._, a uniform distribution between [0 and 10]).
We provide high-level randomized transform for this, as well as a utility
class that allows any transform to be easily randomized:
```python
# use a pre-defined randomized transform
img = cc.RandomAffineElasticTransform()(img)

# randomize a transform
hypernoise = cc.randomize(cc.GaussianNoise)(cc.Uniform(0, 10))
img = hypernoise(img)
```

Last but not least, transforms accept any nested collection 
(`list`, `tuple`, `dict`) of tensors, applies the transform to each 
leaf Tensor, and returns a collection with the same nested structure:
```python
img1 = cc.LoadTransform(dtype='float32')('path/to/mri1.nii.gz')
lab1 = cc.LoadTransform(dtype='long')('path/to/labels1.nii.gz')
img2 = cc.LoadTransform(dtype='float32')('path/to/mri2.nii.gz')
lab2 = cc.LoadTransform(dtype='long')('path/to/labels2.nii.gz')

dat = [(img1, lab1), (img2, lab2)]
dat = cc.ElasticTransform()(dat)
```

It is also possible (as briefly showed earlier) to pass positional or even 
keyword arguments. In this case, we return special `Args`, `Kwargs` or 
`ArgsAndKwargs` object that act like tuples and dictionaries (but allow us
to properly deal with transfoms composition and such). These structures 
understand implicit unpacking, and always unpack values (which differs from 
native `dict`, which unpack keys).
```python
img1, lab1 = cc.ElasticTransform()(img1, lab1)
img1, lab1 = cc.ElasticTransform()(image=img1, label=lab1)
img1, img2, lab1 = cc.ElasticTransform()(img1, img2, label=lab1)
dat = cc.ElasticTransform()(image=img1, label=lab1)
img1, lab1 = dat['image'], dat['label']
# etc.
```

It is then possible to take advantage of positional arguments, keywords 
and/or dictionaries to apply some transforms to a subset of inputs. 
This is done using the `cc.MappedTransform` meta-transform (or the `cc.map`
utility function):
```python
# using positionals
geom = cc.RandomElasticTransform()
noise = cc.GaussianNoiseTransform()
trf = geom + cc.map(noise, None)
img, lab = trf(img, lab)

# using keywords
geom = cc.RandomElasticTransform()
noise = cc.GaussianNoiseTransform()
trf = geom + cc.map(image=noise)
img, lab = trf(image=img, label=lab)

# using dictionaries
dat = [dict(image=img1, label=lab1), 
       dict(image=img2, label=lab2)]
geom = cc.RandomElasticTransform()
noise = cc.GaussianNoiseTransform()
trf = geom + cc.map(image=noise, nested=True) # !! must be `nested`
dat = trf(dat)
```

Alternatively, a transform can be applied selectively to a set of keys
(or to all *but* a set of keys):
```python
geom = cc.RandomElasticTransform()
noise = cc.GaussianNoiseTransform()
trf = geom + cc.include_keys(noise, "image")
img, lab = trf(image=img, label=lab)

geom = cc.RandomElasticTransform()
noise = cc.GaussianNoiseTransform()
trf = geom + cc.exclude_keys(noise, "label")
img, lab = trf(image=img, label=lab)
```

### Plug it in your project

Because cornucopia transforms are implemented in pure PyTorch, they can be run
**on the CPU or GPU**, and benefit greatly from being run on the GPU.
We advise to only call `LoadTransform` (and eventually `PatchTransform`) in
your dataloader, and apply all other augmentations inside a PyTorch module.
The `BatchedTransform` class allows a transform to be applied to a batched
tensor or a nested structure of batched tensors:
```python
batched_transform = cc.batch(transform)  # or cc.BatchedTransform(transform)
img, lab = batched_transform(img, lab)   # input shapes: [B, C, X, Y, Z]
```

**Happy augmentation!**

## Current API (useful subset)

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


## Other augmentation packages

There are other great, and much more mature, augmentation packages out-there (although few run on the GPU). Here's a non-exhaustive list:
- [MONAI](https://github.com/Project-MONAI/MONAI)
- [TorchIO](https://github.com/fepegar/torchio)
- [Albumentations](https://github.com/albumentations-team/albumentations) (2D only)
- [Volumentations](https://github.com/ZFTurbo/volumentations) (3D extension of Albumentations)

## Contributions

If you find this project useful and wish to contribute, please reach out!
