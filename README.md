# Cornucopia

Cornucopia, or horn of plenty, from latin _cornu_ (horn) and _copi_ (plenty), is a symbol of abundance from classical antiquity.

The `cornucopia` package provides a generic framework for preprocessing, augmentation, and domain randomization; along with an abundance of specific layers,
mostly targetted at (medical) imaging. `cornucopia` is written using a PyTorch backend, and therefore runs on the CPU or GPU. However, since gradients are not
expected to backpropagate through its layers, it can be used within any dataloader pipeline, independent of the downstream learning framework
(pytorch, tensorflow, jax, ...).

## Installation

```shell
pip install git+https://github.com/balbasty/cornucopia
```

## Usage

Let's start with importing corncucopia:
```python
import cornucopia as cc
```

Transforms are simply Pytorch modules that expect (`list` or `tuple` or `dict` of) tensors with a channel but no batch dimension (e.g., `[C, X, Y, Z]`). 
To make this clear, the name of (almost) all transforms implemented has the suffix `Transform`. For example, here's how to add random noise to an image:
```python
# Load an MRI and ensure that it is reshaped as [C, X, Y, Z]
img = cc.LoadTransform(dtype='float32')('path/to/mri.nii.gz')

# Add random noise
img = cc.GaussianNoiseTransform()(img)
```

Sometimes, the exact same transform must be applied to multiple images (say, a geometric transform). In this case, multiple images can be provided:
```python
img = cc.LoadTransform(dtype='float32')('path/to/mri.nii.gz')
lab = cc.LoadTransform(dtype='long')('path/to/labels.nii.gz')

# Apply a random elastic transform
img, lab = cc.ElasticTransform()(img, lab)
```

Note that if one wants to have different random parameters applied ot different channels of an image, the keyword `shared=False` can be used:
```python
img = cc.ElasticTransform(shared=False)(img)
```
The default value for `shared` can differ from transform to transform. For example, the default for `ElasticTransform` is `True` (since we expect that most people want to apply the same deformation to different channels), whereas the default for `GaussianNoiseTransform` is `False` (since we want to resample noise in each channel).

We offer a bunch of utilities to apply transforms to randomly activate the application of a transform, or randomly choose a transform to apply from a set of transforms:
```python
# 20% chance of adding noise
img = cc.MaybeTransform(cc.GaussianNoiseTransform(), 0.2)(img)
# randomly apply either Gaussian or Chi noise
img = cc.SwitchTransform([cc.GaussianNoiseTransform(), cc.ChiNoiseTransform()])(img)  
```

Transforms can be composed together using the `SequentialTransform` class, or by simply adding them together:
```python
# programatic instantiation of a sequence
seq = cc.SequentialTransform([cc.ElasticTransform(), cc.GaussianNoiseTransform()])
# syntaxic sugar
seq = cc.ElasticTransform() + cc.GaussianNoiseTransform()

img = seq(img)
```

Better augmentation can be obtained if the parameters of a random transform (_e.g._, Gaussian noise variance) are themselves sampled from a prior distribution (_e.g._, a uniform distribution between [0 and 10]). We provide high-level randomized transform for this, as well as a utility class that allows any transform to be easily randomized:
```python
# use a pre-defined randomized transform
img = cc.RandomAffineElasticTransform()(img)

# randomize a transform
hypernoise = cc.RandomizedTransform(cc.GaussianNoise, cc.Uniform(0, 10))
img = hypernoise(img)
```

### Plug it in your project

Because cornucopia transforms are implemented in pure PyTorch, they can be run **on the CPU or GPU**, and benefit greatly from being run on the GPU.
We advise to only call `LoadTransform` (and eventually `PatchTransform`) in your dataloader, and apply all other augmentations inside a PyTorch module.
For example, here's how we wrap cornucopia inside a PyTorch module:
```python
class Augmenter(nn.Module):

    def __init__(self, transform: cc.Transform):
        super().__init__()
        self.transform = transform
        

    def forward(self, x):
        img = torch.empty_like(x)
        for i, x1 in enumerate(x):
            img[i]= self.transform(x1)
        return img
```

**Happy augmentation!**

## Current API (useful subset)

### Meta transforms
```python
cc.SequentialTransform(transforms: List[Transform])
cc.MaybeTransform(transform: Transform, prob=0.5, shared=False)
cc.SwitchTransform(transforms: List[Transform], prob=0, shared=False)
cc.RandomizedTransform(transform_class, sample, ksample=None, shared=False)
cc.MappedTransform(**map)
cc.SplitChannels()
cc.CatChannels()

# helper
cc.randomize(transform_class, shared=False)(*args, **kwargs) -> RandomizedTransform
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
cc.GFactorTransform(noise: Transform, shape=5, vmin=1, vmax=4)
cc.GammaNoiseTransform(mean=1, sigma=0.1, shared=False)
```

### Intensity
```python
cc.MultFieldTransform(shape=5, vmin=0, vmax=1, shared=False)
cc.AddFieldTransform(shape=5, vmin=0, vmax=1, shared=False)
cc.GlobalMultTransform(value=1, shared=False)
cc.RandomGlobalMultTransform(value: Sampler or float or pair[float] = (0.5, 2), shared=True)
cc.GlobalAdditiveTransform(value=0, shared=False)
cc.RandomGlobalAdditiveTransform(value: Sampler or float or pair[float] = 1, shared=True)
cc.GammaTransform(gamma=1, vmin=None, vmax=None, shared=False)
cc.ZTransform(shared=False)
cc.QuantileTransform(pmin=0.01, pmax=0.99, vmin=0, vmax=1, clamp=False, shared=False)
```

### Contrast
```python
cc.ContrastMixtureTransform(nk=16, keep_background=True, shared='channels')
cc.ContrastLookupTransform(nk=16, keep_background=True, shared=False)
```

### Point-spread function
```python
cc.SmoothTransform(fwhm=1)
cc.LowResSliceTransform(resolution=3, thickness=0.8, axis=-1, noise: Transform = None)
cc.LowResTransform(resolution=2, noise: Transform = None)
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

# Randomized versions
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
cc.RandomGaussianMixtureTransform(mu=255, sigma=16, fwhm=2, background=None, shared='channels')
cc.SmoothLabelMap(nb_classes=2, shape=5, soft=False, shared=False)
cc.ErodeLabelMap(labels, radius=3, output_labels=0)
cc.RandomErodeLabelTransform(labels=0.5, radius=3, output_labels=0, shared=False)
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
