# Getting started

### Import

Let's start with importing cornucopia:
```python
import cornucopia as cc
```

### Overview

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
to re-sample noise in each channel).

### Random transforms

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
img = (0.2 * gauss)(img)                           # -> MaybeTransform
img = (gauss | chi)(img)                           # -> SwitchTransform
img = cc.ctx.switch({gauss: 0.5, chi: 0.5})(img)   # -> SwitchTransform
```

### Sequences of transforms

Transforms can be composed together using the `SequentialTransform` class,
or by simply adding them together:
```python
# programatic instantiation of a sequence
seq = cc.SequentialTransform([
       cc.ElasticTransform(), 
       cc.GaussianNoiseTransform()
])

# syntactic sugar
seq = cc.ElasticTransform() + cc.GaussianNoiseTransform()

img = seq(img)
```

### Randomized parameters

Better augmentation can be obtained if the parameters of a random transform
(_e.g._, Gaussian noise variance) are themselves sampled from a prior
distribution (_e.g._, a uniform distribution between [0 and 10]).
We provide high-level randomized transform for this, as well as a utility
class that allows any transform to be easily randomized:
```python
# use a pre-defined randomized transform
img = cc.RandomAffineElasticTransform()(img)

# randomize a transform
hypernoise = cc.ctx.randomize(cc.GaussianNoise, cc.Uniform(0, 10))
img = hypernoise(img)
```

### Nested structures of tensors

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

### Apply different transforms to different tensors

It is then possible to take advantage of positional arguments, keywords
and/or dictionaries to apply some transforms to a subset of inputs.
This is done using the `cc.MappedTransform` meta-transform (or the 
`cc.ctx.map` utility function):
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
trf = geom + cc.ctx.include(noise, "image")
img, lab = trf(image=img, label=lab)

geom = cc.RandomElasticTransform()
noise = cc.GaussianNoiseTransform()
trf = geom + cc.ctx.exclude(noise, "label")
img, lab = trf(image=img, label=lab)
```

### Batching

Because cornucopia transforms are implemented in pure PyTorch, they can be run
**on the CPU or GPU**, and benefit greatly from being run on the GPU.
We advise to only call `LoadTransform` (and eventually `PatchTransform`) in
your dataloader, and apply all other augmentations inside a PyTorch module.
The `BatchedTransform` class allows a transform to be applied to a batched
tensor or a nested structure of batched tensors:
```python
batched_transform = cc.ctx.batch(transform) # or cc.BatchedTransform(transform)
img, lab = batched_transform(img, lab)      # input shapes: [B, C, X, Y, Z]
```

**Happy augmentation!**