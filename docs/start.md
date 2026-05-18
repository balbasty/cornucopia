---
icon: fontawesome/solid/rocket
---

# Getting started

## Import {#import}

Let's start with importing cornucopia:
```python
import cornucopia as cc
```

## Common principles {#principles}

!!! tip "Cornucopia uses an object-oriented model."

    Transforms are **first instantiated**, and **then applied** to a tensor:
    !!! example
        ```python
        xform = cc.ElasticTransform()
        deformed = xform(image)
        ```
    See the section
    [**Overview**](#overview) on this page.

!!! tip "All transforms take tensors with *one channel dimension* and *no batch dimension*."

    Their shape must be `[C, X, Y, Z]` or `[C, X, Y]`.

    See the section [**Overview**](#overview) on this page.

!!! tip "All transforms accept **positional or keyword** arguments."

    !!! example
        ```python
        image, label = xform(image, label)
        image, label = xform(img=image, lab=label)
        ```

    See the section [**Nested structures of tensors**](#nested) on this page.

!!! tip "Arguments can be **nested structures of tensors**."

    !!! example
        ```python
        output = xform({"images": [img1, img2], "labels": [lab1, lab]})
        ```

    See the section [**Nested structures of tensors**](#nested) on this page.

!!! tip "The interaction between a transform and its arguments can be controlled."

    Users can control how transforms interact with the arguments on which
    they act using the attributes `include`, `exclude`, `consume`, and `returns`.

    See [`Transform`][cornucopia.base.Transform] and the section
    [**Apply different transforms to different tensors**](#map) on this page.

!!! tip "Transforms are _final_, _non-final_ and/or _special_."

    A [`Transform`][cornucopia.base.Transform] can be a
    [`FinalTransform`][cornucopia.base.FinalTransform],
    [`NonFinalTransform`][cornucopia.base.NonFinalTransform], or
    [`SpecialTransform`][cornucopia.base.SpecialTransform].

    - Final transforms are fully deterministic and are applied in the same
      way to each input.
    - Non-final tranforms generate a final transform. It may either be
      because their parameters interact with the shape or content of a tensor,
      or because some of their parameters are randomized.
    - Special transforms act on other transforms. They never directly
      inherit from [`FinalTransform`][cornucopia.base.FinalTransform], but
      can still be deterministic if the transforms that they act on are
      themselves final. They may or may not also inherit from
      [`NonFinalTransform`][cornucopia.base.NonFinalTransform].

!!! tip "Non-final transforms can share their parameters across channels and/or tensors."

    This can be controlled by their `shared` attribute, whose default
    value depends on each transform:

    - For some of them, sharing does not make sense. For example, new
      Gaussian noise should be sampled for each tensor and each channel.
    - For others, sharing is more intuitive. For example geometric
      transforms, or transforms that modify the field of view are shared
      by default.

    Cornucopia uses sensible defaults for each transform.

## Overview {#overview}

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
# > A single deformation field is sampled and applied to both images.
img, lab = cc.ElasticTransform()(img, lab)
```

Note that if one wants to have different random parameters applied ot
different channels of an image, the keyword `shared=False` can be used:

```python
img = cc.ElasticTransform(shared=False)(img)
```

The default value for `shared` can differ from transform to transform.
For example, the default for `ElasticTransform` is `True` (since we expect
that most people want to apply the same deformation to different channels),
whereas the default for `GaussianNoiseTransform` is `False` (since we want
to re-sample noise in each channel). In general, `shared` can take the values:

- `"channels+tensors"` or `True`: the same transformation parameters
   are used for all tensors and all their channels
- `"channels"`: a different set of parameters is sampled for each tensor,
  but they are shared across their channels.
- `"tensors"`: a different set of parameters is sampled for each channel
  of the first (valid) tensor, which are then applied channel-wise to
  all other (valid) tensors. This assumes that all valid tensors have the
  same number of channels.
- `""` or `False`: a different set of parameteres is sampled for each
  channel of each tensor.

## Random transforms {#random}

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

## Sequences of transforms {#sequence}

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

## Randomized parameters {#randomized}

Better augmentation can be obtained if the parameters of a random transform
(_e.g._, Gaussian noise variance) are themselves sampled from a prior
distribution (_e.g._, a uniform distribution between 0 and 10).
We provide high-level randomized transform for this, as well as a utility
class that allows any transform to be easily randomized:

```python
# use a pre-defined randomized transform
img = cc.RandomAffineElasticTransform()(img)

# randomize a transform
hypernoise = cc.ctx.randomize(cc.GaussianNoise, cc.Uniform(0, 10))
img = hypernoise(img)
```

## Nested structures of tensors {#nested}

Last but not least, transforms accept any nested collection
(`list`, `tuple`, `dict`) of tensors, applies the transform to each
leaf tensor, and returns a collection with the same nested structure:

```python
img1 = cc.LoadTransform(dtype='float32')('path/to/mri1.nii.gz')
lab1 = cc.LoadTransform(dtype='long')('path/to/labels1.nii.gz')
img2 = cc.LoadTransform(dtype='float32')('path/to/mri2.nii.gz')
lab2 = cc.LoadTransform(dtype='long')('path/to/labels2.nii.gz')

# lists of tuples
dat = [(img1, lab1), (img2, lab2)]
dat = cc.ElasticTransform()(dat)
[(wimg1, wlab1), (wimg2, wlab2)] = dat

# or list of dictionaries
dat = [{"img": img1, "seg": lab1}, {"img": img2, "seg": lab2}]
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

## Apply different transforms to different tensors {#map}

It is then possible to take advantage of positional arguments, keywords
and/or dictionaries to apply some transforms to a subset of inputs.
This is done using the `cc.MappedTransform` meta-transform (or the
`cc.ctx.map` utility function):

```python
# using positionals
geom = cc.RandomElasticTransform()
noise = cc.GaussianNoiseTransform()
trf = cc.SequentialTranform([geom, cc.map(noise, None)])
img, lab = trf(img, lab)

# using keywords
geom = cc.RandomElasticTransform()
noise = cc.GaussianNoiseTransform()
trf = cc.SequentialTranform([geom, cc.map(image=noise)])
img, lab = trf(image=img, label=lab)

# using dictionaries
dat = [dict(image=img1, label=lab1), dict(image=img2, label=lab2)]
geom = cc.RandomElasticTransform()
noise = cc.GaussianNoiseTransform()
trf = cc.SequentialTranform([
    geom,
    cc.map(image=noise, nested=True)  # !! must be `nested`
])
dat = trf(dat)
```

Alternatively, a transform can be applied selectively to a set of keys
(or to all *but* a set of keys):

```python
geom = cc.RandomElasticTransform()
noise = cc.GaussianNoiseTransform()
trf = cc.SequentialTranform([geom, cc.ctx.include(noise, "image")])
img, lab = trf(image=img, label=lab)

geom = cc.RandomElasticTransform()
noise = cc.GaussianNoiseTransform()
trf = cc.SequentialTranform([geom, cc.ctx.exclude(noise, "label")])
img, lab = trf(image=img, label=lab)
```

## Batching {#batch}

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
