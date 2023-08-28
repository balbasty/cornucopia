<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="docs/icons/cornucopia_lightorange.svg">
  <source media="(prefers-color-scheme: light)"  srcset="docs/icons/cornucopia_orange.svg">
  <img alt="Cornucopia logo" src="docs/icons/cornucopia_orange.svg">
</picture>

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


## Installation

### Dependencies

- `pytorch >= 1.8`
- `numpy`
- `nibabel`
- `torch-interpol`
- `torch-distmap`

### Conda

```sh
conda install cornucopia -c balbasty -c pytorch 
```

### Pip

```sh
pip install cornucopia
```

## Documentation

Read the [documentation](https://cornucopia.readthedocs.io) and in particular:
- [installation](https://cornucopia.readthedocs.io/en/latest/install/)
- [get started](https://cornucopia.readthedocs.io/en/latest/start/)
- [examples](https://cornucopia.readthedocs.io/en/latest/examples/overview/)
- [API](https://cornucopia.readthedocs.io/en/latest/api/overview/)

## Other augmentation packages

There are other great, and much more mature, augmentation packages 
out-there (although few run on the GPU). Here's a non-exhaustive list:
- [MONAI](https://github.com/Project-MONAI/MONAI)
- [TorchIO](https://github.com/fepegar/torchio)
- [Albumentations](https://github.com/albumentations-team/albumentations) (2D only)
- [Volumentations](https://github.com/ZFTurbo/volumentations) (3D extension of Albumentations)

## Contributions

If you find this project useful and wish to contribute, please reach out!
