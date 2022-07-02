# Cornucopia

Cornucopia, or horn of plenty, from latin _cornu_ (horn) and _copi_ (plenty), is a symbol of abundance from classical antiquity.

The `cornucopia` package provides a generic framework for preprocessing, augmentation, and domain randomization; along with an abundance of specific layers, 
mostly targetted at (medical) imaging. `cornucopia` is written using a PyTorch backend, and therefore runs on the CPU or GPU. However, since gradients are not 
expected to backpropagate through its layers, it can be used within any dataloader pipeline, independent of the downstream learning framework
(pytorch, tensorflow, jax, ...).


.. toctree::
   :maxdepth: 2
   :caption: Contents:
