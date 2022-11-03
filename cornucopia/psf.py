__all__ = ['SmoothTransform', 'RandomSmoothTransform',
           'LowResTransform', 'RandomLowResTransform',
           'LowResSliceTransform', 'RandomLowResSliceTransform']

from .base import Transform, RandomizedTransform
from .utils.conv import smoothnd
from .utils.py import ensure_list
from .random import Uniform, RandInt, Fixed, sym_range, upper_range, \
    lower_range
from interpol import resize


class SmoothTransform(Transform):
    """Apply Gaussian smoothing"""

    def __init__(self, fwhm=1):
        """

        Parameters
        ----------
        fwhm : float
            Full-width at half-maximum of the Gaussian kernel
        """
        super().__init__()
        self.fwhm = fwhm

    def apply_transform(self, x, parameters):
        return smoothnd(x, fwhm=ensure_list(self.fwhm, x.dim()-1))


class RandomSmoothTransform(RandomizedTransform):
    """Apply Gaussian smoothing with random FWHM"""

    def __init__(self, fwhm=2, shared=False):
        """

        Parameters
        ----------
        fwhm : Sampler or  float
            Sampler or upper bound for the full-width at half-maximum
        shared : bool
            Use the same fwhm for all channels/tensors
        """
        super().__init__(SmoothTransform,
                         dict(fwhm=Uniform.make(upper_range(fwhm))),
                         shared=shared)


class LowResSliceTransform(Transform):
    """
    Model a low-resolution slice direction, with Gaussian profile
    """

    def __init__(self, resolution=3, thickness=0.8, axis=-1, noise=None):
        """

        Parameters
        ----------
        resolution : float
            Resolution of the slice dimension, in terms of high-res voxels
        thickness : float in 0..1
            Slice thickness, as a proportion of resolution
        axis : int
            Slice axis
        noise : Transform, optional
            A transform that adds noise in the low-resolution space
        """
        super().__init__()
        self.resolution = resolution
        self.noise = noise
        self.axis = axis
        self.thickness = thickness

    def get_parameters(self, x):
        if self.noise:
            return self.noise.get_parameters(x)
        return None

    def apply_transform(self, x, parameters):
        ndim = x.dim() - 1
        fwhm = [0] * ndim
        fwhm[self.axis] = self.resolution * self.thickness
        y = smoothnd(x, fwhm=fwhm)
        if self.noise is not None:
            y = self.noise.apply_transform(y, parameters)
        factor = [1] * ndim
        factor[self.axis] = 1/self.resolution
        y = resize(y[None], factor=factor, anchor='f')[0]
        factor[self.axis] = self.resolution
        y = resize(y[None], factor=factor, shape=x.shape[1:], anchor='f')[0]
        return y


class RandomLowResSliceTransform(RandomizedTransform):
    """Random low-resolution slice direction, with Gaussian profile"""

    def __init__(self, resolution=3, thickness=0.1, axis=None, noise=None,
                 shared=False):
        """

        Parameters
        ----------
        resolution : Sampler or float
            Sampler or upper bound for the resolution of the slice dimension,
            in terms of high-res voxels
        thickness : Sampler or float in 0..1
            Sampler or lower bound for the slice thickness,
            as a proportion of resolution
        axis : int
            Slice axis. If None, select one randomly.
        noise : Transform, optional
            A transform that adds noise in the low-resolution space
        shared : bool
            Use the same resolution for all channels/tensors
        """
        super().__init__(RandomLowResSliceTransform,
                         dict(resolution=Uniform.make(upper_range(resolution)),
                              thickness=Uniform.make(lower_range(thickness, 1)),
                              axis=Fixed.make(axis),
                              noise=Fixed.make(noise)),
                         shared=shared)

    def get_parameters(self, x):
        resolution = self.sample['resolution']()
        thickness = self.sample['thickness']()
        axis = self.sample['axis']()
        noise = self.sample['noise']()
        if axis is None:
            axis = -(RandInt(1, x.dim()-1)())
        return RandomLowResSliceTransform(resolution, thickness, axis, noise)


class LowResTransform(Transform):
    """Model a lower-resolution image"""

    def __init__(self, resolution=2, noise=None):
        """

        Parameters
        ----------
        resolution : float or list[float]
            Resolution of the low-res image, in terms of high-res voxels
        noise : Transform, optional
            A transform that adds noise in the low-resolution space
        """
        super().__init__()
        self.resolution = resolution
        self.noise = noise

    def apply_transform(self, x, parameters):
        ndim = x.dim() - 1
        resolution = ensure_list(self.resolution, ndim)
        y = smoothnd(x, fwhm=resolution)
        if self.noise is not None:
            y = self.noise.apply_transform(y, parameters)
        factor = [1/r for r in resolution]
        y = resize(y[None], factor=factor, anchor='f')[0]
        y = resize(y[None], factor=resolution, shape=x.shape[1:], anchor='f')[0]
        return y


class RandomLowResTransform(RandomizedTransform):
    """Random lower-resolution image"""

    def __init__(self, resolution=2, shared=False):
        """

        Parameters
        ----------
        resolution : Sampler or  float
            Sampler or upper bound for the output resolution
        shared : bool
            Use the same resolution for all channels/tensors
        """
        super().__init__(LowResTransform,
                         dict(resolution=Uniform.make(upper_range(resolution))),
                         shared=shared)
