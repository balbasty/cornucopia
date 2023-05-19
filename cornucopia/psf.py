__all__ = ['SmoothTransform', 'RandomSmoothTransform',
           'LowResTransform', 'RandomLowResTransform',
           'LowResSliceTransform', 'RandomLowResSliceTransform']

from .base import Transform, RandomizedTransform, prepare_output
from .utils.conv import smoothnd
from .utils.py import ensure_list
from .random import Uniform, RandInt, Fixed, upper_range, lower_range
from torch.nn.functional import interpolate
import math


class SmoothTransform(Transform):
    """Apply Gaussian smoothing"""

    def __init__(self, fwhm=1, *, returns=None):
        """

        Parameters
        ----------
        fwhm : float
            Full-width at half-maximum of the Gaussian kernel
        returns : [list or dict of] {'input', 'output'}, default='output'
            Which tensors to return.
        """
        super().__init__(returns=returns)
        self.fwhm = fwhm

    def apply_transform(self, x, parameters):
        y = smoothnd(x, fwhm=ensure_list(self.fwhm, x.dim()-1))
        return prepare_output(dict(input=x, output=y), self.returns)


class RandomSmoothTransform(RandomizedTransform):
    """Apply Gaussian smoothing with random FWHM"""

    def __init__(self, fwhm=2, *, returns=None, shared=False):
        """

        Parameters
        ----------
        fwhm : Sampler or  float
            Sampler or upper bound for the full-width at half-maximum
        returns : [list or dict of] {'input', 'output'}, default='output'
            Which tensors to return.
        shared : bool
            Use the same fwhm for all channels/tensors
        """
        super().__init__(SmoothTransform,
                         dict(fwhm=Uniform.make(upper_range(fwhm)),
                              returns=returns),
                         shared=shared)


class LowResSliceTransform(Transform):
    """
    Model a low-resolution slice direction, with Gaussian profile
    """

    def __init__(self, resolution=3, thickness=0.8, axis=-1, noise=None,
                 *, returns=None):
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
        returns : [list or dict of] {'input', 'lowres', 'output'}, default='output'
            Which tensors to return.
        """
        super().__init__(returns=returns)
        self.resolution = resolution
        self.noise = noise
        self.axis = axis
        self.thickness = thickness

    def get_parameters(self, x):
        ndim = x.dim() - 1
        factor = [1] * ndim
        factor[self.axis] = 1/self.resolution
        oshape = [math.ceil(s*f) for s, f in zip(x.shape[1:], factor)]
        if self.noise:
            fake_x = x.new_zeros([]).expand([len(x), *oshape])
            return self.noise.get_parameters(fake_x)
        return None

    def apply_transform(self, x, parameters):
        ndim = x.dim() - 1
        mode = ('trilinear' if ndim == 3 else
                'bilinear' if ndim == 2 else
                'linear')
        fwhm = [0] * ndim
        fwhm[self.axis] = self.resolution * self.thickness
        y = smoothnd(x, fwhm=fwhm)
        if self.noise is not None:
            y = self.noise.apply_transform(y, parameters)
        factor = [1] * ndim
        factor[self.axis] = 1/self.resolution
        ishape = x.shape[1:]
        oshape = [math.ceil(s*f) for s, f in zip(ishape, factor)]
        y = interpolate(y[None], size=oshape, align_corners=True, mode=mode)[0]
        if self.noise is not None:
            y = self.noise.apply_transform(y, parameters)
        z = interpolate(y[None], size=ishape, align_corners=True, mode=mode)[0]
        return prepare_output(dict(input=x, lowres=y, output=z), self.returns)


class RandomLowResSliceTransform(RandomizedTransform):
    """Random low-resolution slice direction, with Gaussian profile"""

    def __init__(self, resolution=3, thickness=0.1, axis=None, noise=None,
                 *, returns=None, shared=False):
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
        returns : [list or dict of] {'input', 'lowres', 'output'}, default='output'
            Which tensors to return.
        shared : bool
            Use the same resolution for all channels/tensors
        """
        super().__init__(RandomLowResSliceTransform,
                         dict(resolution=Uniform.make(upper_range(resolution, min=1)),
                              thickness=Uniform.make(lower_range(thickness, 1)),
                              axis=Fixed.make(axis),
                              noise=noise,
                              returns=returns),
                         shared=shared)

    def get_parameters(self, x):
        resolution = self.sample['resolution']()
        thickness = self.sample['thickness']()
        axis = self.sample['axis']()
        noise = self.sample['noise']
        returns = self.sample['returns']
        if axis is None:
            axis = -(RandInt(1, x.dim()-1)())
        return LowResSliceTransform(resolution, thickness, axis, noise, returns)


class LowResTransform(Transform):
    """Model a lower-resolution image"""

    def __init__(self, resolution=2, noise=None, *, returns=None):
        """

        Parameters
        ----------
        resolution : float or list[float]
            Resolution of the low-res image, in terms of high-res voxels
        noise : Transform, optional
            A transform that adds noise in the low-resolution space
        returns : [list or dict of] {'input', 'lowres', 'output'}, default='output'
            Which tensors to return.
        """
        super().__init__(returns=returns)
        self.resolution = resolution
        self.noise = noise

    def get_parameters(self, x):
        ndim = x.dim() - 1
        resolution = ensure_list(self.resolution, ndim)
        factor = [1/r for r in resolution]
        oshape = [math.ceil(s*f) for s, f in zip(x.shape[1:], factor)]
        if self.noise:
            fake_x = x.new_zeros([]).expand([len(x), *oshape])
            return self.noise.get_parameters(fake_x)
        return None

    def apply_transform(self, x, parameters):
        ndim = x.dim() - 1
        mode = ('trilinear' if ndim == 3 else
                'bilinear' if ndim == 2 else
                'linear')
        resolution = ensure_list(self.resolution, ndim)
        y = smoothnd(x, fwhm=resolution)
        factor = [1/r for r in resolution]
        ishape = x.shape[1:]
        oshape = [math.ceil(s*f) for s, f in zip(ishape, factor)]
        y = interpolate(y[None], size=oshape, align_corners=True, mode=mode)[0]
        if self.noise is not None:
            y = self.noise.apply_transform(y, parameters)
        z = interpolate(y[None], size=ishape, align_corners=True, mode=mode)[0]
        return prepare_output(dict(input=x, lowres=y, output=z), self.returns)


class RandomLowResTransform(RandomizedTransform):
    """Random lower-resolution image"""

    def __init__(self, resolution=2, noise=None, *, returns=None, shared=False):
        """

        Parameters
        ----------
        resolution : Sampler or  float
            Sampler or upper bound for the output resolution
        noise : Transform, optional
            A transform that adds noise in the low-resolution space
        returns : [list or dict of] {'input', 'lowres', 'output'}, default='output'
            Which tensors to return.
        shared : bool
            Use the same resolution for all channels/tensors
        """
        super().__init__(LowResTransform,
                         dict(resolution=Uniform.make(upper_range(resolution, min=1)),
                              noise=noise,
                              returns=returns),
                         shared=shared)
