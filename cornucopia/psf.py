"""
This module contains transforms that modify the resolution and/or
point spread function of an image.
"""
__all__ = [
    'SmoothTransform',
    'RandomSmoothTransform',
    'LowResTransform',
    'RandomLowResTransform',
    'LowResSliceTransform',
    'RandomLowResSliceTransform',
]

from .base import FinalTransform, NonFinalTransform
from .special import RandomizedTransform
from .baseutils import prepare_output
from .utils.conv import smoothnd
from .utils.py import ensure_list
from .random import Sampler, Uniform, RandInt, make_range
from torch.nn.functional import interpolate
import math


class SmoothTransform(FinalTransform):
    """Apply Gaussian smoothing"""

    def __init__(self, fwhm=1, **kwargs):
        """

        Parameters
        ----------
        fwhm : float
            Full-width at half-maximum of the Gaussian kernel

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output'}
            Which tensors to return.
        """
        super().__init__(**kwargs)
        self.fwhm = fwhm

    def apply(self, x):
        return smoothnd(x, fwhm=ensure_list(self.fwhm, x.dim()-1))


class RandomSmoothTransform(RandomizedTransform):
    """Apply Gaussian smoothing with random FWHM"""

    def __init__(self, fwhm=2, *, shared=False, **kwargs):
        """

        Parameters
        ----------
        fwhm : Sampler or  float
            Sampler or upper bound for the full-width at half-maximum

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'output'}
            Which tensors to return.
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Use the same fwhm for all channels/tensors
        """
        super().__init__(SmoothTransform,
                         dict(fwhm=Uniform.make(make_range(0, fwhm))),
                         shared=shared, **kwargs)


class LowResSliceTransform(NonFinalTransform):
    """
    Model a low-resolution slice direction, with Gaussian profile
    """

    class Final(FinalTransform):

        def __init__(self, axis, resolution, thickness, noise=None, **kwargs):
            super().__init__(**kwargs)
            self.axis = axis
            self.resolution = resolution
            self.thickness = thickness
            self.noise = noise

        def apply(self, x):
            ndim = x.dim() - 1
            mode = ('trilinear' if ndim == 3 else
                    'bilinear' if ndim == 2 else
                    'linear')

            def interpol(x, shape):
                return interpolate(
                    x[None], size=shape, align_corners=True, mode=mode
                )[0]

            fwhm = [0] * ndim
            fwhm[self.axis] = self.resolution * self.thickness
            y = smoothnd(x, fwhm=fwhm)
            factor = [1] * ndim
            factor[self.axis] = 1/self.resolution
            ishape = x.shape[1:]
            oshape = [max(2, math.ceil(s*f)) for s, f in zip(ishape, factor)]
            y = interpol(y, oshape)
            if self.noise:
                y = self.noise(y)
            z = interpol(y, ishape)
            return prepare_output(
                dict(input=x, lowres=y, output=z),
                self.returns
            )

    def __init__(self, resolution=3, thickness=0.8, axis=-1, noise=None,
                 **kwargs):
        """

        Parameters
        ----------
        resolution : float
            Resolution of the slice dimension, in terms of high-res voxels.
            This is the distance between the centers of two consecutive slices.
        thickness : float in 0..1
            Slice thickness, as a proportion of resolution.
            This is how much data is averaged into one low-resolution slice.
        axis : int
            Slice axis
        noise : Transform, optional
            A transform that adds noise in the low-resolution space

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'lowres', 'output'}
            Which tensors to return.
        """
        super().__init__(**kwargs)
        self.resolution = resolution
        self.noise = noise
        self.axis = axis
        self.thickness = thickness

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self

        noise = None
        if self.noise:
            factor = [1] * (x.ndim-1)
            factor[self.axis] = 1/self.resolution
            ishape = x.shape[1:]
            oshape = [max(2, math.ceil(s*f)) for s, f in zip(ishape, factor)]
            fake_x = x.new_zeros([]).expand([len(x), *oshape])
            noise = self.noise.make_final(fake_x, max_depth-1)

        return self.Final(
            self.axis, self.resolution, self.thickness, noise,
            **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomLowResSliceTransform(RandomizedTransform):
    """Random low-resolution slice direction, with Gaussian profile"""

    def __init__(self, resolution=3, thickness=0.1, axis=None, noise=None,
                 *, shared=False, **kwargs):
        """
        Parameters
        ----------
        resolution : Sampler or float
            Distribution from which to sample the resolution.
            If a `float`, sample from `Uniform(1, value)`.
            To force a fixed value, pass `Fixed(value)`.
            The resolution is the distance (in terms of high-res voxels)
            between the centers of two consecutive low-res slices.
        thickness : Sampler or float in 0..1
            Distribution from which to sample the resolution.
            If a `float`, sample from `Uniform(value, 1)`.
            To force a fixed value, pass `Fixed(value)`.
            Thickness is defined as a proportion of the resolution, and
            determines how much data is averaged into one low-resolution slice.
        axis : int
            Slice axis. If None, select one randomly.
        noise : Transform, optional
            A transform that adds noise in the low-resolution space

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'lowres', 'output'}
            Which tensors to return.
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Use the same resolution for all channels/tensors
        """
        super().__init__(
            LowResSliceTransform,
            dict(resolution=Uniform.make(make_range(1, resolution)),
                 thickness=Uniform.make(make_range(thickness, 1)),
                 axis=axis,
                 noise=noise),
            shared=shared,
            **kwargs
        )

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return self.make_per_channel(x, max_depth)
        resolution = self.sample['resolution']
        if isinstance(resolution, Sampler):
            resolution = resolution()
        thickness = self.sample['thickness']
        if isinstance(thickness, Sampler):
            thickness = thickness()
        axis = self.sample['axis']
        if axis is None:
            axis = RandInt(x.ndim-2)
        if isinstance(axis, Sampler):
            axis = axis()
        noise = self.sample['noise']
        # if noise:
        #     noise = noise.make_final(x, max_depth-1)
        return LowResSliceTransform(
            resolution, thickness, axis, noise, **self.get_prm()
        ).make_final(x, max_depth-1)


class LowResTransform(NonFinalTransform):
    """Model a lower-resolution image"""

    class Final(FinalTransform):

        def __init__(self, resolution, noise=None, **kwargs):
            super().__init__(**kwargs)
            self.resolution = resolution
            self.noise = noise

        def apply(self, x):
            ndim = x.dim() - 1
            mode = ('trilinear' if ndim == 3 else
                    'bilinear' if ndim == 2 else
                    'linear')

            def interpol(x, shape):
                return interpolate(
                    x[None], size=shape, align_corners=True, mode=mode
                )[0]

            resolution = ensure_list(self.resolution, ndim)
            y = smoothnd(x, fwhm=resolution)
            factor = [1/r for r in resolution]
            ishape = x.shape[1:]
            oshape = [math.ceil(s*f) for s, f in zip(ishape, factor)]
            y = interpol(y, oshape)
            if self.noise is not None:
                y = self.noise(y)
            z = interpol(y, ishape)
            return prepare_output(
                dict(input=x, lowres=y, output=z),
                self.returns
            )

    def __init__(self, resolution=2, noise=None, **kwargs):
        """

        Parameters
        ----------
        resolution : float or list[float]
            Resolution of the low-res image, in terms of high-res voxels
        noise : Transform, optional
            A transform that adds noise in the low-resolution space
        returns : [list or dict of] {'input', 'lowres', 'output'}
            Which tensors to return.
        """
        super().__init__(**kwargs)
        self.resolution = resolution
        self.noise = noise

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        noise = None
        if self.noise:
            ndim = x.dim() - 1
            resolution = ensure_list(self.resolution, ndim)
            factor = [1/r for r in resolution]
            oshape = [math.ceil(s*f) for s, f in zip(x.shape[1:], factor)]
            fake_x = x.new_zeros([]).expand([len(x), *oshape])
            noise = self.noise.make_final(fake_x, max_depth-1)
        return self.Final(
            self.resolution, noise, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomLowResTransform(RandomizedTransform):
    """Random lower-resolution image"""

    def __init__(self, resolution=2, noise=None, *, shared=False, **kwargs):
        """
        Parameters
        ----------
        resolution : Sampler or float
            Distribution from which to sample the resolution.
            If a `float`, sample from `Uniform(1, value)`.
            To force a fixed value, pass `Fixed(value)`.
            The resolution is the distance (in terms of high-res voxels)
            between the centers of two consecutive low-res voxels.
        noise : Transform, optional
            A transform that adds noise in the low-resolution space

        Other Parameters
        ------------------
        returns : [list or dict of] {'input', 'lowres', 'output'}
            Which tensors to return.
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Use the same resolution for all channels/tensors
        """
        super().__init__(
            LowResTransform,
            dict(resolution=Uniform.make(make_range(1, resolution)),
                 noise=noise),
            shared=shared,
            **kwargs
        )
