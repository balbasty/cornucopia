"""
This module contains transforms that modify the resolution and/or
point spread function of an image.
"""
__all__ = [
    'SmoothTransform',
    'RandomSmoothTransform',
    'LowResSliceFinalTransform',
    'LowResSliceTransform',
    'RandomLowResSliceTransform',
    'LowResFinalTransform',
    'LowResTransform',
    'RandomLowResTransform',
]

# stdlib
import math
from math import inf

# dependencies
import torch
import typing_extensions as tx
from torch import Tensor
from torch.nn.functional import interpolate

# internals
from .base import Transform, FinalTransform, NonFinalTransform
from .special import RandomizedTransform
from .baseutils import Returned, prepare_output
from .utils.conv import smoothnd
from .utils.py import make_vector
from .random import Sampler, Uniform, RandInt, make_range
from . import typing as cct


class SmoothTransform(FinalTransform):
    """Apply Gaussian smoothing"""

    def __init__(self, fwhm: float = 1, **kwargs) -> None:
        """

        Parameters
        ----------
        fwhm : float
            Full-width at half-maximum of the Gaussian kernel

        Other Parameters
        ------------------
        returns : [(list | dict) of] {'input', 'output'}
            Which tensors to return.
        """
        super().__init__(**kwargs)
        self.fwhm = fwhm

    def xform(self, x: Tensor) -> Tensor:
        return smoothnd(x, fwhm=make_vector(self.fwhm, x.dim()-1))


class RandomSmoothTransform(RandomizedTransform):
    """Apply Gaussian smoothing with random FWHM"""

    Final = Next = SmoothTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        fwhm: cct.SamplerOrBound[float] = 2,
        *,
        shared: cct.SharedType = False,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        fwhm : Sampler | float
            Sampler or upper bound for the full-width at half-maximum

        Other Parameters
        ------------------
        returns : [(list | dict) of] {'input', 'output'}
            Which tensors to return.
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Use the same fwhm for all channels/tensors
        """
        super().__init__(SmoothTransform,
                         dict(fwhm=Uniform.make(make_range(0, fwhm))),
                         shared=shared, **kwargs)


class LowResSliceFinalTransform(FinalTransform):
    """Model a low-resolution slice direction, with Gaussian profile"""

    def __init__(
        self,
        axis: int = -1,
        resolution: float = 3.0,
        thickness: float = 0.8,
        noise: tx.Optional[Transform] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.axis = axis
        self.resolution = resolution
        self.thickness = thickness
        self.noise = noise

    @property
    def is_final(self) -> bool:
        return self.noise.is_final if self.noise else True

    def _get_smallshape(self, x: Tensor) -> tx.Sequence[int]:
        factor = [1] * (x.ndim-1)
        factor[self.axis] = 1/self.resolution
        ishape = x.shape[1:]
        return [
            (s*f).ceil().long().clamp_min(2).detach().item()
            if torch.is_tensor(f) else
            max(2, math.ceil(s*f))
            for s, f in zip(ishape, factor)
        ]

    def make_final(self, x, /, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        noise = None
        if self.noise:
            oshape = self._get_smallshape(x)
            fake_x = x.new_zeros([]).expand([len(x), *oshape])
            noise = self.noise.make_final(fake_x, max_depth-1)
            max_depth -= 1
        return type(self)(
            self.axis, self.resolution, self.thickness, noise, **self.get_prm()
        )

    def xform(self, x: Tensor) -> Returned:
        ndim = x.dim() - 1
        mode = ('trilinear' if ndim == 3 else
                'bilinear' if ndim == 2 else
                'linear')

        def interpol(x, shape):
            return interpolate(
                x[None], size=shape, align_corners=True, mode=mode
            )[0]

        ishape = x.shape[1:]
        oshape = self._get_smallshape(x)
        # 1. smooth
        fwhm = [0] * ndim
        fwhm[self.axis] = self.resolution * self.thickness
        y = smoothnd(x, fwhm=fwhm)
        # 2. downsample
        y = interpol(y, oshape)
        # 3. add noise in low-res space
        if self.noise:
            y = self.noise(y)
        # 4. upsample back to high-res space
        z = interpol(y, ishape)
        # > return
        return prepare_output(
            dict(input=x, lowres=y, output=z),
            self.returns
        )

class LowResSliceTransform(NonFinalTransform):
    """
    Model a low-resolution slice direction, with Gaussian profile
    """

    Final = Next = LowResSliceFinalTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        resolution: float = 3,
        thickness: float = 0.8,
        axis: int = -1,
        noise: tx.Optional[Transform] = None,
        **kwargs
    ) -> None:
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
        returns : [(list | dict) of] {'input', 'lowres', 'output'}
            Which tensors to return.
        """
        super().__init__(**kwargs)
        self.resolution = resolution
        self.noise = noise
        self.axis = axis
        self.thickness = thickness

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self

        noise = self.noise
        if noise:
            factor = [1] * (x.ndim-1)
            factor[self.axis] = 1/self.resolution
            ishape = x.shape[1:]
            oshape = [
                (s*f).ceil().long().clamp_min(2).detach().item()
                if torch.is_tensor(f) else
                max(2, math.ceil(s*f))
                for s, f in zip(ishape, factor)
            ]
            fake_x = x.new_zeros([]).expand([len(x), *oshape])
            if not noise.is_final:
                print("make_final:", fake_x.shape, noise, max_depth)
                noise = noise.make_final(fake_x, max_depth)
                max_depth -= 1
                print(noise)

        return self.Next(
            self.axis, self.resolution, self.thickness, noise,
            **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomLowResSliceTransform(RandomizedTransform):
    """Random low-resolution slice direction, with Gaussian profile"""

    Next = LowResSliceTransform
    """The transform type returned by `make_final(..., max_depth=1)`."""

    Final = LowResSliceFinalTransform
    """The transform type returned by `make_final(..., max_depth=inf)`."""

    def __init__(
        self,
        resolution: cct.SamplerOrBound[float] = 3,
        thickness: cct.SamplerOrBound[float] = 0.1,
        axis: tx.Union[Sampler, int, None] = None,
        noise: tx.Optional[Transform] = None,
        *,
        shared: cct.SharedType = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        resolution : Sampler | float
            Distribution from which to sample the resolution.
            If a `float`, sample from `Uniform(1, value)`.
            To force a fixed value, pass `Fixed(value)`.
            The resolution is the distance (in terms of high-res voxels)
            between the centers of two consecutive low-res slices.
        thickness : Sampler | float in 0..1
            Distribution from which to sample the resolution.
            If a `float`, sample from `Uniform(value, 1)`.
            To force a fixed value, pass `Fixed(value)`.
            Thickness is defined as a proportion of the resolution, and
            determines how much data is averaged into one low-resolution slice.
        axis : Sampler | int | None
            Slice axis. If None, select one randomly.
        noise : Transform | None
            A transform that adds noise in the low-resolution space

        Other Parameters
        ------------------
        returns : [(list | dict) of] {'input', 'lowres', 'output'}
            Which tensors to return.
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
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

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
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


class LowResFinalTransform(FinalTransform):
    """Model a lower-resolution image."""

    def __init__(
        self,
        resolution: cct.VectorLike,
        noise: tx.Optional[Transform] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        resolution : [list of] float
            Resolution of the low-res image, in terms of high-res voxels.
        noise : Transform | None
            A transform that adds noise in the low-resolution space

        Other Parameters
        ----------------
        returns : [(list | dict) of] {'input', 'lowres', 'output'}
            Which tensors to return.
        """
        super().__init__(**kwargs)
        self.resolution = resolution
        self.noise = noise

    def xform(self, x: Tensor) -> Returned:
        ndim = x.dim() - 1
        mode = ('trilinear' if ndim == 3 else
                'bilinear' if ndim == 2 else
                'linear')

        def interpol(x, shape):
            return interpolate(
                x[None], size=shape, align_corners=True, mode=mode
            )[0]

        resolution = make_vector(self.resolution, ndim)
        y = smoothnd(x, fwhm=resolution)
        factor = resolution.reciprocal()
        ishape = x.shape[1:]
        oshape = [
            (s*f).ceil().long().detach().item()
            for s, f in zip(ishape, factor)
        ]
        y = interpol(y, oshape)
        if self.noise is not None:
            y = self.noise(y)
        z = interpol(y, ishape)
        return prepare_output(
            dict(input=x, lowres=y, output=z),
            self.returns
        )

class LowResTransform(NonFinalTransform):
    """Model a lower-resolution image"""

    Final = Next = LowResFinalTransform
    """The transform type returned by `make_final`."""

    def __init__(
        self,
        resolution: cct.VectorLike = 2,
        noise: tx.Optional[Transform] = None,
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        resolution : [list of] float
            Resolution of the low-res image, in terms of high-res voxels
        noise : Transform | None
            A transform that adds noise in the low-resolution space

        Other Parameters
        ----------------
        returns : [(list | dict) of] {'input', 'lowres', 'output'}
            Which tensors to return.
        """
        super().__init__(**kwargs)
        self.resolution = resolution
        self.noise = noise

    def make_final(self, x: Tensor, max_depth: int = inf) -> Transform:
        if max_depth == 0:
            return self
        noise = None
        if self.noise:
            ndim = x.dim() - 1
            resolution = make_vector(self.resolution, ndim)
            factor = resolution.reciprocal()
            oshape = [
                (s*f).ceil().long().detach().item()
                for s, f in zip(x.shape[1:], factor)
            ]
            fake_x = x.new_zeros([]).expand([len(x), *oshape])
            noise = self.noise.make_final(fake_x, max_depth-1)
        return self.Next(
            self.resolution, noise, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomLowResTransform(RandomizedTransform):
    """Random lower-resolution image"""

    Next = LowResTransform
    """The transform type returned by `make_final(..., max_depth=1)`."""

    Final = LowResFinalTransform
    """The transform type returned by `make_final(..., max_depth=inf)`."""

    def __init__(
        self,
        resolution: cct.SamplerOrBound[float] = 2,
        noise: tx.Optional[Transform] = None,
        *,
        shared: cct.SharedType = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        resolution : Sampler | float
            Distribution from which to sample the resolution.
            If a `float`, sample from `Uniform(1, value)`.
            To force a fixed value, pass `Fixed(value)`.
            The resolution is the distance (in terms of high-res voxels)
            between the centers of two consecutive low-res voxels.
        noise : Transform | None
            A transform that adds noise in the low-resolution space

        Other Parameters
        ------------------
        returns : [(list | dict) of] {'input', 'lowres', 'output'}
            Which tensors to return.
        shared : {'channels', 'tensors', 'channels+tensors', ''} | bool
            Use the same resolution for all channels/tensors
        """
        super().__init__(
            LowResTransform,
            dict(resolution=Uniform.make(make_range(1, resolution)),
                 noise=noise),
            shared=shared,
            **kwargs
        )
