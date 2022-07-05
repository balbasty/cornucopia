from .base import Transform
from .utils.conv import smoothnd
from .utils.py import ensure_list
from interpol import resize


__all__ = ['SmoothTransform', 'LowResTransform', 'LowResSliceTransform']


class SmoothTransform(Transform):
    """Apply Gaussian smoothing"""

    def __init__(self, fwhm=1):
        super().__init__()
        self.fwhm = fwhm

    def apply_transform(self, x, parameters):
        return smoothnd(x, fwhm=ensure_list(self.fwhm, x.dim()-1))


class LowResSliceTransform(Transform):
    """
    Model a low-resolution slice direction, with Gaussian profile
    """

    def __init__(self, resolution=3, thickness=0.8, axis=-1, noise=None):
        """

        Parameters
        ----------
        resolution : float
            Resolution of the slice dimension
        thickness : float in 0..1
            Slice thickness as a proportion of resolution
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


class LowResTransform(Transform):
    """Model a lower-resolution image"""

    def __init__(self, resolution=2, noise=None):
        """

        Parameters
        ----------
        resolution : float or list[float]
            Resolution
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
