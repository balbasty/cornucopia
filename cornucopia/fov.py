import torch
import math
from .base import Transform
from .utils.py import ensure_list
from .utils.padding import pad


class FlipTransform(Transform):
    """Flip one or more axes"""

    def __init__(self, axis=None, shared=True):
        """

        Parameters
        ----------
        axis : [list of] int
            Axes to flip
        shared : bool or {'channels', 'tensors'}
        """
        super().__init__(shared=shared)
        self.axis = axis

    def apply_transform(self, x, parameters):
        axis = self.axis
        if axis is None:
            axis = list(range(1, x.dim()+1))
        axis = ensure_list(axis)
        return x.flip(axis)


class PatchTransform(Transform):
    """Extract a patch from the volume"""

    def __init__(self, shape=64, center=0, bound='dct2', shared=True):
        """

        Parameters
        ----------
        shape : [list of] int
            Patch shape
        center : [list of] float
            Patch center, in relative coordinates -1..1
        bound : str
            Boundary condition in case padding is necessary
        shared : bool or {'channels', 'tensors'}
        """
        super().__init__(shared=shared)
        self.shape = shape
        self.center = center
        self.bound = bound

    def get_parameters(self, x):
        ndim = x.dim() - 1
        shape = ensure_list(self.shape, ndim)
        center = ensure_list(self.center, ndim)
        center_vox = [(s-1)/2 for s in x.shape[1:]]
        crop = []
        padding = []
        for ss, cc, cv, sv in zip(shape, center, center_vox, x.shape[1:]):
            first = int(math.floor(cv + cc - ss/2))
            pad_first = max(0, -first)
            last = first + ss
            pad_last = max(0, sv - last)
            first = max(0, first)
            last = min(sv, last)
            crop.append(slice(first, last))
            padding.extend([pad_first, pad_last])
        return crop, padding

    def apply_transform(self, x, parameters):
        crop, padding = parameters
        crop = tuple([Ellipsis, *crop])
        x = x[crop]
        x = pad(x, padding, mode=self.bound)
        return x

