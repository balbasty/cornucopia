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
            Boundary condition in case padding is needed
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


class CropTransform(Transform):
    """Crop a tensor by some amount"""

    def __init__(self, cropping, unit='vox', side='both', shared=True):
        """

        Parameters
        ----------
        cropping : [list of] int or float
            Amount of cropping. If `side` is `None`, pre and post cropping
            must be provided in turn.
        unit : {'vox', 'pct'}
            Padding unit
        side : {'pre', 'post', 'both', None}
            Side to crop
        shared
        """
        super().__init__(shared=shared)
        self.cropping = cropping
        self.unit = unit
        self.side = side

    def apply_transform(self, x, parameters):
        ndim = x.dim() - 1
        cropping = self.cropping
        if side is not None:
            cropping = ensure_list(cropping, ndim)
            if self.unit[0] == 'p':
                cropping = [int(math.ceil(c * s))
                            for c, s in zip(cropping, x.shape[1:])]
            cropping = [slice(c, -c if c else None) for c in cropping]
        else:
            cropping = ensure_list(cropping)
            cropping = [0] * (2*ndim - len(cropping))
            if self.unit[0] == 'p':
                shape2 = [s for s in x.shape[1:] for _ in range(2)]
                cropping = [int(math.ceil(c * s))
                            for p, s in zip(cropping, shape2)]
            cropping = [slice(c0, -c1 if c1 else None)
                        for c0, c1 in zip(cropping[::2], cropping[1::2])]

        cropping = (Ellipsis, *cropping)
        return x[cropping]


class PadTransform(Transform):
    """Pad a tensor by some amount"""

    def __init__(self, padding, unit='vox', side='both', bound='dct2', value=0,
                 shared=True):
        """

        Parameters
        ----------
        padding : [list of] int or float
            Amount of padding. If `side` is `None`, pre and post padding
            must be provided in turn.
        unit : {'vox', 'pct'}
            Padding unit
        side : {'pre', 'post', 'both', None}
            Side to pad
        bound : str
            Boundary condition
        value : float
            Value for case `bound='const'`
        shared
        """
        super().__init__(shared=shared)
        self.padding = padding
        self.unit = unit
        self.side = side
        self.bound = bound
        self.value = value

    def apply_transform(self, x, parameters):
        ndim = x.dim() - 1
        padding = self.padding
        if side is not None:
            padding = ensure_list(padding, ndim)
            if self.unit[0] == 'p':
                padding = [int(math.ceil(p * s))
                           for p, s in zip(padding, x.shape[1:])]
        else:
            padding = ensure_list(padding)
            padding = [0] * (2 * ndim - len(padding))
            if self.unit[0] == 'p':
                shape2 = [s for s in x.shape[1:] for _ in range(2)]
                padding = [int(math.ceil(p * s))
                           for p, s in zip(padding, shape2)]

        return pad(x, padding, mode=bound, side=self.side, value=self.value)


class PowerTwoTransform(Transform):
    """Pad the volume such that the tensor shape can be divided by 2**x"""

    def __init__(self, exponent=1, bound='dct2', shared='channels'):
        """

        Parameters
        ----------
        exponent : [list of] int
            Ensure that the shape can be divided by 2 ** exponent
        bound : [list of] str
            Boundary condition for padding
        shared : bool or {'channels', 'tensors'}
        """
        super().__init__(shared=shared)
        self.exponent = exponent
        self.bound = bound

    def get_parameters(self, x):
        shape = x.shape[1:]
        exponent = ensure_list(self.exponent, len(shape))
        bigshape = [max(2 ** e, s) for s, e in zip(exponent, shape)]
        return bigshape

    def apply_transform(self, x, parameters):
        return PatchTransform(parameters, bound=self.bound)(x)


