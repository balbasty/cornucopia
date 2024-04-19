__all__ = [
    'FlipTransform',
    'RandomFlipTransform',
    'PermuteAxesTransform',
    'RandomPermuteAxesTransform',
    'PatchTransform',
    'RandomPatchTransform',
    'CropTransform',
    'PadTransform',
    'PowerTwoTransform',
    'Rot90Transform',
    'Rot180Transform',
    'RandomRot90Transform',
]
import math
from random import shuffle
from .base import FinalTransform, NonFinalTransform, PerChannelTransform
from .utils.py import ensure_list
from .utils.padding import pad
from .random import Uniform, RandKFrom, Sampler, RandInt, make_range


class FlipTransform(FinalTransform):
    """Flip one or more axes"""

    def __init__(self, axis=None, **kwargs):
        """
        Parameters
        ----------
        axis : [list of] int
            Axes to flip. By default, flip all axes.
        """
        super().__init__(**kwargs)
        self.axis = axis

    def xform(self, x):
        axis = self.axis
        if axis is None:
            axis = list(range(1, x.ndim))
        axis = ensure_list(axis)
        return x.flip(axis)

    def make_inverse(self):
        return self


class RandomFlipTransform(NonFinalTransform):
    """Randomly flip one or more axes"""

    def __init__(self, axes=None, *, shared=True, **kwargs):
        """
        Parameters
        ----------
        axes : Sampler or [list of] int
            Axes that can be flipped (default: all)

        Other Parameters
        ----------------
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Apply the same flip to all channels and/or tensors
        """
        axes = kwargs.pop('axis', axes)
        super().__init__(shared=shared, **kwargs)
        self.axes = axes

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return PerChannelTransform(
                [self.make_final(x[i:i+1], max_depth) for i in range(len(x))],
                **self.get_prm()
            ).make_final(x, max_depth-1)
        axes = self.axes or range(1, x.ndim)
        if not isinstance(axes, Sampler):
            rand_axes = RandKFrom(ensure_list(axes))
        rand_axes = rand_axes()
        return FlipTransform(rand_axes).make_final(x, max_depth-1)


class PermuteAxesTransform(FinalTransform):
    """Permute axes"""

    def __init__(self, permutation=None, **kwargs):
        """
        Parameters
        ----------
        permutation : [list of] int
            Axes permutation. By default, reverse axes.
            Only applies to spatial axes, so axes are numbered [C, 0, 1, 2]
        """
        super().__init__(**kwargs)
        self.permutation = permutation

    def xform(self, x):
        permutation = self.permutation
        if permutation is None:
            permutation = list(reversed(range(x.dim()-1)))
        permutation = [0] + [p+1 for p in permutation]
        return x.permute(permutation)

    def make_inverse(self):
        if self.permutation:
            i = range(len(self.permutation))
            iperm = [i[p] for p in self.permutation]
            return PermuteAxesTransform(iperm, **self.get_prm())
        else:
            return self


class RandomPermuteAxesTransform(NonFinalTransform):
    """Randomly permute axes"""

    def __init__(self, axes=None, *, shared=True, **kwargs):
        """
        Parameters
        ----------
        axes : [list of] int
            Axes that can be permuted (default: all)

        Other Parameters
        ----------------
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Apply the same permutation to all channels and/or tensors
        """
        super().__init__(shared=shared, **kwargs)
        self.axes = axes

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return PerChannelTransform(
                [self.make_final(x[i:i+1], max_depth) for i in range(len(x))],
                **self.get_prm()
            ).make_final(x, max_depth-1)
        axes = list(self.axes or range(x.ndim-1))
        shuffle(axes)
        return PermuteAxesTransform(
            axes, **self.get_prm()
        ).make_final(x, max_depth-1)


class Rot90Transform(FinalTransform):
    """
    Apply a 90 (or 180) rotation along one or several axes
    """

    def __init__(self, axis=0, negative=False, double=False, **kwargs):
        """
        Parameters
        ----------
        axis : int or list[int]
            Rotation axis (indexing does not account for the channel axis)
        negative : bool or list[bool]
            Rotate by -90 deg instead of 90 deg
        double : bool or list[bool]
            Rotate be 180 instead of 90 (`negative` is then unused)
        """
        super().__init__(**kwargs)
        self.axis = ensure_list(axis)
        self.negative = ensure_list(negative, len(self.axis))
        self.double = ensure_list(double, len(self.axis))

    def apply(self, x):
        # this implementation is suboptimal. We should fuse all transpose
        # and all flips into a single "transpose + flip" operation so that
        # a single allocation happens. This will be fine for now.

        ndim = x.ndim - 1
        axis = [1 + (ndim + a if a < 0 else a) for a in self.axis]
        for ax, neg, dbl in zip(axis, self.negative, self.double):
            if dbl:
                if ndim == 2:
                    dims = [1, 2]
                else:
                    assert ndim == 3
                    dims = [d for d in (1, 2, 3) if d != ax]
                x = x.flip(dims)
            else:
                if ndim == 2:
                    dims = [1, 2]
                else:
                    assert ndim == 3
                    dims = [d for d in (1, 2, 3) if d != ax]
                x = x.transpose(*dims).flip(dims[1] if neg else dims[0])
        return x


class Rot180Transform(Rot90Transform):
    """Apply a 180 deg rotation along one or several axes"""

    def __init__(self, axis=0, **kwargs):
        """
        Parameters
        ----------
        axis : int or list[int]
            Rotation axis (indexing does not account for the channel axis)
        """
        super().__init__(axis, double=True, **kwargs)


class RandomRot90Transform(NonFinalTransform):
    """Random set of 90 transforms"""

    def __init__(self, axes=None, max_rot=2, negative=True,
                 *, shared=True, **kwargs):
        """
        Parameters
        ----------
        axes : int or list[int]
            Axes along which rotations can happen.
            If `None`, all axes.
        max_rot : int or Sampler
            Maximum number of consecutive rotations.
        negative : bool
            Whether to authorize negative rotations.

        Other Parameters
        ----------------
        shared : {'channels', 'tensors', 'channels+tensors', ''}
            Apply the same permutation to all channels and/or tensors
        """
        super().__init__(shared=shared, **kwargs)
        self.axes = axes
        self.max_rot = RandInt.make(make_range(1, max_rot))
        self.negative = negative

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        if 'channels' not in self.shared and len(x) > 1:
            return PerChannelTransform(
                [self.make_final(x[i:i+1], max_depth) for i in range(len(x))],
                **self.get_prm()
            ).make_final(x, max_depth-1)
        ndim = x.ndim - 1
        max_rot = self.max_rot
        if isinstance(max_rot, Sampler):
            max_rot = max_rot()
        axes = self.axes
        if axes is None:
            axes = list(range(ndim))
        if isinstance(axes, (int, list, tuple)):
            axes = ensure_list(axes, max_rot, crop=False)
        if not isinstance(axes, Sampler):
            axes = RandKFrom(axes, max_rot, replacement=True)

        axes = ensure_list(axes(), max_rot)
        negative = RandKFrom([False, True], max_rot, replacement=True)() \
            if self.negative else [False] * max_rot
        return Rot90Transform(
            axes, negative, **self.get_prm()
        ).make_final(max_depth-1)


class CropPadTransform(FinalTransform):
    """Crop and/or pad a tensor"""

    def __init__(self, crop, pad, bound='zero', value=0, **kwargs):
        """
        Parameters
        ----------
        crop : list[slice]
            Slicing operator per dimension.
        pad : list[int]
            Left and right padding per dimensions
        bound : [list of] str
            Boundary condition for padding
        value : number
            Padding value in case `bound='constant`
        """
        super().__init__(**kwargs)
        self.crop = crop
        self.pad = pad
        self.bound = bound
        self.value = value

    def xform(self, x):
        crop = tuple([Ellipsis, *self.crop])
        x = x[crop]
        x = pad(x, self.pad, mode=self.bound, value=self.value)
        return x

    def make_inverse(self):
        ipad = [slice(left, (-right) or None) for left, right in self.pad]
        icrop = [[s.start or 0, -s.stop if s.stop else 0] for s in self.crop]
        return CropPadTransform(
            ipad, icrop, bound=self.bound, value=self.value, **self.get_prm()
        )


class PatchTransform(NonFinalTransform):
    """Extract a patch from the volume"""

    def __init__(self, shape=64, center=0, bound='zero',
                 *, shared='channels', **kwargs):
        """
        Parameters
        ----------
        shape : [list of] int
            Patch shape
        center : [list of] float
            Patch center, in relative coordinates -1..1
        bound : str
            Boundary condition in case padding is needed

        Other Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensor', ''}
        """
        kwargs.setdefault('shared', shared)
        super().__init__(**kwargs)
        self.shape = shape
        self.center = center
        self.bound = bound

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        ndim = x.dim() - 1
        shape = ensure_list(self.shape, ndim)
        center = ensure_list(self.center, ndim)
        center = [(c + 1) / 2 * (s - 1) for c, s in zip(center, x.shape[1:])]
        crop = []
        padding = []
        for ss, cc, sv in zip(shape, center, x.shape[1:]):
            first = int(math.floor(cc - ss/2))
            pad_first = max(0, -first)
            last = first + ss
            pad_last = max(0, last - sv)
            first = max(0, first)
            last = min(sv, last)
            last = (last - sv) or None  # ensure negative for CropPad
            crop.append(slice(first, last))
            padding.extend([pad_first, pad_last])
        return CropPadTransform(
            crop, padding, bound=self.bound, **self.get_prm()
        ).make_final(x, max_depth-1)


class RandomPatchTransform(NonFinalTransform):
    """Extract a (randomly located) patch from the volume.

    This transform ensures that the patch is fully contained within the
    original field of view (unless the patch size is larger than the
    input shape).
    """

    def __init__(self, patch_size, bound='zero', *, shared=True, **kwargs):
        """

        Parameters
        ----------
        shape : [list of] int
            Patch shape
        bound : str
            Boundary condition in case padding is needed

        Other Parameters
        ------------------
        shared : {'channels', 'tensors', 'channels+tensors', None}
            Extract the same patch from all channels and/or tensors
        """
        kwargs.setdefault('shared', shared)
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.bound = bound

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        shape = x.shape[1:]
        patch_size = ensure_list(self.patch_size, len(shape))
        min_center = [max(p/s - 1, -1) for p, s in zip(patch_size, shape)]
        max_center = [min(1 - p/s, 1) for p, s in zip(patch_size, shape)]
        center = [Uniform(mn, mx)() for mn, mx in zip(min_center, max_center)]
        return PatchTransform(
            patch_size, center, self.bound, **self.get_prm()
        ).make_final(x, max_depth-1)


class CropTransform(NonFinalTransform):
    """Crop a tensor by some amount"""

    def __init__(self, cropping, unit='vox', side='both', **kwargs):
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
        """
        kwargs.setdefault('shared', 'channels')
        super().__init__(**kwargs)
        self.cropping = cropping
        self.unit = unit
        self.side = side

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        ndim = x.dim() - 1
        cropping = self.cropping
        if self.side is not None:
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
                            for c, s in zip(cropping, shape2)]
            cropping = [slice(c0, -c1 if c1 else None)
                        for c0, c1 in zip(cropping[::2], cropping[1::2])]
        return CropPadTransform(
            cropping, [0]*(2*ndim), **self.get_prm()
        ).make_final(x, max_depth-1)


class PadTransform(NonFinalTransform):
    """Pad a tensor by some amount"""

    def __init__(self, padding, unit='vox', side='both', bound='zero', value=0,
                 **kwargs):
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
            Value for case `bound='constant'`
        """
        kwargs.setdefault('shared', 'channels')
        super().__init__(**kwargs)
        self.padding = padding
        self.unit = unit
        self.side = side
        self.bound = bound
        self.value = value

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        ndim = x.dim() - 1
        padding = self.padding
        if self.side is not None:
            padding = ensure_list(padding, ndim)
            if self.unit[0] == 'p':
                padding = [int(math.ceil(p * s))
                           for p, s in zip(padding, x.shape[1:])]

        else:
            padding = ensure_list(padding)
            padding = [0] * (2 * ndim - len(padding)) + padding
            if self.unit[0] == 'p':
                shape2 = [s for s in x.shape[1:] for _ in range(2)]
                padding = [int(math.ceil(p * s))
                           for p, s in zip(padding, shape2)]

        if self.side == 'pre':
            padding = [p for pz in zip(padding, [0]*ndim) for p in pz]
        elif self.side == 'post':
            padding = [p for zp in zip([0]*ndim, padding) for p in zp]
        elif self.side == 'both':
            padding = [p for pp in zip(padding, padding) for p in pp]

        return CropPadTransform(
            [slice(None)]*ndim, padding, bound=self.bound, value=self.value,
            **self.get_prm()
        ).make_final(x, max_depth-1)


class PowerTwoTransform(NonFinalTransform):
    """Pad the volume such that the tensor shape can be divided by 2**x"""

    def __init__(self, exponent=1, bound='zero', **kwargs):
        """

        Parameters
        ----------
        exponent : [list of] int
            Ensure that the shape can be divided by 2 ** exponent
        bound : [list of] str
            Boundary condition for padding
        """
        kwargs.setdefault('shared', 'channels')
        super().__init__(**kwargs)
        self.exponent = exponent
        self.bound = bound

    def make_final(self, x, max_depth=float('inf')):
        if max_depth == 0:
            return self
        shape = x.shape[1:]
        exponent = ensure_list(self.exponent, len(shape))
        bigshape = [max(2 ** e, s) for e, s in zip(exponent, shape)]
        return PatchTransform(
            bigshape, bound=self.bound, **self.get_prm()
        ).make_final(x, max_depth-1)
