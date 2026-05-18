# stdlib
import itertools

# dependencies
import torch

from cornucopia.utils.py import make_vector

# internals
from .warps import affine_flow, apply_flow

# optionals
try:
    import nibabel
except ImportError:
    nibabel = None

try:
    import pillow
    import numpy as np
except ImportError:
    pillow = None

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import numpy as np
except ImportError:
    np = None


class Loader:
    """Base class for file loaders"""
    def __init__(self, ndim=None, dtype=None, device=None, **kwargs):
        self.ndim = ndim
        self.dtype = dtype
        self.device = device

    def convert(self, x):
        dtype = self.dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        return torch.as_tensor(x, dtype=dtype, device=self.device)


class BabelLoader(Loader):
    """Loader with nibabel backend"""

    EXT = ['.nii', '.nii.gz', '.mgh', '.mgz', '.mnc', '.img', '.hdr']

    def __init__(
        self,
        ndim=3,
        dtype=None,
        device=None,
        to_ras=True,  # or 'reslice'
        **kwargs
    ):
        super().__init__(ndim, dtype, device)
        self.to_ras = to_ras

    def __call__(self, x):

        # --- Load into a tensor ---
        f = nibabel.load(x)
        x = self.convert(f.get_fdata())

        # --- reorient to RAS layout ---

        shape = x.shape[-3:]
        aff = torch.as_tensor(f.affine)
        voxel_size = (aff[:3, :3] ** 2).sum(0) ** 0.5

        if self.to_ras is True:
            # Permute and flip to match RAS layout
            perm = aff[:3, :3].abs().argmax(1).tolist()
            perm += list(range(3, x.ndim))
            x = x.permute(perm)
            aff = aff[:, perm + [-1]]
            flipdims = (aff[:3, :3].diag() < 0).nonzero().flatten().tolist()
            x = x.flip(flipdims)

        elif self.to_ras:
            # Reslice on a canonical RAS grid

            if self.to_ras == 'reslice':
                # Preserve RAS voxel size
                perm = aff[:3, :3].abs().argmax(1).tolist()
                perm += list(range(3, x.ndim))
                voxel_size = voxel_size[perm]
            else:
                # Provided RAS voxel size
                voxel_size = make_vector(
                    self.to_ras, 3, dtype=aff.dtype, device=aff.device)

            aff0 = torch.eye(4)
            aff0[[0, 1, 2], [0, 1, 2]] = voxel_size
            vox2vox = aff0.inverse().matmul(aff)

            corners = itertools.product([False, True], repeat=self.ndim)
            corners = [
                [shape[i] - 1 if top else 0 for i, top in enumerate(c)] + [1]
                for c in corners
            ]
            corners = np.asarray(corners).T
            corners = vox2vox[:3, :].matmul(corners)
            mx = torch.max(corners, dim=1).values.floor().to(torch.int64)
            mn = torch.min(corners, dim=1).values.ceil().to(torch.int64)
            shape = (mx - mn + 1).tolist()
            offset = np.eye(4)
            offset[:3, -1] = mn
            aff = aff0 @ offset

            flow = affine_flow(aff0, shape, with_identity=True)
            x = apply_flow(x[None], flow, has_identity=True)[0]

        # --- squeeze axes to match (C, *spatial) ---

        channel_dims = 0
        if x.ndim > 3 and x.shape[3] > 1:
            channel_dims += 1
            x = x.movedim(3, 0)
        if x.ndim > 4 and x.shape[4] > 1:
            channel_dims += 1
            x = x.movedim(4, 0)
        x = x.squeeze()
        x = x.reshape([-1, *x.shape[channel_dims:]])
        if self.ndim:
            while x.ndim < self.ndim + 1:
                x = x[..., None]
        return x


class TiffLoader(Loader):
    """Loader with tifffile backend"""

    EXT = ['.tiff', '.tif']

    def __call__(self, x):
        with tifffile.TiffFile(x) as f:
            x = self.convert(f.asarray(series=0, level=0))
            axes = f.series[self.series].levels[self.level].axes
        perm = []
        if 'C' in axes:
            perm.append(axes.index('C'))
        if 'T' in axes:
            perm.append(axes.index('T'))
        dimzyx = [axes.index(A) for A in 'ZYX' if A in axes]
        for i in range(x.ndim):
            if i not in perm and i not in dimzyx:
                perm.append(i)
        perm += dimzyx

        x = x.permute(*perm)

        x = x.squeeze()
        x = x.reshape([-1, *x.shape[-len(dimzyx):]])
        if self.ndim:
            while x.ndim < self.ndim + 1:
                x = x[..., None]
        return x


class PillowLoader(Loader):
    """Loader with pillow backend"""

    EXT = ['.bmp', '.eps', '.gif', '.icns', '.ico', '.jpg', '.jpeg',
           '.png', '.apng']

    def __init__(self, ndim=None, dtype=None, device=None, rot90=True, **kwargs):
        super().__init__(ndim, dtype, device)
        self.rot90 = rot90

    def __call__(self, x):
        with pillow.Image.open(x) as f:
            f.load()
            x = self.convert(np.array(f))
        if x.ndim > 2:
            x = x.movedim(-1, 0)
        else:
            x = x[None]
        if self.rot90:
            x = x.transpose(-1, -2).flip(-1)
        if self.ndim:
            while x.ndim < self.ndim + 1:
                x = x[..., None]
        return x


class NumpyLoader(Loader):
    """Loader with numpy backend"""

    EXT = ['.npy', '.npz']

    def __init__(self, ndim=None, dtype=None, device=None, field=None):
        super().__init__(ndim, dtype, device)
        self.field = field

    def __call__(self, x):
        reader = np.load(x)
        if np.isarray(reader):
            x = reader
        else:
            field = self.field or 'arr_0'
            if field in reader.keys():
                x = reader[field]
                reader.close()
            else:
                reader.close()
                raise ValueError(f'No field "{field}" in numpy file')
        x = self.convert(x)
        if self.ndim:
            while x.ndim < self.ndim + 1:
                x = x[None]
        return x


loaders = {}


def register_loader(klass, ext=None, priority=False):
    if not ext:
        if hasattr(klass, 'EXT'):
            ext = klass.EXT
        else:
            return
    if isinstance(ext, (list, tuple)):
        for ext1 in ext:
            register_loader(klass, ext1, priority)
        return
    if ext not in loaders:
        loaders[ext] = []
    if priority:
        loaders[ext].insert(klass, 0)
    else:
        loaders[ext].append(klass)


if nibabel:
    register_loader(BabelLoader)
if tifffile:
    register_loader(TiffLoader)
if pillow:
    register_loader(PillowLoader)
if np:
    register_loader(NumpyLoader)
