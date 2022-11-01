import torch
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

    def __init__(self, ndim=3, dtype=None, device=None, to_ras=True, **kwargs):
        super().__init__(ndim, dtype, device)
        self.to_ras = to_ras

    def __call__(self, x):
        f = nibabel.load(x)
        x = self.convert(f.get_fdata())

        # reorient to RAS layout
        if self.to_ras:
            aff = torch.as_tensor(f.affine)
            perm = aff[:3, :3].abs().argmax(1).tolist()
            perm += list(range(3, x.dim()))
            x = x.permute(perm)
            aff = aff[:, perm + [-1]]
            flipdims = (aff[:3, :3].diag() < 0).nonzero().flatten().tolist()
            x = x.flip(flipdims)

        channel_dims = 0
        if x.dim() > 3 and x.shape[3] > 1:
            channel_dims += 1
            x = x.movedim(3, 0)
        if x.dim() > 4 and x.shape[4] > 1:
            channel_dims += 1
            x = x.movedim(4, 0)
        x = x.squeeze()
        x = x.reshape([-1, *x.shape[channel_dims:]])
        if self.ndim:
            while x.dim() < self.ndim + 1:
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
        for i in range(x.dim()):
            if i not in perm and i not in dimzyx:
                perm.append(i)
        perm += dimzyx

        x = x.permute(*perm)

        x = x.squeeze()
        x = x.reshape([-1, *x.shape[-len(dimzyx):]])
        if self.ndim:
            while x.dim() < self.ndim + 1:
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
        if x.dim() > 2:
            x = x.movedim(-1, 0)
        else:
            x = x[None]
        if self.rot90:
            x = x.transpose(-1, -2).flip(-1)
        if self.ndim:
            while x.dim() < self.ndim + 1:
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
            while x.dim() < self.ndim + 1:
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
