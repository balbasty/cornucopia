__all__ = ['ToTensorTransform', 'LoadTransform']

import torch
import os.path
from .base import Transform
from .utils.io import loaders


class ToTensorTransform(Transform):
    """Convert to Tensor (or to other dtype/device)"""

    def __init__(self, dim=None, dtype=None, device=None):
        """

        Parameters
        ----------
        dtype : torch.dtype, optional
        device : torch.device, optional
        """
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.device = device

    def apply_transform(self, x, parameters):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device).squeeze()
        if self.dim:
            for _ in range(max(0, self.dim + 1 - x.dim())):
                x = x[None]
            if x.dim() > self.dim + 1:
                raise ValueError(f'Too many dimensions: '
                                 f'{x.dim()} > 1 + {self.dim}')
        return x


class LoadTransform(Transform):

    def __init__(self, ndim=None, dtype=None, device=None):
        super().__init__(shared='channels')
        self.ndim = ndim
        self.dtype = dtype
        self.device = device

    def apply_transform(self, x, parameters):
        try:
            return torch.as_tensor(x, dtype=self.dtype, device=self.device)
        except Exception:
            pass

        if isinstance(x, str):
            parts, ext = os.path.splitext(x)
            if ext.lower() in ('.gz', '.bz', '.bz2', '.gzip', '.bzip2'):
                _, preext = os.path.splitext(parts)
                ext = preext + ext
            ext = ext.lower()
            if ext in loaders:
                for loader in loaders[ext]:
                    try:
                        return loader(self.ndim, self.dtype, self.device)(x)
                    except Exception as e:
                        pass

        all_loaders = set(loader for loader_ext in loaders.values()
                          for loader in loader_ext)
        for loader in all_loaders:
            try:
                return loader(self.ndim, self.dtype, self.device)(x)
            except Exception as e:
                pass

        raise ValueError(f'Could not load {x}')



