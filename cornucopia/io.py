__all__ = ['ToTensorTransform', 'LoadTransform']

import torch
import os.path
from .base import FinalTransform
from .utils.io import loaders


class ToTensorTransform(FinalTransform):
    """Convert to Tensor (or to other dtype/device)"""

    def __init__(self, dim=None, dtype=None, device=None, **kwargs):
        """
        Parameters
        ----------
        dtype : torch.dtype, optional
        device : torch.device, optional
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.dtype = dtype
        self.device = device

    def apply(self, x):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device).squeeze()
        if self.dim:
            for _ in range(max(0, self.dim + 1 - x.dim())):
                x = x[None]
            if x.dim() > self.dim + 1:
                raise ValueError(f'Too many dimensions: '
                                 f'{x.dim()} > 1 + {self.dim}')
        return x


class LoadTransform(FinalTransform):
    """
    Load data from disk
    """

    def __init__(self, ndim=None, dtype=None, *, device=None,
                 returns=None, append=False, include=None, exclude=None,
                 **kwargs):
        """
        Parameters
        ----------
        ndim : int, optional
            Number of spatial dimensions (default: guess from file)
        dtype : str or torch.dtype, optional
            Data type (default: guess from file)
        device : str or torch.device
            Device on which to load data (default: cpu)

        Other Parameters
        ------------------
        to_ras : bool, default=True
            Reorient data so that it has a RAS layout.
            Only used by Babel reader.
        rot90 : bool, default=True
            Rotate by 90 degrees in-plane.
            Only used by Pillow reader.
        field : str, default="arr_0"
            Field to load from a npz file.
            Only used by Numpy reader.
        """
        super().__init__(
            returns=returns,
            append=append,
            include=include,
            exclude=exclude,
        )
        self.ndim = ndim
        self.dtype = dtype
        self.device = device
        self.kwargs = kwargs

    def apply(self, x):
        try:
            return torch.as_tensor(x, dtype=self.dtype, device=self.device)
        except Exception:
            pass

        exceptions = []
        if isinstance(x, str):
            parts, ext = os.path.splitext(x)
            if ext.lower() in ('.gz', '.bz', '.bz2', '.gzip', '.bzip2'):
                _, preext = os.path.splitext(parts)
                ext = preext + ext
            ext = ext.lower()
            if ext in loaders:
                for loader in loaders[ext]:
                    try:
                        return loader(self.ndim, self.dtype, self.device,
                                      **self.kwargs)(x)
                    except Exception as e:
                        exceptions.append(str(e))
                        pass

        all_loaders = set(loader for loader_ext in loaders.values()
                          for loader in loader_ext)
        for loader in all_loaders:
            try:
                return loader(self.ndim, self.dtype, self.device)(x)
            except Exception as e:
                exceptions.append(str(e))
                pass

        message = [f'Could not load {x}:'] + exceptions
        message = '\n'.join(message)
        raise ValueError(message)
