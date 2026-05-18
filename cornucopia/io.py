"""This module contains transforms that load data from disk."""
__all__ = ['ToTensorTransform', 'LoadTransform']
# stdlib
import os.path
from os import PathLike
from typing import Optional, Union, List
from pathlib import Path

# dependencies
import torch
from torch import Tensor

# internals
from .base import FinalTransform
from .utils.io import loaders


class ToTensorTransform(FinalTransform):
    """Convert to Tensor (or to other dtype/device)"""

    def __init__(
        self,
        ndim: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        ndim : int, optional
            Number of spatial dimensions (default: guess from data)
        dtype : torch.dtype, optional
            Returned data type (default: keep same)
        device : torch.device, optional
            Returned device (default: keep same)

        Other Parameters
        ----------------
        returns, append, prefix, include, exclude, consume
            See [`Transform`][cornucopia.base.Transform] for details.
        """
        super().__init__(**kwargs)
        self.dim = ndim
        self.dtype = dtype
        self.device = device

    def xform(self, x: Tensor) -> Tensor:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device).squeeze()
        if self.dim:
            for _ in range(max(0, self.dim + 1 - x.ndim)):
                x = x[None]
            if x.ndim > self.dim + 1:
                raise ValueError(f'Too many dimensions: '
                                 f'{x.ndim} > 1 + {self.dim}')
        return x


class LoadTransform(FinalTransform):
    """
    Load data from disk.

    Available loaders are:

    - `BabelLoader`: for medical image formats (nifti, mgz, minc, etc.)
    - `TiffLoader`: for TIFF files (including multi-page)
    - `PillowLoader`: for common image formats (png, jpg, etc., with optional rot90)
    - `NumpyLoader`: for .npy and .npz files (with optional field name)

    Custom loaders can be added by registering them in
    `cornucopia.utils.io.loaders`.
    """

    def __init__(
        self,
        ndim: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        device: Optional[Union[torch.device, str]] = None,
        returns: Optional[List[str]] = None,
        append: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        ndim : int | None
            Number of spatial dimensions (default: guess from file)
        dtype : torch.dtype | str | None
            Data type (default: guess from file)
        device : torch.device | str | None
            Device on which to load data (default: cpu)

        Other Parameters
        ------------------
        to_ras : bool, default=True
            Reorient data so that it has a RAS layout.
            Only used by `BabelLoader`.
        rot90 : bool, default=True
            Rotate by 90 degrees in-plane.
            Only used by `PillowLoader`.
        field : str, default="arr_0"
            Field to load from a npz file.
            Only used by `NumpyLoader`.
        returns, append, prefix, include, exclude, consume
            See [`Transform`][cornucopia.base.Transform] for details.
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

    def xform(self, x: Union[str, Tensor]) -> Tensor:
        try:
            return torch.as_tensor(x, dtype=self.dtype, device=self.device)
        except Exception:
            pass

        if isinstance(x, str):
            x = Path(x)

        exceptions = []
        if isinstance(x, PathLike):
            parts, ext = os.path.splitext(str(x))
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
