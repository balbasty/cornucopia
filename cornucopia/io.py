import torch
from .base import Transform


__all__ = ['ToTensorTransform']


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
