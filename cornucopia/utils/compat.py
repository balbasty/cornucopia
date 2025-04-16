from typing import Optional
import torch

Tensor = torch.Tensor


torch_version = torch.__version__
torch_version = torch_version.split("+")[0]     # remove local modifier
torch_version = torch_version.split(".")[:2]    # major + minor
torch_version = tuple(map(int, torch_version))  # integer


if torch_version < (1, 9):
    def clamp(
        x: Tensor,
        min: Optional[Tensor] = None,
        max: Optional[Tensor] = None,
        **kwargs
    ):
        if torch.is_tensor(min):
            x = torch.maximum(x, min, **kwargs)
            min = None
        if torch.is_tensor(max):
            x = torch.minimum(x, max, **kwargs)
            max = None
        if min is not None or max is not None:
            x = x.clamp(min, max, **kwargs)
        return x
else:
    clamp = torch.clamp
