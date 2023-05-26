import os
import torch
from torch import Tensor
from typing import List
from .version import torch_version


@torch.jit.script
def list_reverse_int(x: List[int]) -> List[int]:
    if len(x) == 0:
        return x
    return [x[i] for i in range(-1, -len(x)-1, -1)]


@torch.jit.script
def list_cumprod_int(x: List[int], reverse: bool = False,
                     exclusive: bool = False) -> List[int]:
    if len(x) == 0:
        lx: List[int] = []
        return lx
    if reverse:
        x = list_reverse_int(x)

    x0 = 1 if exclusive else x[0]
    lx = [x0]
    all_x = x[:-1] if exclusive else x[1:]
    for x1 in all_x:
        x0 = x0 * x1
        lx.append(x0)
    if reverse:
        lx = list_reverse_int(lx)
    return lx


@torch.jit.script
def sub2ind_list(subs: List[Tensor], shape: List[int]):
    """Convert sub indices (i, j, k) into linear indices.

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    subs : (D,) list[tensor]
        List of sub-indices. The first dimension is the number of dimension.
        Each element should have the same number of elements and shape.
    shape : (D,) list[int]
        Size of each dimension. Its length should be the same as the
        first dimension of ``subs``.

    Returns
    -------
    ind : (...) tensor
        Linear indices
    """
    ind = subs[-1]
    subs = subs[:-1]
    ind = ind.clone()
    stride = list_cumprod_int(shape[1:], reverse=True, exclusive=False)
    for i, s in zip(subs, stride):
        ind += i * s
    return ind


if torch_version('>=', (1, 10)):

    if not int(os.environ.get('PYTORCH_JIT', '1')):
        def meshgrid_list_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(*x, indexing='ij')

        def meshgrid_list_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(*x, indexing='xy')

    else:
        @torch.jit.script
        def meshgrid_list_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x, indexing='ij')

        @torch.jit.script
        def meshgrid_list_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x, indexing='xy')

else:
    if not int(os.environ.get('PYTORCH_JIT', '1')):
        def meshgrid_list_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x)

        def meshgrid_list_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            grid = torch.meshgrid(*x)
            if len(grid) > 1:
                grid[0] = grid[0].transpose(0, 1)
                grid[1] = grid[1].transpose(0, 1)
            return grid

    else:
        @torch.jit.script
        def meshgrid_list_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x)

        @torch.jit.script
        def meshgrid_list_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            grid = torch.meshgrid(x)
            if len(grid) > 1:
                grid[0] = grid[0].transpose(0, 1)
                grid[1] = grid[1].transpose(0, 1)
            return grid
