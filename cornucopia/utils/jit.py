# stdlib
import os
from typing import List

# dependencies
import torch
from torch import Tensor

# internals
from .version import torch_version


IS_JITSCRIPT_ACTIVATED = int(os.environ.get('PYTORCH_JIT', '1'))
IS_JITSCRIPT_DEPRECATED = torch_version('>=', (2, 10))
HAS_FLOOR_DIVIDE = torch_version('>=', (1, 6))

if IS_JITSCRIPT_DEPRECATED:
    # `torch.jit.script` is deprecated, but there is no one-to-one
    # alternative. The new `torch.compile` does not work with nested
    # decorated functions. Only the top-level function must be compiled.
    # I therefore remove the `@torch.jit.script` decorator entirely,
    # and hope that users will use `torch.compile` on their code
    # (although it is unlikely).
    def jitscript(fn):
        return fn
else:
    jitscript = torch.jit.script


@jitscript
def list_reverse_int(x: List[int]) -> List[int]:
    if len(x) == 0:
        return x
    return [x[i] for i in range(-1, -len(x)-1, -1)]


@jitscript
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


@jitscript
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

    if IS_JITSCRIPT_DEPRECATED or not IS_JITSCRIPT_ACTIVATED:
        def meshgrid_list_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(*x, indexing='ij')

        def meshgrid_list_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(*x, indexing='xy')

    else:
        @jitscript
        def meshgrid_list_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x, indexing='ij')

        @jitscript
        def meshgrid_list_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x, indexing='xy')

else:
    if not IS_JITSCRIPT_ACTIVATED:
        def meshgrid_list_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x)

        def meshgrid_list_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            grid = torch.meshgrid(*x)
            if len(grid) > 1:
                grid[0] = grid[0].transpose(0, 1)
                grid[1] = grid[1].transpose(0, 1)
            return grid

    else:
        @jitscript
        def meshgrid_list_ij(x: List[torch.Tensor]) -> List[torch.Tensor]:
            return torch.meshgrid(x)

        @jitscript
        def meshgrid_list_xy(x: List[torch.Tensor]) -> List[torch.Tensor]:
            grid = torch.meshgrid(x)
            if len(grid) > 1:
                grid[0] = grid[0].transpose(0, 1)
                grid[1] = grid[1].transpose(0, 1)
            return grid
