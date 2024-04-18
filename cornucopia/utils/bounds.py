"""Boundary conditions

There is no common convention to name boundary conditions.
This file lists all possible aliases and provides tool to "convert"
between them. It also defines function that can be used to implement
these boundary conditions.

=========   ===========   ===============================   =======================   =======================
NITorch     SciPy         PyTorch                           Other                     Description
=========   ===========   ===============================   =======================   =======================
replicate   nearest       nearest                           repeat, border            a  a | a b c d |  d  d
zero        constant(0)   zero                              zeros                     0  0 | a b c d |  0  0
dct2        reflect       reflection(align_corners=True)    neumann                   b  a | a b c d |  d  c
dct1        mirror        reflection(align_corners=False)                             c  b | a b c d |  c  b
dft         wrap                                            circular                  c  d | a b c d |  a  b
dst2                                                        antireflect, dirichlet   -b -a | a b c d | -d -c
dst1                                                        antimirror               -a  0 | a b c d |  0 -d
"""  # noqa: E501
import torch
from torch import Tensor
from typing import Tuple


doc_bounds = ('nearest', 'zero', 'reflect', 'mirror', 'wrap',
              'antireflect', 'antimirror')
pad_bounds = ('nearest', 'constant', 'reflect', 'mirror', 'wrap',
              'antireflect', 'antimirror')
doc_bounds_str = '{' + ', '.join(doc_bounds) + '}'
pad_bounds_str = '{' + ', '.join(pad_bounds) + '}'

nitorch_bounds = ('replicate', 'zero', 'dct2', 'dct1', 'dst2', 'dst1', 'dft')
scipy_bounds = ('nearest', 'constant', 'reflect', 'mirror', 'wrap')
pytorch_bounds = ('nearest', 'zero', 'reflection')
other_bounds = ('repeat', 'zeros', 'neumann', 'circular',
                'antireflect', 'dirichlet', 'antimirror')
all_bounds = (*nitorch_bounds, *scipy_bounds, *pytorch_bounds, *other_bounds)


def to_nitorch(bound):
    """Convert boundary type to NITorch's convention.

    Parameters
    ----------
    bound : [list of] str or bound_like
        Boundary condition in any convention
    as_enum : bool, default=False
        Return BoundType rather than str

    Returns
    -------
    bound : [list of] str or BoundType
        Boundary condition in NITorch's convention

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        b = b.lower() if isinstance(b, str) else b
        if b in ('replicate', 'repeat', 'border', 'nearest'):
            obound.append('replicate')
        elif b in ('zero', 'zeros', 'constant'):
            obound.append('zero')
        elif b in ('dct2', 'reflect', 'reflection', 'neumann'):
            obound.append('dct2')
        elif b in ('dct1', 'mirror'):
            obound.append('dct1')
        elif b in ('dft', 'wrap', 'circular'):
            obound.append('dft')
        elif b in ('dst2', 'antireflect', 'dirichlet'):
            obound.append('dst2')
        elif b in ('dst1', 'antimirror'):
            obound.append('dst1')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def to_scipy(bound):
    """Convert boundary type to SciPy's convention.

    Parameters
    ----------
    bound : [list of] str or bound_like
        Boundary condition in any convention

    Returns
    -------
    bound : [list of] str
        Boundary condition in SciPy's convention

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        b = b.lower()
        if b in ('replicate', 'border', 'nearest'):
            obound.append('border')
        elif b in ('zero', 'zeros', 'constant'):
            obound.append('constant')
        elif b in ('dct2', 'reflect', 'reflection', 'neumann'):
            obound.append('reflect')
        elif b in ('dct1', 'mirror'):
            obound.append('mirror')
        elif b in ('dft', 'wrap', 'circular'):
            obound.append('wrap')
        elif b in ('dst2', 'antireflect', 'dirichlet'):
            raise ValueError(f'Boundary condition {b} not available in SciPy.')
        elif b in ('dst1', 'antimirror'):
            raise ValueError(f'Boundary condition {b} not available in SciPy.')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def to_torch(bound):
    """Convert boundary type to PyTorch's convention.

    Parameters
    ----------
    bound : [list of] str or bound_like
        Boundary condition in any convention

    Returns
    -------
    [list of]
        bound : str
            Boundary condition in PyTorchs's convention
        align_corners : bool or None

    """
    intype = type(bound)
    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        b = b.lower()
        if b in ('replicate', 'border', 'nearest'):
            obound.append(('nearest', None))
        elif b in ('zero', 'zeros', 'constant'):
            obound.append(('zero', None))
        elif b in ('dct2', 'reflect', 'reflection', 'neumann'):
            obound.append(('reflection', True))
        elif b in ('dct1', 'mirror'):
            obound.append(('reflection', False))
        elif b in ('dft', 'wrap', 'circular'):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        elif b in ('dst2', 'antireflect', 'dirichlet'):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        elif b in ('dst1', 'antimirror'):
            raise ValueError(f'Boundary condition {b} not available in Torch.')
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    if issubclass(intype, (list, tuple)):
        obound = intype(obound)
    else:
        obound = obound[0]
    return obound


def dft(i, n):
    """Apply DFT (circulant/wrap) boundary conditions to an index

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : {1, 0, -1}
        Sign of the transformation (always 1 for dft)

    """
    return dft_script(i, n) if torch.is_tensor(i) else dft_int(i, n)


def dft_int(i, n):
    return i % n, 1


@torch.jit.script
def dft_script(i, n: int) -> Tuple[Tensor, int]:
    return i.remainder(n), 1


def replicate(i, n):
    """Apply replicate (nearest/border) boundary conditions to an index

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : {1, 0, -1}
        Sign of the transformation (always 1 for replicate)

    """
    fn = replicate_script if torch.is_tensor(i) else replicate_int
    return fn(i, n)


def replicate_int(i, n):
    return min(max(i, 0), n-1), 1


@torch.jit.script
def replicate_script(i, n: int) -> Tuple[Tensor, int]:
    return i.clamp(min=0, max=n-1), 1


def dct2(i, n):
    """Apply DCT-II (reflect) boundary conditions to an index

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : {1, 0, -1}
        Sign of the transformation (always 1 for dct2)

    """
    return dct2_script(i, n) if torch.is_tensor(i) else dct2_int(i, n)


def dct2_int(i: int, n: int) -> Tuple[int, int]:
    n2 = n * 2
    i = (n2 - 1) - i if i < 0 else i
    i = i % n2
    i = (n2 - 1) - i if i >= n else i
    return i, 1


@torch.jit.script
def dct2_script(i, n: int) -> Tuple[Tensor, int]:
    n2 = n * 2
    i = torch.where(i < 0, (n2 - 1) - i, i)
    i = i.remainder(n2)
    i = torch.where(i >= n, (n2 - 1) - i, i)
    return i, 1


def dct1(i, n):
    """Apply DCT-I (mirror) boundary conditions to an index

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : {1, 0, -1}
        Sign of the transformation (always 1 for dct1)

    """
    return dct1_script(i, n) if torch.is_tensor(i) else dct1_int(i, n)


def dct1_int(i: int, n: int) -> Tuple[int, int]:
    if n == 1:
        return 0, 1
    n2 = (n - 1) * 2
    i = abs(i) % n2
    i = n2 - i if i >= n else i
    return i, 1


@torch.jit.script
def dct1_script(i, n: int) -> Tuple[Tensor, int]:
    if n == 1:
        return torch.zeros_like(i), 1
    n2 = (n - 1) * 2
    i = i.abs().remainder(n2)
    i = torch.where(i >= n, n2 - i, i)
    return i, 1


def dst1(i, n):
    """Apply DST-I (antimirror) boundary conditions to an index

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, 0, -1}
        Sign of the transformation

    """
    return dst1_script(i, n) if torch.is_tensor(i) else dst1_int(i, n)


def dst1_int(i: int, n: int) -> Tuple[int, int]:
    n2 = 2 * (n + 1)

    # sign
    ii = (2*n - i) if i < 0 else i
    ii = (ii % n2) % (n + 1)
    x = 0 if ii == n else 1
    x = -x if (i / (n + 1)) % 2 >= 1 else x

    # index
    i = -i - 2 if i < 0 else i
    i = i % n2
    i = (n2 - 2) - i if i > n else i
    i = min(max(i, 0), n-1)
    return i, x


@torch.jit.script
def dst1_script(i, n: int) -> Tuple[Tensor, Tensor]:
    n2 = 2 * (n + 1)

    # sign
    #   zeros
    ii = torch.where(i < 0, 2*n - i, i).remainder(n2).remainder(n + 1)
    x = (ii != n).to(torch.int8)
    #   +/- ones
    x = torch.where((i / (n + 1)).remainder(2) >= 1, -x, x)

    # index
    i = torch.where(i < 0, -2 - i, i)
    i = i.remainder(n2)
    i = torch.where(i > n, (n2 - 2) - i, i)
    i = i.clamp(0, n-1)
    return i, x


def dst2(i, n):
    """Apply DST-II (antireflect) boundary conditions to an index

    Parameters
    ----------
    i : int or tensor
        Index
    n : int
        Length of the field of view

    Returns
    -------
    i : int or tensor
        Index that falls inside the field of view [0, n-1]
    s : [tensor of] {1, 0, -1}
        Sign of the transformation (always 1 for dct1)

    """
    return dst2_script(i, n) if torch.is_tensor(i) else dst2_int(i, n)


def dst2_int(i: int, n: int) -> Tuple[int, int]:
    x = -1 if (i/n) % 2 >= 1 else 1
    return dct2_int(i, n)[0], x


@torch.jit.script
def dst2_script(i, n: int) -> Tuple[Tensor, Tensor]:
    x = torch.ones([1], dtype=torch.int8, device=i.device)
    x = torch.where((i / n).remainder(2) >= 1, -x, x)
    return dct2_script(i, n)[0], x


nearest = border = replicate
reflect = neumann = dct2
mirror = dct1
antireflect = dirichlet = dst2
antimirror = dst1
wrap = circular = dft
