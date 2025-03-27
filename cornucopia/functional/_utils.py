
# stdlib
from typing import Union, Mapping, Sequence, TypeVar

# external
import torch


T = TypeVar('T')
Tensor = torch.Tensor
Value = Union[float, Tensor]
Output = Union[Tensor, Mapping[str, Tensor], Sequence[Tensor]]
OneOrMore = Union[T, Sequence[T]]


def _unsqz_spatial(x: Value, ndim: int) -> Value:
    if torch.is_tensor(x):
        x = x[(Ellipsis,) + (None,) * ndim]
    return x


def _backend(
    *tensors_or_dtypes_or_devices, dtype=None, device=None, **kwargs
):
    if dtype and device:
        return
    for tensor_or_dtype_or_device in tensors_or_dtypes_or_devices:
        if torch.is_tensor(tensor_or_dtype_or_device):
            dtype = dtype or tensor_or_dtype_or_device.dtype
            device = device or tensor_or_dtype_or_device.device
        elif isinstance(tensor_or_dtype_or_device, torch.device):
            dtype = dtype or tensor_or_dtype_or_device
        elif isinstance(tensor_or_dtype_or_device, torch.device):
            device = device or tensor_or_dtype_or_device
        elif isinstance(tensor_or_dtype_or_device, str):
            device = device or torch.device(tensor_or_dtype_or_device)
        if dtype and device:
            return
    return dict(dtype=dtype, device=device)


def _backend_float(
    *tensors_or_dtypes_or_devices, dtype=None, device=None, **kwargs
):
    if dtype and device:
        return
    for tensor_or_dtype_or_device in tensors_or_dtypes_or_devices:
        if torch.is_tensor(tensor_or_dtype_or_device):
            if tensor_or_dtype_or_device.dtype.is_floating_point:
                dtype = dtype or tensor_or_dtype_or_device.dtype
            device = device or tensor_or_dtype_or_device.device
        elif isinstance(tensor_or_dtype_or_device, torch.device):
            if tensor_or_dtype_or_device.is_floating_point:
                dtype = dtype or tensor_or_dtype_or_device
        elif isinstance(tensor_or_dtype_or_device, torch.device):
            device = device or tensor_or_dtype_or_device
        elif isinstance(tensor_or_dtype_or_device, str):
            device = device or torch.device(tensor_or_dtype_or_device)
        if dtype and device:
            return
    if dtype is None or not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    return dict(dtype=dtype, device=device)


def _affine2axes(affine):
    """
    Compute mappings between voxel (ijk) and anatomical (RAS) axes

    Parameters
    ----------
    affine : (D, D) array, optional
        Affine matrix (linear part only)

    Returns
    -------
    vox2anat : (D,) list[{"LR", "RL", "AP", "PA", "IS", "SI}]
        Anatomical axis and polarity of each voxel axis
    anat2vox : dict[str, tuple[int, str]]
        Voxel axis and polarity of each anatomical axis.
        Keys are in `{"LR", "RL", "AP", "PA", "IS", "SI"}`. Values are in
        `{(0, "+"), (0, "-"), (1, "+"), (1, "-"), (2, "+"), (2, "-")}`
    """
    if affine is None:
        # Assume RAS
        return (
            ["LR", "PA", "IS"],
            {"LR": (0, "+"), "RL": (0, "-"),
             "PA": (1, "+"), "AP": (1, "-"),
             "IS": (2, "+"), "SI": (2, "-")}
        )

    affine = torch.as_tensor(affine)
    ndim = len(affine)

    voxel_size = (affine**2).sum(0)**0.5
    affine = affine / voxel_size

    # add noise to avoid issues if there's a 45 deg angle somewhere
    affine = affine + (torch.rand([ndim, ndim]).to(affine) - 0.5) * 1e-5

    # project onto canonical axes
    onehot = affine.square().round().int()
    index = [onehot[:, i].tolist().index(1) for i in range(ndim)]
    sign = [
        -1 if affine[index[i], i] < 0 else 1
        for i in range(ndim)
    ]
    anatnames = ['LR', 'PA', 'IS'][:ndim]
    voxnames = list(range(ndim))

    vox2anat = [
        anatnames[index[i]][::-1] if sign[i] else index[i]
        for i in range(ndim)
    ]
    anat2vox = {}
    if 'LR' in vox2anat:
        anat2vox['LR'] = (voxnames[vox2anat.index('LR')], '+')
        anat2vox['RL'] = (voxnames[vox2anat.index('LR')], '-')
    else:
        anat2vox['RL'] = (voxnames[vox2anat.index('RL')], '+')
        anat2vox['LR'] = (voxnames[vox2anat.index('RL')], '-')
    if 'PA' in vox2anat:
        anat2vox['PA'] = (voxnames[vox2anat.index('PA')], '+')
        anat2vox['AP'] = (voxnames[vox2anat.index('PA')], '-')
    else:
        anat2vox['AP'] = (voxnames[vox2anat.index('AP')], '+')
        anat2vox['PA'] = (voxnames[vox2anat.index('AP')], '-')
    if 'IS' in vox2anat:
        anat2vox['IS'] = (voxnames[vox2anat.index('IS')], '+')
        anat2vox['SI'] = (voxnames[vox2anat.index('IS')], '-')
    else:
        anat2vox['SI'] = (voxnames[vox2anat.index('SI')], '+')
        anat2vox['IS'] = (voxnames[vox2anat.index('SI')], '-')

    return vox2anat, anat2vox


def _affine2layout(affine) -> str:
    vox2anat, _ = _affine2axes(affine)
    return "".join(name[-1:] for name in vox2anat)


def _axis_name2index(axes, layout):
    if not isinstance(layout, (str, list)):
        layout = _affine2layout(layout)
    if isinstance(layout, str):
        layout = [
            {"L": "R", "P": "A", "I": "S"}.get(ax, ax)
            for ax in layout.upper()
        ]
    if isinstance(axes, int):
        return axes
    if isinstance(axes, str):
        axes = axes[0].upper()
        axes = {"L": "R", "P": "A", "I": "S"}.get(axes, axes)
        return layout.index(axes)
    if isinstance(axes, (list, tuple)):
        return type(axes)(
            _axis_name2index(ax, layout)
            for ax in axes
        )
    if isinstance(axes, dict):
        return type(axes)({
            k: _axis_name2index(ax, layout)
            for k, ax in axes.items()
        })
    return axes
