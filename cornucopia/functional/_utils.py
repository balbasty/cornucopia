
# stdlib
from typing import Union, Mapping, Sequence

# external
import torch


Tensor = torch.Tensor
Value = Union[float, Tensor]
Output = Union[Tensor, Mapping[Tensor], Sequence[Tensor]]


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
