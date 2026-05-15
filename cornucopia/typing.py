# stdlib
from numbers import Number
import numpy.typing as npt

# dependencies
import typing_extensions as tx
from torch import Tensor, device

# internals
from .random import Sampler

T = tx.TypeVar('T')

# --- Simple values ---
ScalarOrSequence = tx.Union[T, tx.Sequence[T]]
NumberOrTensor = tx.Union[Number, Tensor]
VectorLike = tx.Union[tx.Sequence[Number], Tensor]
ScalarOrVectorLike = tx.Union[Number, VectorLike]

# --- Common transform parameters ---
IncludeKeyType = ScalarOrSequence[str]
ExcludeKeyType = ScalarOrSequence[str]
SharedStrType = tx.Literal["channels+tensors", "channels", "tensors", ""]
SharedType = tx.Union[SharedStrType, bool]
ReturnsStrType = tx.Union[tx.Literal['input', 'output'], str]
ReturnsType = tx.Union[
    ReturnsStrType,
    tx.Sequence[ReturnsStrType],
    tx.Mapping[str, ReturnsStrType]
]

# --- Samplers ---

SamplerOrBound = tx.Union[Sampler, T]
SamplerOrBoundOrBool = tx.Union[SamplerOrBound, bool]

# --- Other ---
TorchDevice = tx.Union[device, str]
TorchBound = tx.Literal['zeros', 'border', 'reflection', 'circular']
