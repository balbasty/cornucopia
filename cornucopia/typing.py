# stdlib
from ast import Import
from numbers import Number

# dependencies
import typing_extensions as tx
from torch import Tensor, device

# internals
from .random import Sampler

# optionals
if tx.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
else:
    try:
        import numpy as np
    except ImportError:
        np = None
    try:
        import numpy.typing as npt
    except ImportError:
        npt = None


# --- TypeVars ---
BuiltinScalar = tx.Union[bool, int, float, complex]
BuiltinNumber = tx.Union[int, float, complex]
TypeT = tx.TypeVar('T', default=tx.Any)
BuiltinScalarT = tx.TypeVar('BuiltinScalarT', bound=BuiltinScalar, default=float)
BuiltinNumberT = tx.TypeVar('BuiltinNumberT', bound=BuiltinNumber, default=float)

if tx.TYPE_CHECKING or np:
    ScalarT = tx.TypeVar('ScalarT', bound=tx.Union[BuiltinScalar, np.generic], default=float)
    NumberT = tx.TypeVar('NumberT', bound=tx.Union[BuiltinNumber, np.number], default=float)
else:
    ScalarT = BuiltinScalar
    NumberT = BuiltinNumber

TypeT_co = tx.TypeVar('TypeT_co', covariant=True, default=tx.Any)
BuiltinScalarT_co = tx.TypeVar('BuiltinScalarT_co', bound=BuiltinScalar, covariant=True, default=float)
BuiltinNumberT_co = tx.TypeVar('BuiltinNumberT_co', bound=BuiltinNumber, covariant=True, default=float)

if tx.TYPE_CHECKING or np:
    ScalarT_co = tx.TypeVar('ScalarT_co', bound=tx.Union[BuiltinScalar, np.generic], covariant=True, default=float)
    NumberT_co = tx.TypeVar('NumberT_co', bound=tx.Union[BuiltinNumber, np.number], covariant=True, default=float)
else:
    ScalarT_co = BuiltinScalar
    NumberT_co = BuiltinNumber

# --- Simple values ---
ItemOrSequence = tx.Union[TypeT_co, tx.Sequence[TypeT_co]]
ScalarOrSequence = tx.Union[ScalarT_co, tx.Sequence[ScalarT_co]]
NumberOrSequence = ScalarOrSequence[NumberT_co]

# --- Tensor shapes ---
Shape = tuple[int, ...]
AnyShape = tuple[tx.Any, ...]
ShapeLike = ItemOrSequence[tx.SupportsIndex]

ShapeT = tx.TypeVar('ShapeT', bound=Shape, default=Shape)
ShapeT_co = tx.TypeVar('ShapeT_co', bound=Shape, default=Shape, covariant=True)

# --- Annotated arrays ---
Integral = tx.TypeVar('Integral', bound=int, default=int, covariant=True)
if tx.TYPE_CHECKING or npt:
    ArrayLikeT = tx.TypeVar('ArrayLikeT', bound=npt.ArrayLike, default=npt.ArrayLike)
else:
    class ArrayProtocol(tx.Protocol):
        def __array__(self, dtype=None, copy=None): ...

    ArrayLikeT = tx.Union[NumberOrSequence, ArrayProtocol]

# --- Tensor-like ---
TensorLike = tx.Union[ArrayLikeT, Tensor]
NumberOrTensor = tx.Union[NumberT_co, Tensor]
VectorLike = tx.Union[ScalarT_co, tx.Sequence[ScalarT_co], Tensor]

# --- Common transform parameters ---
IncludeT = ItemOrSequence[str]
ExcludeT = ItemOrSequence[str]
ConsumeT = ItemOrSequence[str]
SharedStrT = tx.Literal["channels+tensors", "channels", "tensors", ""]
SharedT = tx.Union[SharedStrT, bool]
ReturnsStrT = tx.Union[tx.Literal['input', 'output'], str]
ReturnsT = tx.Union[
    ReturnsStrT,
    tx.Sequence[ReturnsStrT],
    tx.Mapping[str, ReturnsStrT]
]
AppendT = tx.Union[bool, str]
PrefixT = tx.Union[bool, str]

# --- Samplers ---
SamplerOrBound = tx.Union[Sampler, tx.Tuple[NumberT, NumberT], NumberT]
SamplerOrBoundOrBool = tx.Union[SamplerOrBound, bool]

# --- Other ---
TorchDevice = tx.Union[device, str]
TorchBound = tx.Literal['zeros', 'border', 'reflection', 'circular']
