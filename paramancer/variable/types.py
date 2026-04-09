from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Concatenate, Literal, ParamSpec, TypeAlias

from torch import Tensor, nn

if TYPE_CHECKING:
    from .flat import FlatVar
    from .pair import PairVar
    from .parameter import ParameterBundle
    from ..optim.optimizer import Optimizer
    from ..optim.step import OptimizerStep

# Scalars
ScalarLike: TypeAlias = float | Tensor

# Raw optimization-state payloads
TensorVarType: TypeAlias = Tensor
TupleVarType: TypeAlias = tuple[TensorVarType, ...]
FlatRawVarType: TypeAlias = TensorVarType | TupleVarType
PairRawVarType: TypeAlias = tuple[FlatRawVarType, FlatRawVarType]

# Wrapper-aware aliases
FlatVarLike: TypeAlias = "FlatRawVarType | FlatVar"
PairVarLike: TypeAlias = "PairRawVarType | PairVar"
AlgoVar: TypeAlias = "FlatVar | PairVar"
AlgoVarLike: TypeAlias = "FlatVarLike | PairVarLike"

OptVarType: TypeAlias = FlatRawVarType
AlgoVarType: TypeAlias = FlatRawVarType | PairRawVarType
OptVarLike: TypeAlias = FlatVarLike

# Common callable aliases for flat optimization variables
FlatRawVarToTen: TypeAlias = Callable[[FlatRawVarType], Tensor]
FlatRawVarToFlatRawVar: TypeAlias = Callable[[FlatRawVarType], FlatRawVarType]
PairRawVarToPairRawVar: TypeAlias = Callable[[PairRawVarType], PairRawVarType]
VoidToScal: TypeAlias = Callable[[], ScalarLike]
FlatRawVarToScal: TypeAlias = Callable[[FlatRawVarType], ScalarLike]
FlatRawVarXFlatRawVarToScal: TypeAlias = Callable[
    [FlatRawVarType, FlatRawVarType], ScalarLike
]

SmoothObjType: TypeAlias = FlatRawVarToTen
GradMapType: TypeAlias = FlatRawVarToFlatRawVar
ProxMapType: TypeAlias = FlatRawVarToFlatRawVar
FlatLinOpType: TypeAlias = FlatRawVarToFlatRawVar
PairLinOpType: TypeAlias = PairRawVarToPairRawVar
LinOpType: TypeAlias = FlatLinOpType | PairLinOpType
MetricFnType: TypeAlias = FlatRawVarToScal

MomentumSchedType: TypeAlias = VoidToScal
LineSearchSchedType: TypeAlias = FlatRawVarXFlatRawVarToScal

MetricSpec: TypeAlias = Literal["default"] | MetricFnType
StepsizeSchedType: TypeAlias = MomentumSchedType | LineSearchSchedType

# Flattened raw variable payloads
FlattendType: TypeAlias = tuple[Tensor, ...]
TensorSpec: TypeAlias = tuple[Literal["tensor"]]
TupleSpec: TypeAlias = tuple[Literal["tuple"], int]
PairSpec: TypeAlias = tuple[Literal["pair"], int, int]
VSpecType: TypeAlias = TensorSpec | TupleSpec | PairSpec
PSpecType: TypeAlias = TensorSpec | TupleSpec

# Parameters
# The naming of parameters need to be worked out. Perhaps we can use `TensorParType`, `TupleParType`, `ParListType`, `RawParType` for the raw types. We can rename the wrapper `ParameterBundle` to `AlgoPar` or something else to be consistent with `AlgoVar`.
FlatParameter = Tensor
TupleParameter = tuple[Tensor, ...]
ParameterList = nn.ParameterList
ParameterType = FlatParameter | TupleParameter | ParameterList


def is_parameter_type(param: object) -> bool:
    return isinstance(param, (FlatParameter, ParameterList)) or (
        isinstance(param, tuple)
        and all(isinstance(item, FlatParameter) for item in param)
    )


IndexType: TypeAlias = Literal["all"] | int | tuple[int, ...]
IndexMapType: TypeAlias = dict[str, IndexType]

ParameterLike: TypeAlias = "ParameterType | ParameterBundle"

P = ParamSpec("P")

# Parametric callable aliases
FlatVarXPrmXAnyToTen: TypeAlias = Callable[
    Concatenate[FlatVarLike, ParameterType, P], Tensor
]
FlatVarXPrmXAnyToFlatVar: TypeAlias = Callable[
    Concatenate[FlatVarLike, ParameterType, P], FlatVarLike
]
FlatVarXAnyToTen: TypeAlias = Callable[Concatenate[FlatVarLike, P], Tensor]
FlatVarXAnyToFlatVar: TypeAlias = Callable[
    Concatenate[FlatVarLike, P], FlatVarLike
]

ParamSmoothObjType: TypeAlias = FlatVarXPrmXAnyToTen
ParamGradMapType: TypeAlias = FlatVarXPrmXAnyToFlatVar
ParamProxMapType: TypeAlias = FlatVarXPrmXAnyToFlatVar
PSmoothObjType: TypeAlias = FlatVarXAnyToTen
PGradMapType: TypeAlias = FlatVarXAnyToFlatVar
PLinOpType: TypeAlias = FlatVarXAnyToFlatVar

# Decorator types for optimizer/step wrapper normalization
Owner: TypeAlias = "OptimizerStep | Optimizer"
FlatWrapperIn: TypeAlias = "Callable[Concatenate[Owner, FlatVar, P], FlatVar]"
FlatWrapperOut: TypeAlias = Callable[
    Concatenate[Owner, FlatVarLike, P], FlatVarLike
]
PairWrapperIn: TypeAlias = "Callable[Concatenate[Owner, PairVar, P], PairVar]"
PairWrapperOut: TypeAlias = Callable[
    Concatenate[Owner, PairVarLike, P], PairVarLike
]

WrapperIn: TypeAlias = "Callable[Concatenate[Owner, AlgoVar, P], AlgoVar]"
WrapperOut: TypeAlias = Callable[Concatenate[Owner, AlgoVarLike, P], AlgoVarLike]
