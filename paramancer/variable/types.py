from __future__ import annotations

from torch import Tensor, nn

from typing import TYPE_CHECKING
from typing import Literal, TypeAlias, ParamSpec, Concatenate
from collections.abc import Callable

if TYPE_CHECKING:
    from .inner import InnerVar
    from .parameter import ParameterBundle
    from ..optim.step import OptimizerStep
    from ..optim.optimizer import Optimizer

# Scalar
ScalarLike: TypeAlias = float | Tensor

# Types allowed for AlgoVar and OptVar construction
TensorVar: TypeAlias = Tensor
TupleVar: TypeAlias = tuple[TensorVar, ...]
NestedVar: TypeAlias = tuple[TupleVar, TupleVar]
FlatInnerVarType: TypeAlias = TensorVar | TupleVar
InnerVarType: TypeAlias = FlatInnerVarType | NestedVar
OptVarType: TypeAlias = FlatInnerVarType
AlgoVarType: TypeAlias = InnerVarType
BaseVariableType: TypeAlias = FlatInnerVarType # to be removed
VariableType: TypeAlias = InnerVarType # to be removed

FlatInnerVarLike: TypeAlias = "FlatInnerVarType | InnerVar"
InnerVarLike: TypeAlias = "InnerVarType | InnerVar"
OptVarLike: TypeAlias = FlatInnerVarLike
AlgoVarLike: TypeAlias = InnerVarLike
BaseVariableLike: TypeAlias = FlatInnerVarLike # to be removed
VariableLike: TypeAlias = InnerVarLike # to be removed


# Aliases for callables with various args and returns types
OptVarToTen: TypeAlias = Callable[[OptVarType], Tensor]
OptVarToOptVar: TypeAlias = Callable[[OptVarType], OptVarType]
VoidToScal: TypeAlias = Callable[[], ScalarLike]
OptVarToScal: TypeAlias = Callable[[OptVarType], ScalarLike]
OptVarXOptVarToScal: TypeAlias = Callable[[OptVarType, OptVarType], ScalarLike]

# Common function types used across paramancer
SmoothObjType: TypeAlias = OptVarToTen
GradMapType: TypeAlias = OptVarToOptVar
ProxMapType: TypeAlias = OptVarToOptVar
LinOpType: TypeAlias = OptVarToOptVar
MetricFnType: TypeAlias = OptVarToScal

MomentumSchedType: TypeAlias = VoidToScal
LineSearchSchedType: TypeAlias = OptVarXOptVarToScal

MetricSpec: TypeAlias = Literal["default"] | MetricFnType
StepsizeSchedType: TypeAlias = MomentumSchedType | LineSearchSchedType


# for flattened variable to be passed to `OptimizerID.apply`
FlattendType: TypeAlias = tuple[Tensor, ...]
TensorSpec: TypeAlias = tuple[Literal["tensor"]]
TupleSpec: TypeAlias = tuple[Literal["tuple"], int]
NestedSpec: TypeAlias = tuple[Literal["nested"], int, int]
VSpecType: TypeAlias = TensorSpec | TupleSpec | NestedSpec
PSpecType: TypeAlias = TensorSpec | TupleSpec


# Types allowed for Parameter
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

# Aliases for callables with various args and returns types ----
BVarXPrmXAnyToTen: TypeAlias = Callable[
    Concatenate[OptVarLike, ParameterType, P], Tensor
]
BVarXPrmXAnyToBVar: TypeAlias = Callable[
    Concatenate[OptVarLike, ParameterType, P], OptVarLike
]
BVarXAnyToTen: TypeAlias = Callable[
    Concatenate[OptVarLike, P], Tensor
]
BVarXAnyToBVar: TypeAlias = Callable[
    Concatenate[OptVarLike, P], OptVarLike
]

# Aliases for various parametric maps
ParamSmoothObjType: TypeAlias = BVarXPrmXAnyToTen
ParamGradMapType: TypeAlias = BVarXPrmXAnyToBVar
ParamProxMapType: TypeAlias = BVarXPrmXAnyToBVar
# vvvvv Generic aliases (Param* aliases are stricter specializations of these)
PSmoothObjType: TypeAlias = BVarXAnyToTen
PGradMapType: TypeAlias = BVarXAnyToBVar
PLinOpType: TypeAlias = BVarXAnyToBVar


# Decorator type for ensuring Variable input to the `Optimizer` and 
# `OptimizerStep` methods
Owner: TypeAlias = "OptimizerStep | Optimizer"
WrapperIn: TypeAlias = "Callable[Concatenate[Owner, InnerVar, P], InnerVar]"
WrapperOut: TypeAlias = Callable[Concatenate[Owner, InnerVarLike, P], InnerVarLike]
