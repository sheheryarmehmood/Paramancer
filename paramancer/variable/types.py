from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeAlias, ParamSpec, Concatenate
from torch import Tensor, nn

if TYPE_CHECKING:
    from .variable import Variable


# Types allowed for Variable construction
FlatVariable: TypeAlias = Tensor
TupleVariable: TypeAlias = tuple[FlatVariable, ...]
NestedVariable: TypeAlias = tuple[TupleVariable, TupleVariable]
BaseVariableType: TypeAlias = FlatVariable | TupleVariable
VariableType: TypeAlias = BaseVariableType | NestedVariable

BaseVariableLike: TypeAlias = "BaseVariableType | Variable"
VariableLike: TypeAlias = "VariableType | Variable"
ScalarLike: TypeAlias = float | Tensor


# Aliases for callables with various args and returns types
BVarToTen: TypeAlias = Callable[[BaseVariableType], Tensor]
BVarToBVar: TypeAlias = Callable[[BaseVariableType], BaseVariableType]
VoidToScal: TypeAlias = Callable[[], ScalarLike]
VarToScal: TypeAlias = Callable[[VariableType], ScalarLike]
BVarXBVarToScal: TypeAlias = Callable[[VariableType, VariableType], ScalarLike]

# Common function types used across paramancer
SmoothObjType: TypeAlias = BVarToTen
GradMapType: TypeAlias = BVarToBVar
ProxMapType: TypeAlias = BVarToBVar
LinOpType: TypeAlias = BVarToBVar
MetricFnType: TypeAlias = VarToScal

MomentumSchedType: TypeAlias = VoidToScal
LineSearchSchedType: TypeAlias = BVarXBVarToScal

MetricSpec: TypeAlias = Literal["default"] | MetricFnType
StepsizeSchedType: TypeAlias = MomentumSchedType | LineSearchSchedType


# for flattened variable to be passed to `OptimizerID.apply`
ApplyType: TypeAlias = tuple[Tensor, ...]
FlatSpec: TypeAlias = tuple[Literal["flat"]]
TupleSpec: TypeAlias = tuple[Literal["tuple"], int]
NestedSpec: TypeAlias = tuple[Literal["nested"], int, int]
SpecType: TypeAlias = FlatSpec | TupleSpec | NestedSpec


# Types allowed for Parameter
FlatParameter = nn.Parameter
TupleParameter = nn.ParameterList
ParameterType = FlatParameter | TupleParameter

P = ParamSpec("P")

# Aliases for callables with various args and returns types ----
BVarXPrmXAnyToTen: TypeAlias = Callable[
    Concatenate[BaseVariableLike, ParameterType, P], Tensor
]
BVarXPrmXAnyToBVar: TypeAlias = Callable[
    Concatenate[BaseVariableLike, ParameterType, P], BaseVariableLike
]
BVarXAnyToTen: TypeAlias = Callable[
    Concatenate[BaseVariableLike, P], Tensor
]
BVarXAnyToBVar: TypeAlias = Callable[
    Concatenate[BaseVariableLike, P], BaseVariableLike
]

# Aliases for various parametric maps
ParamSmoothObjType: TypeAlias = BVarXPrmXAnyToTen
ParamGradMapType: TypeAlias = BVarXPrmXAnyToBVar
ParamProxMapType: TypeAlias = BVarXPrmXAnyToBVar
# vvvvv Generic aliases (Param* aliases are stricter specializations of these)
PSmoothObjType: TypeAlias = BVarXAnyToTen
PGradMapType: TypeAlias = BVarXAnyToBVar
PLinOpType: TypeAlias = BVarXAnyToBVar
