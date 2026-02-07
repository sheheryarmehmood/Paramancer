from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeAlias, Tuple, Any
from torch import Tensor

if TYPE_CHECKING:
    from .variable import Variable



# Types allowed for Variable construction
FlatVariable: TypeAlias = Tensor
TupleVariable: TypeAlias = Tuple[FlatVariable, ...]
NestedVariable: TypeAlias = Tuple[TupleVariable, TupleVariable]
VariableType: TypeAlias = FlatVariable | TupleVariable | NestedVariable

VariableLike: TypeAlias = "VariableType | Variable"
ScalarLike: TypeAlias = float | Tensor


# ---- Aliases for callables with various args and returns types ----
VarToVar: TypeAlias = Callable[[VariableType], VariableType]
VoidToScal: TypeAlias = Callable[[], ScalarLike]
VarToScal: TypeAlias = Callable[[VariableType], ScalarLike]
VarXVarToScal: TypeAlias = Callable[[VariableType, VariableType], ScalarLike]

# ---- Common function types used across paramancer ----
GradMapType: TypeAlias = VarToVar
ProxMapType: TypeAlias = VarToVar
LinOpType: TypeAlias = VarToVar
MetricFnType: TypeAlias = VarToScal

MomentumSchedType: TypeAlias = VoidToScal
LineSearchSchedType: TypeAlias = VarXVarToScal

MetricSpec: TypeAlias = Literal["default"] | MetricFnType
StepsizeSchedTypes: TypeAlias = MomentumSchedType | LineSearchSchedType


# for flattened variable to be passed to `OptimizerID.apply`
ApplyType: TypeAlias = Tuple[Tensor, ...]
SpecType: TypeAlias = Tuple[Any, ...]
