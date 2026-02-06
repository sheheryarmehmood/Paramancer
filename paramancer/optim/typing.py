from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeAlias, Tuple, Any
from torch import Tensor

if TYPE_CHECKING:
    from .variable import Variable



TensorLike: TypeAlias = Tensor


# Types allowed for Variable construction
FlatVariable: TypeAlias = TensorLike
TupleVariable: TypeAlias = Tuple[FlatVariable, ...]
NestedVariable: TypeAlias = Tuple[TupleVariable, TupleVariable]
VariableType: TypeAlias = FlatVariable | TupleVariable | NestedVariable

VariableLike: TypeAlias = "VariableType | Variable"
ScalarLike: TypeAlias = float | TensorLike


# ---- Aliases for callables with various args and returns types ----
VarToVar: TypeAlias = Callable[[VariableType], VariableType]
VoidToScal: TypeAlias = Callable[[], ScalarLike]
VarToScal: TypeAlias = Callable[[VariableType], ScalarLike]
VarXVarToScal: TypeAlias = Callable[[VariableType, VariableType], ScalarLike]

# ---- Common function types used across paramancer ----
GradMapType: TypeAlias = VarToVar
ProxMapType: TypeAlias = VarToVar
LinOpType: TypeAlias = VarToVar
MetricFnType: TypeAlias = None | VarToScal

MomentumSchedType: TypeAlias = None | VoidToScal
LineSearchSchedType: TypeAlias = None | VarXVarToScal

MetricSpec: TypeAlias = Literal["default"] | MetricFnType
StepsizeSchedTypes: TypeAlias = MomentumSchedType | LineSearchSchedType


# for flattened variable to be passed to `OptimizerID.apply`
ApplyType: TypeAlias = Tuple[TensorLike, ...]
SpecType: TypeAlias = Tuple[Any, ...]
