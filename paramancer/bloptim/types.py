from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ParamSpec, TypeAlias, Concatenate
from torch import Tensor, nn

from ..optim.types import BaseVariableLike


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
BVarXAnyToBVar: TypeAlias = Callable[
    Concatenate[BaseVariableLike, P], BaseVariableLike
]

# Aliases for various parametric maps
ParamObjType: TypeAlias = BVarXPrmXAnyToTen
ParamGradMapType: TypeAlias = BVarXPrmXAnyToBVar
ParamProxMapType: TypeAlias = BVarXPrmXAnyToBVar
ParamLinOpType: TypeAlias = BVarXAnyToBVar

# Note: Perhaps we can split `ParamLinOpType` into `ParamLinOpType` and 
# `ParamAdjLinOpType` by using `TypeVar` for Primal and Dual Space. 