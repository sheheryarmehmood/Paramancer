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
BVarXPrmToTen: TypeAlias = Callable[
    Concatenate[BaseVariableLike, ParameterType, P], Tensor
]
BVarXPrmToBVar: TypeAlias = Callable[
    Concatenate[BaseVariableLike, ParameterType, P], BaseVariableLike
]


# Aliases for various parametric maps
ParamObjType: TypeAlias = BVarXPrmToTen
ParamGradMapType: TypeAlias = BVarXPrmToBVar
ParamProxMapType: TypeAlias = BVarXPrmToBVar
ParamLinOpType: TypeAlias = BVarXPrmToBVar

