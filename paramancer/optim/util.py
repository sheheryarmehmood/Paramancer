from __future__ import annotations
from dataclasses import dataclass
from functools import wraps
import torch

from ..variable import Variable
from ..variable.types import (
    VariableLike, ScalarLike,
    WrapperIn, WrapperOut, Owner, P
)


@dataclass
class OptimizationResult:
    solution: VariableLike
    iterations: int
    metric: float | None
    converged: bool

def to_float_scalar(x: ScalarLike) -> float:
    """Convert a scalar-like (float or 0-dim tensor) into a Python float."""
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError("Expected a scalar tensor for metric value.")
        return float(x.detach().item())
    return float(x)


def ensure_var_input(fn: WrapperIn) -> WrapperOut:
    @wraps(fn)
    def wrapper(
        self: Owner,
        x_in: VariableLike,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> VariableLike:
        _input_is_variable = isinstance(x_in, Variable)
        if hasattr(self, "_input_is_variable"):
            self._input_is_variable = _input_is_variable
        x_var = x_in if _input_is_variable else Variable(x_in)
        x_out = fn(self, x_var, *args, **kwargs)
        return x_out if _input_is_variable else x_out.data
    return wrapper