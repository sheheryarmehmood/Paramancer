from __future__ import annotations

from dataclasses import dataclass
from functools import wraps

import torch

from ..variable.types import (
    AlgoVarLike,
    FlatVarLike,
    FlatWrapperIn,
    FlatWrapperOut,
    Owner,
    P,
    PairVarLike,
    PairWrapperIn,
    PairWrapperOut,
    ScalarLike,
)
from ..variable.util import as_flat_var, as_pair_var, is_flat_var, is_pair_var


@dataclass
class OptimizationResult:
    solution: AlgoVarLike
    iterations: int
    metric: float | None
    converged: bool


def to_float_scalar(x: ScalarLike) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError("Expected a scalar tensor for metric value.")
        return float(x.detach().item())
    return float(x)


def ensure_flat_input(fn: FlatWrapperIn) -> FlatWrapperOut:
    @wraps(fn)
    def wrapper(
        self: Owner,
        x_in: FlatVarLike,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> FlatVarLike:
        input_is_wrapper = is_flat_var(x_in)
        if hasattr(self, "_input_is_wrapper"):
            self._input_is_wrapper = input_is_wrapper
        x_var = as_flat_var(x_in)
        x_out = fn(self, x_var, *args, **kwargs)
        return x_out if input_is_wrapper else x_out.data

    return wrapper


def ensure_pair_input(fn: PairWrapperIn) -> PairWrapperOut:
    @wraps(fn)
    def wrapper(
        self: Owner,
        x_in: PairVarLike,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> PairVarLike:
        input_is_wrapper = is_pair_var(x_in)
        if hasattr(self, "_input_is_wrapper"):
            self._input_is_wrapper = input_is_wrapper
        x_var = as_pair_var(x_in)
        x_out = fn(self, x_var, *args, **kwargs)
        return x_out if input_is_wrapper else x_out.data

    return wrapper
