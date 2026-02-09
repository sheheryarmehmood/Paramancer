from __future__ import annotations
from dataclasses import dataclass
import torch
from typing import Any

from .types import VariableLike, ScalarLike, VariableType, ApplyType, SpecType


@dataclass
class OptimizationResult:
    solution: VariableLike
    iterations: int
    metric: float | None
    converged: bool


def is_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)

def is_tuple_of_tensors(x: Any) -> bool:
    return isinstance(x, tuple) and all(is_tensor(t) for t in x)

def is_nested_variable(x: Any) -> bool:
    # exactly: ( (tensors...), (tensors...) )
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and is_tuple_of_tensors(x[0])
        and is_tuple_of_tensors(x[1])
    )

def is_valid_variable(x: Any) -> bool:
    return is_tensor(x) or is_tuple_of_tensors(x) or is_nested_variable(x)

def flatten(data: VariableType) -> tuple[ApplyType, SpecType]:
    """
    Flattens VariableType into a flat tuple of tensors plus a spec to reconstruct it.

    Spec encodings:
    - ("flat",)
    - ("tuple", n)
    - ("nested", n_left, n_right)
    """
    if is_tensor(data):
        return (data,), ("flat",)
    
    if is_tuple_of_tensors(data):
        return data, ("tuple", len(data))
    
    if is_nested_variable(data):
        left, right = data
        flat = (*left, *right)
        return flat, ("nested", len(left), len(right))

def unflatten(flat: ApplyType, spec: SpecType) -> VariableType:
    """
    Reconstructs VariableType from a flat tuple of tensors and a spec produced by `flatten`.
    """
    if not is_tuple_of_tensors(flat):
        raise TypeError("`flat` must be a tuple of torch.Tensor.")
    
    if not isinstance(spec, tuple) or len(spec) < 1:
        raise TypeError("`spec` must be a non-empty tuple.")
    
    tag = spec[0]
    if tag == "flat":
        if len(flat) != 1:
            raise ValueError(
                f"Spec {spec} expects exactly 1 tensor, got {len(flat)}."
            )
        return flat[0]
    
    if tag == "tuple":
        if len(spec) != 2 or not isinstance(spec[1], int):
            raise TypeError("Spec ('tuple', n) must have an int n.")
        n = spec[1]
        if len(flat) != n:
            raise ValueError(
                f"Spec {spec} expects {n} tensors, got {len(flat)}."
            )
        return tuple(flat)
    
    if tag == "nested":
        if (
            len(spec) != 3 or not isinstance(spec[1], int) or 
            not isinstance(spec[2], int)
        ):
            raise TypeError(
                "Spec ('nested', n_left, n_right) must have two ints."
            )
        nL, nR = spec[1], spec[2]
        if len(flat) != nL + nR:
            raise ValueError(
                f"Spec {spec} expects {nL+nR} tensors, got {len(flat)}."
            )
        left = tuple(flat[:nL])
        right = tuple(flat[nL:nL + nR])
        return (left, right)


def to_float_scalar(x: ScalarLike) -> float:
    """Convert a scalar-like (float or 0-dim tensor) into a Python float."""
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError("Expected a scalar tensor for metric value.")
        return float(x.detach().item())
    return float(x)