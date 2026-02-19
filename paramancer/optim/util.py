from __future__ import annotations
from dataclasses import dataclass
import torch
from typing import Any

from ..variable.types import VariableLike, ScalarLike


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
