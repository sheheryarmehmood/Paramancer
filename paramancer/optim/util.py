from __future__ import annotations
from dataclasses import dataclass
import torch

from typing import Optional
from .typing import VariableLike


@dataclass
class OptimizationResult:
    solution: VariableLike
    iterations: int
    metric: Optional[float]
    converged: bool

