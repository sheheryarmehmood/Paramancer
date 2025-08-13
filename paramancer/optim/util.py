from dataclasses import dataclass
import torch
from typing import Optional, Union
from paramancer.operators.norms import l2

def default_metric(residual):
    try:
        return l2(residual)
    except ArithmeticError:
        return torch.inf


@dataclass
class OptimizationResult:
    solution: Union[torch.Tensor, tuple[torch.Tensor]]
    iterations: int
    metric: Optional[float]
    converged: bool